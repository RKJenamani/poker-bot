import random
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from treys import Card, Deck
from poker_bot.structs import PokerRoundState, NonVerbalBehavior, evaluator

# Define the input and output files
train_file = "data/easier_poker_hand_estimated_strength_1M.pkl"
test_file = "data/easier_poker_hand_exact_strength_25000.pkl"
num_samples = 1000000  # Number of samples to generate
batch_size = 100  # Number of opponent samples per batch
convergence_threshold = 0.01
num_workers = cpu_count()

# Load existing test samples to avoid duplicates in the training data
with open(test_file, "rb") as f:
    test_samples = pickle.load(f)

test_pairs = set(
    (tuple(sorted(d["private_cards"])), tuple(sorted(d["visible_public_cards"])))
    for d in test_samples
)

print(f"Loaded {len(test_pairs)} existing test samples.")

# Function to compute hand strength with incremental convergence
# Function to compute hand strength with incremental convergence
def compute_hand_strength(_):
    while True:
        # Create a fresh deck and shuffle
        deck = Deck()
        deck.shuffle()

        # Draw 2 private cards
        private_cards = sorted([deck.draw(1)[0], deck.draw(1)[0]])

        # Draw 2 visible public cards
        visible_public_cards = sorted([deck.draw(1)[0] for _ in range(2)])

        # Check if this combination already exists in the test set
        key = (tuple(private_cards), tuple(visible_public_cards))
        if key in test_pairs:
            continue  # Skip this sample if it already exists
        else:
            break

    # Prepare the remaining deck for opponent sampling
    known_cards = set(private_cards + visible_public_cards)
    remaining_cards = [c for c in Deck().cards if c not in known_cards]

    # Incrementally estimate hand strength
    total_samples = 0
    wins = 0
    ties = 0
    prev_win_rate = -1  # Set to an invalid initial value

    while True:
        # Sample batches of opponent hands + hidden card (with replacement)
        for _ in range(batch_size):
            other_cards = random.sample(remaining_cards, 3)
            hidden_card = other_cards[0]
            opponent_private_cards = other_cards[1:]
            
            # Compare hands directly
            full_board = visible_public_cards + [hidden_card]
            my_best = evaluator.evaluate(private_cards, full_board)
            opp_best = evaluator.evaluate(opponent_private_cards, full_board)

            if my_best < opp_best:
                wins += 1
            elif my_best == opp_best:
                ties += 1

            total_samples += 1
        
        # Calculate the current win rate including ties
        current_win_rate = (wins + 0.5 * ties) / total_samples
        
        # Check for convergence
        if prev_win_rate != -1 and abs(current_win_rate - prev_win_rate) < convergence_threshold:
            break
        
        prev_win_rate = current_win_rate

    # Return the final result
    return {
        "private_cards": private_cards,
        "visible_public_cards": visible_public_cards,
        "estimated_hand_strength": current_win_rate,
        "total_samples": total_samples
    }

if __name__ == "__main__":
    print("Generating poker hand strength training dataset with incremental convergence...")
    with Pool(num_workers) as pool:
        # Use tqdm for a progress bar
        data = list(tqdm(pool.imap(compute_hand_strength, range(num_samples)), total=num_samples))

    # Save the dataset to a pickle file
    with open(train_file, "wb") as f:
        pickle.dump(data, f)

    # Calculate statistics for the total samples
    sample_counts = [d["total_samples"] for d in data]
    max_samples = max(sample_counts)
    avg_samples = np.mean(sample_counts)
    std_dev_samples = np.std(sample_counts)

    print(f"\nDataset saved to {train_file}")
    print(f"Max Samples for a Datapoint: {max_samples}")
    print(f"Average Samples per Datapoint: {avg_samples:.2f}")
    print(f"Standard Deviation of Samples: {std_dev_samples:.2f}")
