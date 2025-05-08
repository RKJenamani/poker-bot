import itertools
import pickle
import random
from tqdm import tqdm
from multiprocessing import Pool
from treys import Deck, Card
from poker_bot.structs import PokerRoundState, NonVerbalBehavior

# Define the output file
output_file = "data/poker_hand_exact_strength_5M.pkl"
num_workers = 16
sample_size = 5000000

# Create a fresh deck
deck = Deck()
all_cards = deck.cards  # 52 cards as 32-bit integers

# Generate 5 million distinct samples
def generate_sample():
    private_hand = tuple(random.sample(all_cards, 2))
    remaining_cards = [c for c in all_cards if c not in private_hand]
    public_board = tuple(random.sample(remaining_cards, 3))
    return private_hand, public_board

print("Generating 5 million distinct samples...")
samples = set()
while len(samples) < sample_size:
    samples.add(generate_sample())

print(f"Total distinct samples generated: {len(samples)}")

# Define the function to compute a single hand strength
def compute_hand_strength(hand_pair):
    private_cards, visible_public_cards = hand_pair

    # Use a placeholder non-verbal behavior (not relevant for hand strength)
    non_verbal_behavior = NonVerbalBehavior(
        gaze_cards_percentage=0.0,
        gaze_robot_percentage=0.0,
        gaze_shifts_per_second=0.0,
        gaze_mean_fixation_duration=0.0,
        head_pose_shifts_rate_per_second=0.0,
        blinks_per_second=0.0
    )

    # Create the PokerRoundState object
    round_state = PokerRoundState(
        private_cards=list(private_cards),
        visible_public_cards=list(visible_public_cards),
        opponent_chips=5,
        own_chips=5,
        non_verbal_behavior=non_verbal_behavior
    )

    # Compute the exact hand strength
    hand_strength = round_state.hand_strength()

    # Return the result as a dictionary
    return {
        "private_cards": list(private_cards),
        "visible_public_cards": list(visible_public_cards),
        "hand_strength": hand_strength
    }

if __name__ == "__main__":
    print("Computing hand strengths...")
    with Pool(num_workers) as pool:
        # Use tqdm for a progress bar
        data = list(tqdm(pool.imap(compute_hand_strength, samples), total=len(samples)))

    # Save the dataset to a pickle file
    with open(output_file, "wb") as f:
        pickle.dump(data, f)

    print(f"Dataset saved to {output_file}")