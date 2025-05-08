import itertools
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from treys import Deck, Card
from poker_bot.structs import PokerRoundState, NonVerbalBehavior

# Define the output file
output_file = "data/all_easier_poker_hand_exact_strength.pkl"
num_workers = 16

# Create a fresh deck
deck = Deck()
all_cards = deck.cards  # 52 cards as 32-bit integers

# Generate all possible 2-card private hands
all_private_hands = list(itertools.combinations(all_cards, 2))

# Generate all possible 2-card public boards (excluding private hand cards)
def generate_public_boards(private_hand):
    private_set = set(private_hand)
    remaining_cards = [c for c in all_cards if c not in private_set]
    return [(private_hand, public) for public in itertools.combinations(remaining_cards, 2)]

# Prepare the full list of (private, public) combinations
all_hands = []
for private_hand in tqdm(all_private_hands, desc="Generating All Possible Hands"):
    all_hands.extend(generate_public_boards(private_hand))

print(f"Total possible observations: {len(all_hands)}")

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
    print("Generating poker hand strength dataset...")
    with Pool(num_workers) as pool:
        # Use tqdm for a progress bar
        data = list(tqdm(pool.imap(compute_hand_strength, all_hands), total=len(all_hands)))

    # Save the dataset to a pickle file
    with open(output_file, "wb") as f:
        pickle.dump(data, f)

    print(f"Dataset saved to {output_file}")
