import random
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from treys import Card, Deck
from poker_bot.structs import PokerRoundState, NonVerbalBehavior

# Define the output file
output_file = "data/easier_poker_hand_exact_strength_25000.pkl"
num_samples = 25000
num_workers = cpu_count()

# Define the function to compute a single hand strength
def compute_hand_strength(_):
    # Create a fresh deck and shuffle
    deck = Deck()
    deck.shuffle()

    # Draw two private cards for the robot
    private_cards = [deck.draw(1)[0], deck.draw(1)[0]]

    # Draw two visible public cards
    visible_public_cards = [deck.draw(1)[0] for _ in range(2)]

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
        private_cards=private_cards,
        visible_public_cards=visible_public_cards,
        opponent_chips=5,
        own_chips=5,
        non_verbal_behavior=non_verbal_behavior
    )

    # Compute the exact hand strength
    hand_strength = round_state.hand_strength()

    # Return the result as a dictionary
    return {
        "private_cards": private_cards,
        "visible_public_cards": visible_public_cards,
        "hand_strength": hand_strength
    }

if __name__ == "__main__":
    print("Generating poker hand strength dataset...")
    with Pool(num_workers) as pool:
        # Use tqdm for a progress bar
        data = list(tqdm(pool.imap(compute_hand_strength, range(num_samples)), total=num_samples))

    # Save the dataset to a pickle file
    with open(output_file, "wb") as f:
        pickle.dump(data, f)

    print(f"Dataset saved to {output_file}")
