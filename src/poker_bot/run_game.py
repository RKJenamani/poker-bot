import json
from pathlib import Path
from typing import List, Tuple
from treys import Card, Deck, Evaluator
from poker_bot.structs import NonVerbalBehavior, PokerRoundState, PokerRoundOutcome, evaluator
from poker_bot.poker_llm import PokerLLM

def load_nonverbal_behavior(json_path: Path) -> NonVerbalBehavior:
    """Load nonverbal behavior data from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return NonVerbalBehavior(
        gaze_cards_percentage=float(data['gaze_cards_percentage']),
        gaze_robot_percentage=float(data['gaze_robot_percentage']),
        gaze_shifts_per_second=float(data['gaze_shifts_per_second']),
        gaze_mean_fixation_duration=float(data['gaze_mean_fixation_duration']),
        head_pose_shifts_rate_per_second=float(data['head_pose_shifts_rate_per_second']),
        blinks_per_second=float(data['blinks_per_second'])
    )

def get_card_input(prompt: str) -> List[int]:
    """Get card input from user in format 'As Kd' etc."""
    while True:
        try:
            cards_str = input(prompt).strip()
            cards = cards_str.split()
            if len(cards) != 2:
                print("Please enter exactly 2 cards")
                continue
            print(cards)
            return [Card.new(card) for card in cards]
        except ValueError:
            print("Invalid card format. Please use format like 'As Kd'")

def get_visible_cards_input() -> List[int]:
    """Get visible public cards input from user."""
    while True:
        try:
            cards_str = input("Enter visible public cards (e.g., 'Qd Jh Ts'): ").strip()
            cards = cards_str.split()
            if len(cards) != 3:
                print("Please enter exactly 3 cards")
                continue
            return [Card.new(card) for card in cards]
        except ValueError:
            print("Invalid card format. Please use format like 'Qd Jh Ts'")

def get_hidden_cards_input() -> List[int]:
    """Get hidden public cards input from user."""
    while True:
        try:
            cards_str = input("Enter hidden public cards (e.g., '8d 9c'): ").strip()
            cards = cards_str.split()
            if len(cards) != 2:
                print("Please enter exactly 2 cards")
                continue
            return [Card.new(card) for card in cards]
        except ValueError:
            print("Invalid card format. Please use format like '8d 9c'")

def get_opponent_action() -> bool:
    """Get opponent's action (True for all-in, False for fold)."""
    while True:
        action = input("Did opponent go all-in? (y/n): ").lower()
        if action in ['y', 'yes']:
            return True
        elif action in ['n', 'no']:
            return False
        print("Please enter 'y' or 'n'")

def run_game(nonverbal_data_path: Path):
    """Run the poker game loop."""
    # Initialize game state
    opponent_chips = 3
    robot_chips = 3
    round_history: List[PokerRoundOutcome] = []
    poker_llm = PokerLLM()
    
    while True:
        print("\n=== New Round ===")
        print(f"Opponent chips: {opponent_chips}")
        print(f"Robot chips: {robot_chips}")
        
        # Load nonverbal behavior data
        nonverbal_behavior = load_nonverbal_behavior(nonverbal_data_path)
        
        # Get card inputs from human helper
        print("\nEnter robot's private cards:")
        robot_private_cards = get_card_input("Enter robot's cards (e.g., 'As Kd'): ")
        
        print("\nEnter visible public cards:")
        visible_public_cards = get_visible_cards_input()
        
        # Create current round state
        current_round = PokerRoundState(
            private_cards=robot_private_cards,
            visible_public_cards=visible_public_cards,
            opponent_chips=opponent_chips,
            own_chips=robot_chips,
            non_verbal_behavior=nonverbal_behavior
        )
        
        # Get opponent's action
        opponent_all_in = get_opponent_action()
        
        # Get robot's decision using LLM
        print("Querying LLM...")
        robot_all_in = poker_llm.get_decision(current_round, round_history)
        print(f"\nRobot decides to: {'ALL-IN' if robot_all_in else 'FOLD'}")
        
        # Get hidden cards and opponent's cards for outcome
        print("\nEnter hidden public cards:")
        hidden_public_cards = get_hidden_cards_input()
        
        print("\nEnter opponent's private cards:")
        opponent_private_cards = get_card_input("Enter opponent's cards (e.g., 'As Kd'): ")
        
        # Determine who would win
        all_public_cards = visible_public_cards + hidden_public_cards
        robot_best = evaluator.evaluate(robot_private_cards, all_public_cards)
        opponent_best = evaluator.evaluate(opponent_private_cards, all_public_cards)
        would_robot_win = robot_best < opponent_best
        
        # Determine actual winner
        did_robot_win = not opponent_all_in or (robot_all_in and would_robot_win)
        
        # Update chip counts
        if did_robot_win:
            robot_chips += opponent_chips if opponent_all_in else 1
            opponent_chips = 0 if opponent_all_in else opponent_chips - 1
        else:
            opponent_chips += robot_chips if robot_all_in else 1
            robot_chips = 0 if robot_all_in else robot_chips - 1
        
        # Create and store round outcome
        outcome = PokerRoundOutcome(
            private_cards=robot_private_cards,
            visible_public_cards=visible_public_cards,
            opponent_private_cards=opponent_private_cards,
            hidden_public_cards=hidden_public_cards,
            opponent_chips=opponent_chips,
            own_chips=robot_chips,
            non_verbal_behavior=nonverbal_behavior,
            opponent_all_in=opponent_all_in,
            robot_all_in=robot_all_in,
            would_robot_win=would_robot_win,
            did_robot_win=did_robot_win
        )
        round_history.append(outcome)
        
        # Print round outcome
        print("\n=== Round Outcome ===")
        print(outcome)
        
        # Check if game should reset
        if opponent_chips == 0 or robot_chips == 0:
            print("\n=== Game Reset ===")
            opponent_chips = 3
            robot_chips = 3
        
        # Ask if continue
        if input("\nContinue to next round? (y/n): ").lower() not in ['y', 'yes']:
            break

if __name__ == "__main__":
    nonverbal_data_path = Path("data/nonverbal_behavior.json")
    run_game(nonverbal_data_path) 