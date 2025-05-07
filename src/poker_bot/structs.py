from dataclasses import dataclass, field
from typing import List
from pathlib import Path
from treys import Card, Deck, Evaluator
import itertools
import torch

from poker_bot.hand_strength_estimator.model import PokerMLP, load_model, independent_one_hot_encode

evaluator = Evaluator()
hand_strength_model = load_model(Path(__file__).parent / "hand_strength_estimator" / "models" / "poker_mlp.pth")
hand_strength_model.eval()

@dataclass(frozen=True)
class NonVerbalBehavior:
    gaze_cards_percentage: float
    gaze_robot_percentage: float
    gaze_shifts_per_second: float
    gaze_mean_fixation_duration: float
    head_pose_shifts_rate_per_second: float
    blinks_per_second: float

    def __str__(self):
        return (
            f"Gaze at Cards (%): {self.gaze_cards_percentage:.1f}, "
            f"Gaze at Robot (%): {self.gaze_robot_percentage:.1f}, "
            f"Gaze Shifts (/s): {self.gaze_shifts_per_second:.1f}, "
            f"Mean Fixation Duration (s): {self.gaze_mean_fixation_duration:.2f}, "
            f"Head Pose Shifts (/s): {self.head_pose_shifts_rate_per_second:.1f}, "
            f"Blinks (/s): {self.blinks_per_second:.1f}"
        )

@dataclass(frozen=True)
class PokerRoundState:
    private_cards: List[int]
    visible_public_cards: List[int]
    opponent_chips: int
    own_chips: int
    non_verbal_behavior: NonVerbalBehavior

    def __post_init__(self):
        # Ensure valid chip distribution
        assert self.opponent_chips + self.own_chips == 10, \
            f"Total number of chips must be 10, got {self.opponent_chips + self.own_chips}"

    def hand_strength(self) -> float:
        """Compute the exact win probability for the current hand."""
        known_cards = self.private_cards + self.visible_public_cards
        
        # Create a fresh deck and remove known cards
        deck = Deck()
        for card in known_cards:
            deck.cards.remove(card)
        
        wins, ties, total = 0, 0, 0
        
        # Iterate over all possible combinations of 1 hidden public card
        for hidden_card in deck.cards:
            full_board = self.visible_public_cards + [hidden_card]
            my_best = evaluator.evaluate(self.private_cards, full_board)
            remaining_deck = [c for c in deck.cards if c != hidden_card]

            for opp_combo in itertools.combinations(remaining_deck, 2):
                opp_best = evaluator.evaluate(list(opp_combo), full_board)
                if my_best < opp_best:
                    wins += 1
                elif my_best == opp_best:
                    ties += 1
                total += 1
        
        return (wins + 0.5 * ties) / total
    
    def hand_strength_approx(self) -> float:
        """Compute the approximate win probability using a neural network."""
        # Prepare the input for the model
        input_data = independent_one_hot_encode(self.private_cards + self.visible_public_cards)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        
        # Forward pass through the model
        with torch.no_grad():
            output = hand_strength_model(input_tensor)
        
        # Extract the win probability
        win_prob = output.item()
        
        return win_prob

    def __str__(self):
        private = " ".join([Card.int_to_pretty_str(c) for c in self.private_cards])
        visible_public = " ".join([Card.int_to_pretty_str(c) for c in self.visible_public_cards])
        
        return (
            f"Private Cards: {private}\n"
            f"Visible Public Cards: {visible_public}\n"
            f"Opponent Chips: {self.opponent_chips}\n"
            f"Own Chips: {self.own_chips}\n"
            f"Non-Verbal Behavior: {self.non_verbal_behavior}\n"
        )

@dataclass(frozen=True)
class PokerRoundOutcome(PokerRoundState):
    opponent_private_cards: List[int]
    hidden_public_card: int
    opponent_all_in: bool
    robot_all_in: bool
    would_robot_win: bool
    did_robot_win: bool

    def __post_init__(self):
        super().__post_init__()

        # Compute the correct would_robot_win value
        all_public_cards = self.visible_public_cards + [self.hidden_public_card]
        robot_best = evaluator.evaluate(self.private_cards, all_public_cards)
        opponent_best = evaluator.evaluate(self.opponent_private_cards, all_public_cards)
        computed_would_robot_win = robot_best < opponent_best
        computed_did_robot_win = not self.opponent_all_in or (self.robot_all_in and computed_would_robot_win)

        # Validate against the provided values
        if computed_would_robot_win != self.would_robot_win:
            raise ValueError(f"Inconsistent would_robot_win: Expected {computed_would_robot_win}, got {self.would_robot_win}")

        if computed_did_robot_win != self.did_robot_win:
            raise ValueError(f"Inconsistent did_robot_win: Expected {computed_did_robot_win}, got {self.did_robot_win}")
    
    def __str__(self):
        # Use the parent class string as the base
        base_str = super().__str__()
        opponent_private = " ".join([Card.int_to_pretty_str(c) for c in self.opponent_private_cards])
        hidden_public = Card.int_to_pretty_str(self.hidden_public_card)
        
        return (
            f"{base_str.strip()}\n"
            f"Opponent Private Cards: {opponent_private}\n"
            f"Hidden Public Card: {hidden_public}\n"
            f"Opponent All-In: {self.opponent_all_in}\n"
            f"Robot All-In: {self.robot_all_in}\n"
            f"Would Robot Win (Given Cards): {self.would_robot_win}\n"
            f"Did Robot Win (Actual Outcome): {self.did_robot_win}\n"
        )
