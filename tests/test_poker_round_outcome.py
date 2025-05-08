import unittest
from treys import Card
from poker_bot.structs import NonVerbalBehavior, PokerRoundOutcome

class TestPokerRoundOutcome(unittest.TestCase):

    def setUp(self):
        self.non_verbal_behavior = NonVerbalBehavior(
            gaze_cards_percentage=60.0,
            gaze_robot_percentage=15.0,
            gaze_shifts_per_second=8.0,
            gaze_mean_fixation_duration=0.5,
            head_pose_shifts_rate_per_second=5.0,
            blinks_per_second=12.0
        )

    def test_both_all_in_robot_wins(self):
        # Robot should win in this all-in scenario
        outcome = PokerRoundOutcome(
            private_cards=[Card.new('As'), Card.new('Ks')],
            visible_public_cards=[Card.new('Qd'), Card.new('Jh'), Card.new('Ts')],
            opponent_private_cards=[Card.new('2h'), Card.new('3d')],
            hidden_public_cards=[Card.new('8d'), Card.new('9c')],
            opponent_chips=5,
            own_chips=5,
            non_verbal_behavior=self.non_verbal_behavior,
            opponent_all_in=True,
            robot_all_in=True,
            would_robot_win=True,
            did_robot_win=True
        )

    def test_both_all_in_robot_loses(self):
        # Robot should lose in this all-in scenario
        outcome = PokerRoundOutcome(
            private_cards=[Card.new('2h'), Card.new('3d')],
            visible_public_cards=[Card.new('Qd'), Card.new('Jh'), Card.new('Ts')],
            opponent_private_cards=[Card.new('As'), Card.new('Ks')],
            hidden_public_cards=[Card.new('8d'), Card.new('9c')],
            opponent_chips=5,
            own_chips=5,
            non_verbal_behavior=self.non_verbal_behavior,
            opponent_all_in=True,
            robot_all_in=True,
            would_robot_win=False,
            did_robot_win=False
        )

    def test_opponent_fold(self):
        # Opponent folded, so robot wins by default
        outcome = PokerRoundOutcome(
            private_cards=[Card.new('As'), Card.new('Ks')],
            visible_public_cards=[Card.new('Qd'), Card.new('Jh'), Card.new('Ts')],
            opponent_private_cards=[Card.new('2h'), Card.new('3d')],
            hidden_public_cards=[Card.new('8d'), Card.new('9c')],
            opponent_chips=0,
            own_chips=10,
            non_verbal_behavior=self.non_verbal_behavior,
            opponent_all_in=False,
            robot_all_in=True,
            would_robot_win=True,
            did_robot_win=True
        )

    def test_robot_fold(self):
        # Robot folded, so opponent wins
        with self.assertRaises(ValueError):
            PokerRoundOutcome(
                private_cards=[Card.new('As'), Card.new('Ks')],
                visible_public_cards=[Card.new('Qd'), Card.new('Jh'), Card.new('Ts')],
                opponent_private_cards=[Card.new('2h'), Card.new('3d')],
                hidden_public_cards=[Card.new('8d'), Card.new('9c')],
                opponent_chips=10,
                own_chips=0,
                non_verbal_behavior=self.non_verbal_behavior,
                opponent_all_in=True,
                robot_all_in=False,
                would_robot_win=False,
                did_robot_win=True  # This should raise an error
            )

    def test_invalid_chip_distribution(self):
        # Invalid chip distribution
        with self.assertRaises(AssertionError):
            PokerRoundOutcome(
                private_cards=[Card.new('As'), Card.new('Ks')],
                visible_public_cards=[Card.new('Qd'), Card.new('Jh'), Card.new('Ts')],
                opponent_private_cards=[Card.new('2h'), Card.new('3d')],
                hidden_public_cards=[Card.new('8d'), Card.new('9c')],
                opponent_chips=4,
                own_chips=5,
                non_verbal_behavior=self.non_verbal_behavior,
                opponent_all_in=True,
                robot_all_in=True,
                would_robot_win=True,
                did_robot_win=True
            )

if __name__ == "__main__":
    unittest.main()
