import unittest
from treys import Card
from poker_bot.structs import PokerRoundState, PokerRoundOutcome, NonVerbalBehavior
from poker_bot.poker_llm import PokerLLM


class TestPokerLLM(unittest.TestCase):

    def setUp(self):
        self.poker_llm = PokerLLM()

        self.non_verbal_behavior = NonVerbalBehavior(
            gaze_cards_percentage=60.0,
            gaze_robot_percentage=15.0,
            gaze_shifts_per_second=8.0,
            gaze_mean_fixation_duration=0.5,
            head_pose_shifts_rate_per_second=5.0,
            blinks_per_second=12.0
        )

        # Create a current round state
        self.current_round = PokerRoundState(
            private_cards=[Card.new('As'), Card.new('Ks')],
            visible_public_cards=[Card.new('Qd'), Card.new('Jh'), Card.new('Ts')],
            opponent_chips=5,
            own_chips=5,
            non_verbal_behavior=self.non_verbal_behavior
        )

        # Create a round history with a mix of outcomes
        self.round_history = [
            PokerRoundOutcome(
                private_cards=[Card.new('Ac'), Card.new('Kd')],
                visible_public_cards=[Card.new('Qc'), Card.new('Jd'), Card.new('Td')],
                opponent_private_cards=[Card.new('2h'), Card.new('3d')],
                hidden_public_cards=[Card.new('8d'), Card.new('9c')],
                opponent_chips=5,
                own_chips=5,
                non_verbal_behavior=self.non_verbal_behavior,
                opponent_all_in=True,
                robot_all_in=True,
                would_robot_win=True,
                did_robot_win=True
            ),
            PokerRoundOutcome(
                private_cards=[Card.new('2s'), Card.new('3c')],
                visible_public_cards=[Card.new('4h'), Card.new('5s'), Card.new('6d')],
                opponent_private_cards=[Card.new('8h'), Card.new('9d')],
                hidden_public_cards=[Card.new('Td'), Card.new('Jc')],
                opponent_chips=5,
                own_chips=5,
                non_verbal_behavior=self.non_verbal_behavior,
                opponent_all_in=True,
                robot_all_in=True,
                would_robot_win=True,
                did_robot_win=True
            ),
            PokerRoundOutcome(
                private_cards=[Card.new('4c'), Card.new('5d')],
                visible_public_cards=[Card.new('6h'), Card.new('7s'), Card.new('8c')],
                opponent_private_cards=[Card.new('Th'), Card.new('Jd')],
                hidden_public_cards=[Card.new('Qd'), Card.new('Kd')],
                opponent_chips=4,
                own_chips=6,
                non_verbal_behavior=self.non_verbal_behavior,
                opponent_all_in=True,
                robot_all_in=True,
                would_robot_win=True,
                did_robot_win=True
            ),
            PokerRoundOutcome(
                private_cards=[Card.new('6c'), Card.new('7d')],
                visible_public_cards=[Card.new('8h'), Card.new('9s'), Card.new('Ts')],
                opponent_private_cards=[Card.new('Qc'), Card.new('Kd')],
                hidden_public_cards=[Card.new('Ah'), Card.new('2d')],
                opponent_chips=7,
                own_chips=3,
                non_verbal_behavior=self.non_verbal_behavior,
                opponent_all_in=True,
                robot_all_in=False,
                would_robot_win=True,
                did_robot_win=False
            )
        ]

    def test_poker_llm_decision(self):
        # Run the LLM to make a decision
        go_all_in = self.poker_llm.get_decision(self.current_round, self.round_history)

        # Basic sanity check
        self.assertIn(go_all_in, [True, False], "Decision should be either True (all-in) or False (fold)")

if __name__ == "__main__":
    unittest.main()
