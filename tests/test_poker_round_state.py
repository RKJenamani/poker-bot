import unittest
import time
from treys import Card
from poker_bot.structs import NonVerbalBehavior, PokerRoundState


class TestPokerRoundState(unittest.TestCase):

    def setUp(self):
        self.non_verbal_behavior = NonVerbalBehavior(
            gaze_cards_percentage=60.0,
            gaze_robot_percentage=15.0,
            gaze_shifts_per_second=8.0,
            gaze_mean_fixation_duration=0.5,
            head_pose_shifts_rate_per_second=5.0,
            blinks_per_second=12.0
        )

        # Test cases with known private and public cards
        self.test_cases = [
            ([Card.new('2s'), Card.new('3d')], [Card.new('4h'), Card.new('5c'), Card.new('6d')]),
            ([Card.new('7c'), Card.new('8d')], [Card.new('9h'), Card.new('Ts'), Card.new('Js')]),
            ([Card.new('9c'), Card.new('Tc')], [Card.new('2h'), Card.new('3s'), Card.new('4d')]),
            ([Card.new('2h'), Card.new('2d')], [Card.new('3c'), Card.new('4s'), Card.new('5d')]),
            ([Card.new('3d'), Card.new('4s')], [Card.new('5h'), Card.new('6c'), Card.new('7d')]),
            ([Card.new('4c'), Card.new('5d')], [Card.new('6s'), Card.new('7h'), Card.new('8d')]),
            ([Card.new('5d'), Card.new('6s')], [Card.new('7h'), Card.new('8c'), Card.new('9d')]),
            ([Card.new('6c'), Card.new('7h')], [Card.new('8d'), Card.new('9s'), Card.new('Ts')]),
            ([Card.new('7d'), Card.new('8s')], [Card.new('9h'), Card.new('Ts'), Card.new('Js')]),
            ([Card.new('8c'), Card.new('9d')], [Card.new('Ts'), Card.new('Js'), Card.new('Qs')])
        ]

    def test_hand_strength_computation(self):
        total_time = 0
        estimated_total_time = 0
        for i, (private_cards, visible_public_cards) in enumerate(self.test_cases):
            # Create a PokerRoundState instance
            round_state = PokerRoundState(
                private_cards=private_cards,
                visible_public_cards=visible_public_cards,
                opponent_chips=3,
                own_chips=3,
                non_verbal_behavior=self.non_verbal_behavior
            )
            
            # Measure hand strength computation time
            start_time = time.time()
            win_prob = round_state.hand_strength()
            end_time = time.time()
            duration = end_time - start_time
            total_time += duration
            
            # Print the results for each test case
            print(f"Test {i+1}: {round_state}\nWin Probability: {win_prob:.4f}, Time Taken: {duration:.4f} seconds")

            # Measure estimated hand strength with computation time
            start_time = time.time()
            convergence_threshold = 0.001
            estimated_win_prob = round_state.hand_strength_monte_carlo(convergence_threshold=convergence_threshold)
            end_time = time.time()
            estimated_duration = end_time - start_time
            estimated_time = estimated_duration / 1000
            estimated_total_time += estimated_time

            print(f"Estimated Win Probability: {estimated_win_prob:.4f}, Estimated Time Taken: {estimated_time:.4f} seconds")

            assert abs(estimated_win_prob - win_prob) < 0.05, \
                f"Test {i+1} failed: Expected {win_prob} but got {estimated_win_prob}"

            # Ensure the win probability is within the valid range
            self.assertGreaterEqual(win_prob, 0.0)
            self.assertLessEqual(win_prob, 1.0)

            # Ensure the estimated win probability is within the valid range
            self.assertGreaterEqual(estimated_win_prob, 0.0)
            self.assertLessEqual(estimated_win_prob, 1.0)

        # Print average time for all test cases
        average_time = total_time / len(self.test_cases)
        print(f"\nAverage Time Per Test: {average_time:.4f} seconds")

        # Print estimated average time for all test cases
        estimated_average_time = estimated_total_time / len(self.test_cases)
        print(f"Estimated Average Time Per Test: {estimated_average_time:.4f} seconds")

if __name__ == "__main__":
    unittest.main()
