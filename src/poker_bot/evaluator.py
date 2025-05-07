from treys import Evaluator, Deck, Card
import itertools
import time

evaluator = Evaluator()

def compute_exact_win_prob(private_cards, public_cards):
    # Extract known cards
    known_cards = private_cards + public_cards
    
    # Create a fresh deck and remove known cards
    deck = Deck()
    for card in known_cards:
        deck.cards.remove(card)
    
    # Track wins, ties, and total games
    wins = 0
    ties = 0
    total = 0
    
    # Iterate over all possible combinations of 1 hidden public card
    hidden_combos = itertools.combinations(deck.cards, 1)
    for hidden_cards in hidden_combos:
        # Create the full public board (4 visible + 1 hidden)
        full_board = public_cards + list(hidden_cards)
        
        # Evaluate the robot's hand once
        my_best = evaluator.evaluate(private_cards, full_board)
        
        # Remaining deck for opponent hands
        remaining_deck = [c for c in deck.cards if c not in hidden_cards]
        
        # Iterate over all possible opponent hands
        for opp_combo in itertools.combinations(remaining_deck, 2):
            # Convert tuple to list
            opp_best = evaluator.evaluate(list(opp_combo), full_board)
            
            if my_best < opp_best:
                wins += 1
            elif my_best == opp_best:
                ties += 1
            
            total += 1
    
    # Calculate exact win probability
    win_prob = (wins + 0.5 * ties) / total
    return win_prob

# Test the function with 10 random cases
test_cases = [
    ([Card.new('2s'), Card.new('3d')], [Card.new('4h'), Card.new('5c'), Card.new('6d'), Card.new('7s')]),
    ([Card.new('7c'), Card.new('8d')], [Card.new('9h'), Card.new('Ts'), Card.new('Js'), Card.new('Qs')]),
    ([Card.new('9c'), Card.new('Tc')], [Card.new('2h'), Card.new('3s'), Card.new('4d'), Card.new('5h')]),
    ([Card.new('2h'), Card.new('2d')], [Card.new('3c'), Card.new('4s'), Card.new('5d'), Card.new('6h')]),
    ([Card.new('3d'), Card.new('4s')], [Card.new('5h'), Card.new('6c'), Card.new('7d'), Card.new('8s')]),
    ([Card.new('4c'), Card.new('5d')], [Card.new('6s'), Card.new('7h'), Card.new('8d'), Card.new('9s')]),
    ([Card.new('5d'), Card.new('6s')], [Card.new('7h'), Card.new('8c'), Card.new('9d'), Card.new('Ts')]),
    ([Card.new('6c'), Card.new('7h')], [Card.new('8d'), Card.new('9s'), Card.new('Ts'), Card.new('Js')]),
    ([Card.new('7d'), Card.new('8s')], [Card.new('9h'), Card.new('Ts'), Card.new('Js'), Card.new('Qs')]),
    ([Card.new('8c'), Card.new('9d')], [Card.new('Ts'), Card.new('Js'), Card.new('Qs'), Card.new('Ks')])
]

total_time = 0
for i, (private_cards, public_cards) in enumerate(test_cases):
    start_time = time.time()
    win_prob = compute_exact_win_prob(private_cards, public_cards)
    end_time = time.time()
    duration = end_time - start_time
    total_time += duration
    print(f"Test {i+1}: Win Probability: {win_prob:.4f}, Time Taken: {duration:.4f} seconds")

average_time = total_time / len(test_cases)
print(f"\nAverage Time Per Test: {average_time:.4f} seconds")
