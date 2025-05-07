# PokerBot

This repository contains the code for a two-player poker bot that combines supervised learning and large language models (LLMs) to estimate hand strength and personalize gameplay based on non-verbal behavior. Our version of the poker game is designed for rapid rounds and real-time adaptation to human behavior.

### Game Rules

This is a two-player card game between a robot (you) and a human opponent. The game follows these rules:

* Each player starts with 3 chips and is dealt two private cards.
* Five public cards are available, four face-up and one face-down on the table.

1. **Entry Cost**: Each round requires one chip to enter.
2. **Opponent’s Turn**: The human opponent acts first, choosing one of two actions:

   * **Fold**: The opponent exits the round, and you win the two chips on the table.
   * **All-In**: The opponent commits all chips, signaling a strong hand or a potential bluff.
3. **Your Turn**: After observing the opponent’s action and non-verbal behavior, you choose one of two actions:

   * **Fold**: You exit, conceding the two chips to the opponent.
   * **All-In**: You commit all your chips.
4. **Showdown**: If both players go all-in, the player with the stronger 7-card hand (two private cards + five public cards) wins the pot.
5. **Cards Reveal**: At the end of each round, all seven cards (two private cards for each player + five public cards) are revealed, regardless of whether the round ended in a fold or all-in. This consistent reveal allows for continuous learning of the relationship between non-verbal cues and actual card strength.

## Hand Strength Estimation

The bot estimates hand strength using a supervised learning approach, where the win rate is predicted based on the current hand's two private cards and three visible public cards. This is implemented using a Multi-Layer Perceptron (MLP) trained on a dataset of one million poker hands (TODO: add more info). 

## LLM Personalization

The bot uses an LLM to refine its decision-making based on observed opponent play styles and non-verbal behaviors over multiple rounds. The LLM is prompted with summaries of past rounds, including non-verbal behaviors like gaze patterns, head movements, and blinking rates, to generate strategic insights for the current round.

## Setup

```bash
git clone https://github.com/your-username/poker-bot.git
cd poker-bot
pip install -e ".[develop]"
```

## Running the Poker Bot

Work in progress, please use [tests/test_poker_llm.py](tests/test_poker_llm.py) as an entrypoint. 

## Acknowledgements

Special thanks to the developers of [treys](https://github.com/ihendley/treys) for the efficient poker hand evaluation engine used in this project.
