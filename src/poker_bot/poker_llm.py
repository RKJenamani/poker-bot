import os
import json
from pathlib import Path
from typing import List
from poker_bot.structs import PokerRoundState, PokerRoundOutcome
from tomsutils.llm import OpenAILLM

class PokerLLM:

    def __init__(self):
        # Load the prompt skeleton
        with open(Path(__file__).parent / "llm_prompt.txt", 'r') as f:
            self.prompt_skeleton = f.read()

        # Initialize the LLM wrapper
        self.llm = OpenAILLM(model_name="gpt-4o", cache_dir=Path(__file__).parent / "llm_cache")

    def get_decision(self, current_round: PokerRoundState, round_history: List[PokerRoundOutcome]) -> dict:
        # Format the current prompt
        print("Formatting context...")
        prompt = self.format_context(current_round, round_history)
        print("Context formatted.")

        # Generate a decision using the LLM
        print("Generating decision...")
        response = self.llm.sample_completions(prompt, imgs=None, temperature=0.0, seed=0)[0][0]
        print("Decision generated.")
        # Parse the response into a Python dictionary
        return self.parse_response(response)

    def format_context(self, current_round: PokerRoundState, round_history: List[PokerRoundOutcome]) -> str:
        # Use the __str__ methods directly for concise formatting
        history_str = "\n---\n".join(map(str, round_history))

        # Build the full prompt using the current round's str
        prompt = self.prompt_skeleton % (
            history_str.strip(),
            # f"{current_round.hand_strength():.4f}",
            f"{current_round.hand_strength_monte_carlo(convergence_threshold=0.001):.4f}",
            # f"{current_round.hand_strength_approx():.4f}", # Uncomment if using approximate strength estimated using a neural network
            str(current_round).strip()
        )
        return prompt

    def parse_response(self, response: str) -> dict:
        # Remove code block markers if present
        if response.startswith("```python") and response.endswith("```"):
            response = response[9:-3].strip()
        
        # Attempt to parse the cleaned JSON
        try:
            parsed_response = json.loads(response.strip())
            
            # Ensure the response is a dictionary with the expected structure
            if not isinstance(parsed_response, dict):
                raise ValueError(f"Expected a dictionary, got {type(parsed_response)}")
            if "go_all_in" not in parsed_response:
                raise ValueError("Missing 'go_all_in' key in response")
            
            return parsed_response["go_all_in"]
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response: {response}") from e
