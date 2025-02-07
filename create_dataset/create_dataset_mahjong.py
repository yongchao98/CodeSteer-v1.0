import random
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Set, Union
import string
import re
from collections import Counter

@dataclass
class MahjongPuzzle:
    initial_cards: str
    rounds: List[Dict[str, str]]  # List of rounds with add/remove cards
    solution: int
    complexity: int  # 1-5 scale

class MahjongPuzzleGenerator:
    def __init__(self):
        self.valid_cards = string.ascii_uppercase
        
    def check_peng(self, cards: str, new_card: str) -> bool:
        """Check if a Peng pattern exists with the new card"""
        # Count including the new card
        card_count = cards.count(new_card) + 1
        return card_count >= 3

    def check_chi(self, cards: str, new_card: str) -> bool:
        """Check if a Chi pattern exists with the new card"""
        # Convert to list and sort for easier sequence checking
        all_cards = sorted(list(cards + new_card))
        
        # Check all possible consecutive sequences
        for i in range(len(all_cards) - 2):
            # Get three consecutive cards
            seq = all_cards[i:i+3]
            # Check if they form a sequence
            if (ord(seq[1]) == ord(seq[0]) + 1 and 
                ord(seq[2]) == ord(seq[1]) + 1 and
                new_card in seq):
                return True
        return False

    def determine_result(self, cards: str, new_card: str) -> int:
        """Determine the result based on current cards and new card"""
        if self.check_peng(cards, new_card):
            return 1
        elif self.check_chi(cards, new_card):
            return 2
        return 0

    def generate_random_cards(self, num_cards: int) -> str:
        # """Generate random initial cards"""
        # return ''.join(random.choices(self.valid_cards, k=num_cards))
        while True:
            # Create a list of available letters with each letter appearing at most twice
            available_letters = []
            for letter in self.valid_cards:
                available_letters.extend([letter] * 2)  # Each letter can appear up to twice
                
            # Randomly sample from available letters
            selected_cards = random.sample(available_letters, num_cards)
            
            # Verify no card appears more than twice
            card_counts = Counter(selected_cards)
            if all(count <= 2 for count in card_counts.values()):
                return ''.join(selected_cards)

    def generate_puzzle(self, complexity: int) -> MahjongPuzzle:
        """Generate a puzzle with specified complexity"""
        # Higher complexity means more rounds and Chi patterns
        allow_chi = complexity > 2
        num_rounds = complexity
        
        # Generate initial cards
        initial_cards = self.generate_random_cards(13)
        
        # Generate rounds
        rounds = []
        current_cards = initial_cards
        final_result = None
        
        for _ in range(num_rounds):
            new_card = random.choice(self.valid_cards)
            # Generate new card that might create interesting patterns
            if complexity > 1:  # complexity 1 2 3 4 
                # Try to create a pattern
                if current_cards.count(random.choice(current_cards)) >= 2:
                    # Create Peng
                    new_card = random.choice([c for c in current_cards if current_cards.count(c) >= 2])
                else:
                    # Try to create Chi
                    card_list = sorted(list(current_cards))
                    if allow_chi and len(card_list) >= 2:
                        idx = random.randint(0, len(card_list) - 2)
                        if ord(card_list[idx+1]) == ord(card_list[idx]) + 1:
                            new_card = chr(ord(card_list[idx+1]) + 1)
            else:
                new_card = random.choice(self.valid_cards)
            
            # Choose random card to remove
            remove_card = random.choice(current_cards)
            
            # Calculate result before removing card
            final_result = self.determine_result(current_cards, new_card)
            
            rounds.append({
                'add': new_card,
                'remove': remove_card
            })
            
            # Update current cards
            current_cards = current_cards.replace(remove_card, '', 1) + new_card
        
        return MahjongPuzzle(
            initial_cards=initial_cards,
            rounds=rounds,
            solution=final_result,
            complexity=complexity
        )

def generate_dataset(output_dir: str):
    """Generate dataset with varying complexity"""
    os.makedirs(output_dir, exist_ok=True)
    generator = MahjongPuzzleGenerator()
    
    configurations = [
        (1, 4),
        (2, 4),
        (3, 6),
        (4, 6),
        (5, 80),
        (1, 4),
        (2, 4),
        (3, 6),
        (4, 6),
        (5, 80),
    ]
    
    sample_id = 0
    for complexity, num_samples in configurations:
        for _ in range(num_samples):
            puzzle = generator.generate_puzzle(complexity)
            
            sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Format question
            rules = []
            rules.append("1. Initially, there are 13 cards.")
            rules.append("2. Each time, a new card is added, and a result is determined. Then, one card is removed.")
            rules.append("3. When there are two identical cards in hand, and the newly added card is the same as these two cards, the result is determined as \"Peng,\" represented by the number 1.")
            if complexity > 2:
                rules.append("4. If two cards in hand and the new card can form a consecutive letter sequence, the result is determined as \"Chi,\" represented by the number 2. For example, ABC, OPQ, etc.")
                rules.append("5. If the new card does not meet the conditions of 3 and 4, the result is determined as \"Pass,\" represented by the number 0.")
                rules.append("6. \"Peng\" takes precedence over \"Chi.\"")
                rules.append(f"7. The card that is removed does not affect the result determination.")
            else:
                rules.append("4. If the newly added card does not meet the condition in rule 3, the result is determined as \"Pass,\" represented by the number 0.")
                rules.append("5. The removed card does not affect the result determination.")
            
            rounds_text = []
            for i, round_data in enumerate(puzzle.rounds, 1):
                round_text = f"Round {i}: Add a {round_data['add']} card"
                if i < len(puzzle.rounds):  # Don't show removal for last round
                    round_text += f" and remove a {round_data['remove']} card."
                rounds_text.append(round_text)
            
            question = (
                f"There are several letter cards, and the game rules are as follows:"
                f"{chr(10).join(rules)}\n"
                f"Given the initial cards:{puzzle.initial_cards}\n\n"
                f"{chr(10).join(rounds_text)}\n\n"
                f"Based on the above information, determine the number judged in the final round.\n"
                f"Return your answer with the format <<<x>>>, where x is 0 for Pass, "
                f"1 for Peng{', or 2 for Chi' if complexity > 2 else ''}."
            )
            
            # Save files
            with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                f.write(question)
                
            with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                json.dump({
                    'initial_cards': puzzle.initial_cards,
                    'rounds': puzzle.rounds,
                    'solution': puzzle.solution,
                    'complexity': puzzle.complexity
                }, f, indent=2)
                
            sample_id += 1

def check_llm_answer(llm_response: str, solution: int) -> bool:
    """Check if LLM's answer matches the solution"""
    try:
        # Find answer between <<< and >>>
        pattern = r'<<<\s*(\d)\s*>>>'
        match = re.search(pattern, llm_response)
        if not match:
            return False
        
        llm_answer = int(match.group(1))
        return llm_answer == solution
        
    except Exception:
        return False

if __name__ == "__main__":
    output_dir = '../dataset_gather/mahjong_pattern_dataset'
    generate_dataset(output_dir)