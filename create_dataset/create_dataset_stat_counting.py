import random
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any
import re

@dataclass
class CountingPuzzle:
    sequence: str
    rules: List[Dict[str, Any]]
    solution: int
    complexity: int  # 1-5 scale based on number of rules and sequence length

class CountingPuzzleGenerator:
    def __init__(self):
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        
    def calculate_score(self, sequence: str, rules: List[Dict[str, Any]]) -> int:
        """Calculate the total score for a sequence based on given rules"""
        total_score = 0
        
        for rule in rules:
            letter = rule['letter']
            score = rule['score']
            min_consecutive = rule.get('min_consecutive', 2)
            
            # Look for consecutive occurrences
            current_count = 0
            for char in sequence:
                if char == letter:
                    current_count += 1
                else:
                    if current_count >= min_consecutive:
                        total_score += score
                    current_count = 0
                    
            # Check last sequence
            if current_count >= min_consecutive:
                total_score += score
                
        return total_score

    def generate_sequence(self, length: int, num_letters: int) -> str:
        """Generate a random sequence of specified length using given number of letters"""
        available_letters = self.letters[:num_letters]
        return ''.join(random.choice(available_letters) for _ in range(length))

    def generate_rules(self, num_rules: int) -> List[Dict[str, Any]]:
        """Generate scoring rules"""
        rules = []
        for i in range(num_rules):
            rules.append({
                'letter': self.letters[i],
                'score': i + 1,
                'min_consecutive': 2
            })
        return rules

    def generate_puzzle(self, sequence_length: int, num_letters: int, 
                       min_complexity: int = 1, max_complexity: int = 5) -> CountingPuzzle:
        """Generate a counting puzzle with specified parameters"""
        rules = self.generate_rules(num_letters)
        sequence = self.generate_sequence(sequence_length, num_letters)
        solution = self.calculate_score(sequence, rules)
        
        # Calculate complexity based on sequence length and number of rules
        complexity = min(max(
            (sequence_length // 5) + (num_letters // 2),
            min_complexity
        ), max_complexity)
        
        return CountingPuzzle(
            sequence=sequence,
            rules=rules,
            solution=solution,
            complexity=complexity
        )

def format_question(puzzle: CountingPuzzle) -> str:
    """Format the puzzle as a question"""
    question = "The initial total score is 0. Given a string consisting of "
    if len(puzzle.rules) == 1:
        question += f"the letter {puzzle.rules[0]['letter']}, "
    else:
        letters = [rule['letter'] for rule in puzzle.rules]
        question += f"letters {', '.join(letters[:-1])} and {letters[-1]}, "
    
    question += "scan from front to back:\n\n"
    
    for i, rule in enumerate(puzzle.rules, 1):
        question += f"{i}. If there are two or more consecutive {rule['letter']}'s, "
        question += f"add {rule['score']} to the total score.\n"
    
    question += f"\nWhat is the total score of the string {puzzle.sequence}?\n"
    question += "Return your answer with the format <<<number>>>, like <<<5>>> means the total score is 5."
    
    return question

def verify_answer(answer: str, expected: int) -> bool:
    """Verify if the given answer matches the expected solution"""
    # Try to extract number from various formats
    patterns = [
        r'<<<(\d+)>>>',  # Correct format
        r'(\d+)',        # Just the number
        r'total score is (\d+)',  # Natural language
        r'score: (\d+)'  # Alternative format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match:
            try:
                return int(match.group(1)) == expected
            except ValueError:
                continue
    return False

def generate_dataset(output_dir: str):
    """Generate a dataset of counting puzzles with varying complexity"""
    os.makedirs(output_dir, exist_ok=True)
    generator = CountingPuzzleGenerator()
    
    # Define different configurations for variety
    configurations = [
        # (sequence_length, num_letters, min_complexity, max_complexity, num_samples)
        (5, 2, 1, 2, 2),    # Simple cases
        (10, 3, 2, 3, 4),   # Medium complexity
        (15, 4, 3, 4, 4),   # More complex
        (20, 5, 4, 4, 10),   # Advanced
        (25, 6, 4, 5, 20),   # Very complex
        (35, 3, 4, 5, 20),
        (45, 3, 4, 5, 20),
        (55, 3, 4, 5, 20),
        (65, 3, 4, 5, 20),
        (65, 4, 4, 5, 20),
        (65, 5, 4, 5, 20),
        (65, 6, 4, 5, 20),
        (65, 7, 4, 5, 20),
    ]
    
    sample_id = 0
    for seq_len, num_letters, min_complex, max_complex, num_samples in configurations:
        for _ in range(num_samples):
            puzzle = generator.generate_puzzle(seq_len, num_letters, min_complex, max_complex)
            
            # Create sample directory
            sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Format and save question
            question = format_question(puzzle)
            with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                f.write(question)
                
            # Save solution and metadata
            with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                json.dump({
                    'sequence': puzzle.sequence,
                    'rules': puzzle.rules,
                    'solution': puzzle.solution,
                    'complexity': puzzle.complexity
                }, f, indent=2)
                
            sample_id += 1
            print(f"Generated sample {sample_id}: {puzzle.sequence} -> {puzzle.solution}")

if __name__ == "__main__":
    output_dir = '../dataset_gather/statistical_counting_dataset'
    generate_dataset(output_dir)