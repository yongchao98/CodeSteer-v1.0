import random
import os
import json
from typing import List, Dict, Tuple
import itertools
from dataclasses import dataclass
import math

@dataclass
class ArithmeticPuzzle:
    numbers: List[int]
    target: int
    operations: List[str]
    solution: str
    complexity: int  # 1-5 scale based on number of operations and parentheses

class ArithmeticPuzzleGenerator:
    def __init__(self):
        self.operations = ['+', '-', '*', '/']
        self.max_number = 9  # Maximum number allowed in puzzles
        
    def evaluate_expression(self, nums: List[int], ops: List[str], parens: List[Tuple[int, int]] = None) -> float:
        """Evaluate an arithmetic expression with given numbers, operations, and optional parentheses"""
        expression = ""
        if parens is None:
            parens = []
            
        # Convert to string expression
        curr_pos = 0
        open_parens = sorted([p[0] for p in parens])
        close_parens = sorted([p[1] for p in parens])
        
        for i, num in enumerate(nums):
            # Add opening parentheses
            while open_parens and curr_pos == open_parens[0]:
                expression += "("
                open_parens.pop(0)
                curr_pos += 1
                
            expression += str(num)
            curr_pos += 1
            
            # Add closing parentheses
            while close_parens and curr_pos == close_parens[0]:
                expression += ")"
                close_parens.pop(0)
                curr_pos += 1
                
            if i < len(ops):
                expression += ops[i]
                curr_pos += 1
                
        try:
            result = eval(expression)
            # Check if result is an integer
            if isinstance(result, (int, float)) and math.isclose(result, round(result), rel_tol=1e-9):
                return round(result)
            return result
        except:
            return None

    def generate_puzzle(self, num_numbers: int, target: int, min_complexity: int = 1, max_complexity: int = 5) -> ArithmeticPuzzle:
        """Generate a puzzle with specified number of numbers and target value"""
        while True:
            # Generate random numbers
            numbers = [random.randint(1, self.max_number) for _ in range(num_numbers)]
            
            # Try different operation combinations
            for ops in itertools.product(self.operations, repeat=num_numbers-1):
                # Try different parentheses combinations based on complexity
                max_parens = min(2, (num_numbers - 1) // 2)
                for num_parens in range(max_parens + 1):
                    for parens_positions in itertools.combinations(range(num_numbers * 2 - 1), num_parens * 2):
                        if num_parens == 0:
                            parens = []
                        else:
                            parens = [(parens_positions[i], parens_positions[i+1]) 
                                    for i in range(0, len(parens_positions), 2)]
                            
                        result = self.evaluate_expression(numbers, ops, parens)
                        if result == target:
                            # Calculate complexity
                            complexity = self._calculate_complexity(ops, num_parens)
                            if min_complexity <= complexity <= max_complexity:
                                solution = self._format_solution(numbers, ops, parens)
                                return ArithmeticPuzzle(
                                    numbers=numbers,
                                    target=target,
                                    operations=list(ops),
                                    solution=solution,
                                    complexity=complexity
                                )

    def _calculate_complexity(self, ops: Tuple[str, ...], num_parens: int) -> int:
        """Calculate puzzle complexity on a scale of 1-5"""
        # Base complexity from number of operations
        complexity = len(ops) // 2
        
        # Additional complexity from operation types
        if '*' in ops or '/' in ops:
            complexity += 1
            
        # Additional complexity from parentheses
        complexity += num_parens
        
        return min(max(complexity, 1), 5)

    def _format_solution(self, numbers: List[int], ops: List[str], parens: List[Tuple[int, int]]) -> str:
        """Format the solution as a list of numbers and symbols"""
        solution = []
        curr_pos = 0
        open_parens = sorted([p[0] for p in parens])
        close_parens = sorted([p[1] for p in parens])
        
        for i, num in enumerate(numbers):
            # Add opening parentheses
            while open_parens and curr_pos == open_parens[0]:
                solution.append('(')
                open_parens.pop(0)
                curr_pos += 1
                
            solution.append(str(num))
            curr_pos += 1
            
            # Add closing parentheses
            while close_parens and curr_pos == close_parens[0]:
                solution.append(')')
                close_parens.pop(0)
                curr_pos += 1
                
            if i < len(ops):
                solution.append(ops[i])
                curr_pos += 1
                
        return solution

def generate_dataset(output_dir: str):
    """Generate a dataset of arithmetic puzzles with varying complexity"""
    os.makedirs(output_dir, exist_ok=True)
    generator = ArithmeticPuzzleGenerator()
    
    # Define different configurations for variety
    configurations = [
        # (num_numbers, target, min_complexity, max_complexity, num_samples)
        (3, 10, 1, 2, 20),
        (4, 24, 2, 3, 20),
        (5, 78, 3, 4, 20),
        (5, 69, 1, 4, 20),
        (6, 100, 4, 5, 20)
    ]
    
    sample_id = 0
    for num_numbers, target, min_complexity, max_complexity, num_samples in configurations:
        for _ in range(num_samples):
            puzzle = generator.generate_puzzle(num_numbers, target, min_complexity, max_complexity)
            
            # Create sample directory
            sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Format question
            question = (
                f"Given {num_numbers} integers between 1 and 9, perform arithmetic operations "
                f"(addition, subtraction, multiplication, division, and parentheses) between these integers. "
                f"Each number must be used exactly once and only once, and the final result should be {target}. "
                f"Note: The order of the numbers is determined by the input, and you cannot change the order "
                f"of the numbers. You can only insert arithmetic operations or parentheses between the numbers.\n"
                f"Return the answer with a list of values and calculation symbols. Return your answer with the format <<<list of values and symbols>>>, like <<<[1, +, 3]>>> means 1+3, <<<[4, -, (, 6, , 9, )]>>> means 4-69*."
                f"Now the input numbers are: {', '.join(map(str, puzzle.numbers))}, the target values is: {target}\n Your answer:"
            )
            
            # Save question and solution
            with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                f.write(question)
                
            with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                json.dump({
                    'numbers': puzzle.numbers,
                    'target': target,
                    'solution': puzzle.solution,
                    'complexity': puzzle.complexity
                }, f, indent=2)
                
            sample_id += 1
            print(f"Generated sample {sample_id}: {puzzle.numbers} -> {puzzle.solution}")

if __name__ == "__main__":
    output_dir = 'dataset_gather/combinatorial_calculation'
    generate_dataset(output_dir)
