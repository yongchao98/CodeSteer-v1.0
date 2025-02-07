import random
import os
import json
from typing import List, Dict, Tuple
import itertools


class LetterNumberPuzzle:
    def __init__(self, num_letters: int = 5, num_constraints: int = 3, values: List[int] = None):
        self.num_letters = num_letters
        self.num_constraints = num_constraints
        self.custom_values = values
        self.letters = [chr(65 + i) for i in range(num_letters)]  # A, B, C, ...

    def generate_puzzle(self) -> Tuple[str, List[int]]:
        # Generate a valid solution first
        numbers = self._generate_numbers()
        solution = self._generate_solution(numbers)
        constraints = self._generate_constraints(solution)
        question = self._format_question(constraints, numbers)
        return question, solution

    def _generate_numbers(self) -> List[int]:
        """Generate a list of numbers to be used in the puzzle"""
        if self.custom_values:
            if len(self.custom_values) != self.num_letters:
                raise ValueError("Number of values must match number of letters")
            return sorted(self.custom_values)
        return sorted([1, 1, 3, 4, 5])  # Default values if none provided

    def _generate_solution(self, numbers: List[int]) -> List[int]:
        """Generate a valid solution mapping letters to numbers"""
        solution = numbers.copy()
        random.shuffle(solution)
        return solution

    def _generate_constraints(self, solution: List[int]) -> List[str]:
        """Generate constraints based on the solution with priority on equations"""
        constraints = []
        solution_dict = {self.letters[i]: solution[i] for i in range(self.num_letters)}

        # Prioritize equation-based constraints
        equation_constraints = [
            self._generate_equality_constraint,
            self._generate_multiple_constraint,
            self._generate_difference_constraint,
            self._generate_sum_constraint,
        ]

        inequality_constraints = [
            self._generate_greater_than_constraint,
        ]

        max_attempts = 20000
        attempt = 0

        while len(constraints) < self.num_constraints and attempt < max_attempts:
            # Try equation-based constraints first with higher probability
            if random.random() < 0.95 and equation_constraints:  # 80% chance for equation constraints
                constraint_func = random.choice(equation_constraints)
            elif inequality_constraints:
                constraint_func = random.choice(inequality_constraints)
            else:
                break

            constraint = constraint_func(solution_dict)

            if constraint and constraint not in constraints:
                constraints.append(constraint)
                # Remove used constraint type if we want variety
                if random.random() < 0.03:  # 30% chance to remove used constraint type
                    if constraint_func in equation_constraints:
                        equation_constraints.remove(constraint_func)
                    elif constraint_func in inequality_constraints:
                        inequality_constraints.remove(constraint_func)

            attempt += 1

        if len(constraints) < self.num_constraints:
            raise ValueError(
                f"Could only generate {len(constraints)} unique constraints after {max_attempts} attempts. Need {self.num_constraints}. Try different numbers or fewer constraints.")

        return constraints

    def _generate_equality_constraint(self, solution_dict: Dict[str, int]) -> str:
        """Generate an equality constraint (e.g., A = C)"""
        letters = list(solution_dict.keys())
        equal_pairs = [(l1, l2) for l1, l2 in itertools.combinations(letters, 2)
                       if solution_dict[l1] == solution_dict[l2]]
        if equal_pairs:
            l1, l2 = random.choice(equal_pairs)
            return f"{l1} = {l2}"
        return None

    def _generate_multiple_constraint(self, solution_dict: Dict[str, int]) -> str:
        """Generate a multiple constraint (e.g., E = 2.5C or E = 3.2C)"""
        letters = list(solution_dict.keys())
        multiple_pairs = []
        for l1, l2 in itertools.permutations(letters, 2):
            # Check multipliers from 1.1 to 4.9 with one decimal place
            for mult in [round(x * 0.1, 1) for x in range(11, 50)]:
                if abs(solution_dict[l1] - mult * solution_dict[
                    l2]) < 0.0000001:  # Using small epsilon for float comparison
                    multiple_pairs.append((l1, l2, mult))
        if multiple_pairs:
            l1, l2, mult = random.choice(multiple_pairs)
            return f"{l1} = {mult}{l2}"
        return None

    def _generate_difference_constraint(self, solution_dict: Dict[str, int]) -> str:
        """Generate a difference constraint (e.g., B - D = 1 or A - C = 2)"""
        letters = list(solution_dict.keys())
        diff_pairs = []
        for l1, l2 in itertools.permutations(letters, 2):
            diff = solution_dict[l1] - solution_dict[l2]
            #if 0 < abs(diff) <= 3:  # Limit difference to reasonable values
            diff_pairs.append((l1, l2, diff))
        if diff_pairs:
            l1, l2, diff = random.choice(diff_pairs)
            return f"{l1} - {l2} = {diff}"
        return None

    def _generate_sum_constraint(self, solution_dict: Dict[str, int]) -> str:
        """Generate a sum constraint (e.g., A + B = 5)"""
        letters = list(solution_dict.keys())
        sum_pairs = []
        for l1, l2 in itertools.combinations(letters, 2):
            sum_value = solution_dict[l1] + solution_dict[l2]
            #if sum_value <= max(solution_dict.values()) * 2:
            sum_pairs.append((l1, l2, sum_value))
        if sum_pairs:
            l1, l2, sum_val = random.choice(sum_pairs)
            return f"{l1} + {l2} = {sum_val}"
        return None

    def _generate_greater_than_constraint(self, solution_dict: Dict[str, int]) -> str:
        """Generate a greater than constraint (e.g., A > B)"""
        letters = list(solution_dict.keys())
        greater_pairs = [(l1, l2) for l1, l2 in itertools.permutations(letters, 2)
                         if solution_dict[l1] > solution_dict[l2]]
        if greater_pairs:
            l1, l2 = random.choice(greater_pairs)
            return f"{l1} > {l2}"
        return None

    def _format_question(self, constraints: List[str], numbers: List[int]) -> str:
        """Format the puzzle question"""
        question = (
            f"Calculate the number for each letter based on the range and set of inequalities "
            f"that the given letter represents. There are {self.num_letters} letters "
            f"{', '.join(self.letters)}, each representing one of the numbers {', '.join(map(str, numbers))}. "
            f"The system of inequalities is satisfied as follows:\n"
        )

        for i, constraint in enumerate(constraints, 1):
            question += f"{i}. {constraint}\n"

        question += ("\nReturn the answers in alphabetical order starting with A, and each element "
                     "in the list represents the value of the corresponding letter. Return your answer "
                     "with the format <<<list of values>>>, like <<<[1, 2, 3]>>> means that A represents 1, "
                     "B represents 2, and C represents 3.")

        return question

    def verify_solution(self, proposed_solution: List[int], constraints: List[str]) -> bool:
        """Verify if the proposed solution satisfies all constraints"""
        if len(proposed_solution) != self.num_letters:
            return False
            
        solution_dict = {self.letters[i]: proposed_solution[i] for i in range(self.num_letters)}
        
        def parse_term(term: str) -> int:
            if term[0].isalpha():  # Single letter
                return solution_dict[term]
            else:  # Term like '3B'
                multiplier = int(term[:-1])
                letter = term[-1]
                return multiplier * solution_dict[letter]
        
        for constraint in constraints:
            parts = constraint.split()
            
            if len(parts) == 3:  # A = B, A > B, A < B
                left_val = parse_term(parts[0])
                right_val = parse_term(parts[2])
                
                if parts[1] == "=":
                    if left_val != right_val:
                        return False
                elif parts[1] == ">":
                    if left_val <= right_val:
                        return False
                elif parts[1] == "<":
                    if left_val >= right_val:
                        return False
                        
            elif len(parts) == 5:  # A + B = 5, B - D = 1
                left_val = parse_term(parts[0])
                right_val = parse_term(parts[2])
                result = int(parts[4])
                
                if parts[1] == "+":
                    if left_val + right_val != result:
                        return False
                elif parts[1] == "-":
                    if left_val - right_val != result:
                        return False
        
        # Check if all numbers in proposed solution are valid
        required_numbers = self._generate_numbers()
        return sorted(proposed_solution) == sorted(required_numbers)

def read_puzzle(puzzle_dir: str, sample_id: int) -> Tuple[str, List[int]]:
    """Read a puzzle and its solution from files"""
    with open(os.path.join(puzzle_dir, f'sample_{sample_id}_question.txt'), 'r') as f:
        question = f.read()
        
    with open(os.path.join(puzzle_dir, f'sample_{sample_id}_solution.json'), 'r') as f:
        solution = json.load(f)
        
    return question, solution

def parse_llm_answer(answer: str) -> List[int]:
    """Parse the answer from an LLM in the format <<<[1, 2, 3]>>>"""
    try:
        # Extract content between <<< and >>>
        start_idx = answer.find('<<<') + 3
        end_idx = answer.find('>>>')
        if start_idx == -1 or end_idx == -1:
            raise ValueError("Answer not in correct format")
            
        content = answer[start_idx:end_idx].strip()
        # Convert string representation of list to actual list
        solution = eval(content)
        if not isinstance(solution, list):
            raise ValueError("Answer not a list")
        return solution
    except:
        raise ValueError("Could not parse answer")

# Example usage:
if __name__ == "__main__":

    # num_samples: int, output_dir: str, num_letters: int = 5, num_constraints: int = 3, values: List[int] = None):
    """Generate multiple puzzles and save them to files"""
    output_dir = '../dataset_gather/logical_equation'
    os.makedirs(output_dir, exist_ok=True)

    for num_letters, num_constraints, values in [
            (9, 7, [1, 3, 4, 9, 16, 27, 36, 80, 121]),
            (9, 8, [3, 6, 9, 20, 32, 36, 80, 121, 120]),
            (11, 10, [3, 9, 16, 27, 36, 48, 75, 80, 121, 150, 225]),
            (11, 11, [3, 9, 16, 20, 39, 48, 75, 80, 121, 150, 225]),
            (13, 10, [1, 2, 3, 5, 7, 16, 15, 24, 10, 45, 28, 36, 50]),
            (13, 11, [2, 3, 5, 7, 16, 15, 24, 10, 45, 28, 36, 50, 96]),
            (13, 12, [2, 3, 5, 7, 16, 15, 24, 10, 45, 28, 36, 50, 96]),
            (15, 12, [2, 3, 5, 7, 16, 15, 24, 10, 45, 28, 36, 50, 78, 90, 100]),
            (15, 13, [1, 3, 5, 7, 16, 15, 24, 12, 45, 34, 36, 56, 78, 95, 100]),
            (15, 14, [1, 3, 5, 7, 16, 15, 24, 12, 45, 34, 36, 56, 78, 95, 100]),
        ]:
        puzzle_generator = LetterNumberPuzzle(num_letters=num_letters, num_constraints=num_constraints, values=values)

        for i in range(15):
            for iteration in range(20):
                try:
                    question, solution = puzzle_generator.generate_puzzle()
                    break
                except:
                    pass
            print(f"\n\nSample {num_letters}_{num_constraints}_{i} question:\n{question}")
            print(f"Sample {num_letters}_{num_constraints}_{i} solution:\n{solution}")

            sample_dir = os.path.join(output_dir, f'sample_{num_letters}_{num_constraints}_{i}')
            os.makedirs(sample_dir, exist_ok=True)
            # Save question
            with open(os.path.join(sample_dir, f'question.txt'), 'w') as f:
                f.write(question)

            # Save solution
            with open(os.path.join(sample_dir, f'solution.json'), 'w') as f:
                json.dump(solution, f)