import random
import os
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ArrangementPuzzle:
    items: List[str]
    constraints_text: str
    solution: List[str]
    complexity: int  # 1-5 scale for puzzle difficulty


class ArrangementPuzzleGenerator:
    """
    Generates linear arrangement puzzles of varying complexity.
    Supported constraint types include:
      - ("fixed_position", item, pos)
      - ("left_of", item1, item2)
      - ("not_in_position", item, pos)
      - ("right_of", item1, item2)
      - ("adjacent_to", item1, item2)
    Then finds a valid arrangement that satisfies all constraints.
    """

    def __init__(self, max_items=6):
        # For example, generate puzzles with up to 6 items labeled A, B, C, ...
        self.max_items = max_items

    def generate_puzzle(self, num_items: int, complexity: int) -> ArrangementPuzzle:
        items = [chr(ord('A') + i) for i in range(num_items)]

        # Generate constraints (with a variety of possible types)
        constraints = self._generate_constraints(items, complexity)

        # Attempt to solve the puzzle
        solution = self._find_solution(items, constraints)
        if not solution:
            # If no solution found, try again (for demonstration, we loop until we find one).
            return self.generate_puzzle(num_items, complexity)

        # Build the textual constraints for the puzzle prompt
        constraints_text = self._build_constraint_text(constraints)

        puzzle = ArrangementPuzzle(
            items=items,
            constraints_text=constraints_text,
            solution=solution,
            complexity=complexity
        )
        return puzzle

    def _generate_constraints(self, items: List[str], complexity: int) -> List[Tuple]:
        """
        Randomly generate constraints. The 'complexity' can control how many constraints to generate.
        Possible constraints:
          - ("fixed_position", item, position)
          - ("left_of", item1, item2)
          - ("not_in_position", item, position)
          - ("right_of", item1, item2)
          - ("adjacent_to", item1, item2)
        """
        constraints = []
        num_items = len(items)

        constraint_types = [
            "fixed_position",
            "left_of",
            "not_in_position",
            "right_of",
            "adjacent_to"
        ]

        # e.g. complexity=1 => ~1-2 constraints, complexity=5 => ~5-6 constraints
        num_constraints = random.randint(complexity, complexity + 1)

        for _ in range(num_constraints):
            ctype = random.choice(constraint_types)
            if ctype == "fixed_position":
                item = random.choice(items)
                position = random.randint(1, num_items)
                constraints.append((ctype, item, position))
            elif ctype == "left_of":
                item1, item2 = random.sample(items, 2)
                constraints.append((ctype, item1, item2))
            elif ctype == "not_in_position":
                item = random.choice(items)
                position = random.randint(1, num_items)
                constraints.append((ctype, item, position))
            elif ctype == "right_of":
                item1, item2 = random.sample(items, 2)
                constraints.append((ctype, item1, item2))
            elif ctype == "adjacent_to":
                item1, item2 = random.sample(items, 2)
                constraints.append((ctype, item1, item2))

        return constraints

    def _find_solution(self, items: List[str], constraints: List[Tuple]) -> List[str]:
        from itertools import permutations

        for perm in permutations(items):
            if self._check_constraints(perm, constraints):
                return list(perm)
        return None

    def _check_constraints(self, arrangement: Tuple[str, ...], constraints: List[Tuple]) -> bool:
        position_map = {item: i + 1 for i, item in enumerate(arrangement)}

        for ctype, *rest in constraints:
            if ctype == "fixed_position":
                item, pos = rest
                if position_map[item] != pos:
                    return False
            elif ctype == "left_of":
                item1, item2 = rest
                if position_map[item1] >= position_map[item2]:
                    return False
            elif ctype == "not_in_position":
                item, pos = rest
                if position_map[item] == pos:
                    return False
            elif ctype == "right_of":
                item1, item2 = rest
                if position_map[item1] <= position_map[item2]:
                    return False
            elif ctype == "adjacent_to":
                item1, item2 = rest
                if abs(position_map[item1] - position_map[item2]) != 1:
                    return False

        return True

    def _build_constraint_text(self, constraints: List[Tuple]) -> str:
        lines = []
        for idx, (ctype, *rest) in enumerate(constraints, start=1):
            if ctype == "fixed_position":
                item, pos = rest
                lines.append(f"{idx}. Book {item} must be placed in position {pos}.")
            elif ctype == "left_of":
                item1, item2 = rest
                lines.append(f"{idx}. Book {item1} must be to the left of book {item2}.")
            elif ctype == "not_in_position":
                item, pos = rest
                lines.append(f"{idx}. Book {item} cannot be placed in position {pos}.")
            elif ctype == "right_of":
                item1, item2 = rest
                lines.append(f"{idx}. Book {item1} must be to the right of book {item2}.")
            elif ctype == "adjacent_to":
                item1, item2 = rest
                lines.append(f"{idx}. Book {item1} must be adjacent to book {item2}.")
        return "\n".join(lines)


def check_llm_answer(llm_answer: str, puzzle: ArrangementPuzzle) -> bool:
    """
    Attempt to parse the LLM's answer into a list of items in order.
    Then check if it satisfies the puzzle constraints. If it does, return True,
    otherwise False.

    We consider that the LLM might not follow the '<<< >>>' format strictly.
    We'll try to parse anything that looks like a list of strings.
    """
    import re

    pattern = r'<<<(.*?)>>>'
    match = re.search(pattern, llm_answer, flags=re.DOTALL)
    if match:
        raw_list_str = match.group(1).strip()
    else:
        raw_list_str = llm_answer.strip()

    item_pattern = r'"([^"]+)"|\b([A-Za-z])\b'
    found = re.findall(item_pattern, raw_list_str)
    arrangement = [g1 if g1 else g2 for (g1, g2) in found]

    if not arrangement:
        return False

    if len(arrangement) != len(puzzle.items):
        return False

    if sorted(arrangement) != sorted(puzzle.items):
        return False

    # For demonstration, just check if arrangement == puzzle.solution
    return list(arrangement) == puzzle.solution


def generate_dataset(output_dir: str):
    """
    Generate a dataset of arrangement puzzles with varying complexity.
    Each puzzle is saved to a sample directory containing question and solution.
    We'll produce a total of 100 samples by adjusting the num_samples in configurations.
    """
    os.makedirs(output_dir, exist_ok=True)
    generator = ArrangementPuzzleGenerator()

    # We have 4 sets of config, each generating 25 samples => total 100
    configurations = [
        (6, 7, 50),  # 25 samples
        (7, 8, 50),  # 25 samples
        (8, 9, 50),  # 25 samples
        (9, 10, 50),  # 25 samples
    ]

    sample_id = 0
    for (num_items, complexity, num_samples) in configurations:
        for _ in range(num_samples):
            puzzle = generator.generate_puzzle(num_items, complexity)
            sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
            os.makedirs(sample_dir, exist_ok=True)

            # --- Build the question prompt ---
            question_prompt = (
                f"Xiao Wang is a librarian responsible for shelving {num_items} books labeled "
                f"{', '.join(puzzle.items)} on a shelf with positions 1 to {num_items}.\n\n"
                f"The constraints are:\n"
                f"{puzzle.constraints_text}\n\n"
                f"Question:\n"
                f"Based on the above conditions, please arrange the {num_items} books "
                f"from left to right on the bookshelf, i.e. positions 1 through {num_items}. "
                f"Return your answer with the format <<< a list of strings >>>, one solution is enough.for example:\n"
                f"<<<[\"A\", \"B\", \"C\"]>>>\n"
                f"Your answer:\n"
            )

            # --- Save question and ground-truth solution ---
            with open(os.path.join(sample_dir, 'question.txt'), 'w') as f_q:
                f_q.write(question_prompt)

            solution_data = {
                'items': puzzle.items,
                'constraints_text': puzzle.constraints_text,
                'correct_solution': puzzle.solution,
                'complexity': puzzle.complexity
            }
            with open(os.path.join(sample_dir, 'solution.json'), 'w') as f_s:
                json.dump(solution_data, f_s, indent=2)

            sample_id += 1
            print(f"[INFO] Generated sample {sample_id}: {puzzle.items}, complexity={puzzle.complexity}")


if __name__ == "__main__":
    output_dir = "../dataset_gather/permutations_and_combinations"
    generate_dataset(output_dir)
