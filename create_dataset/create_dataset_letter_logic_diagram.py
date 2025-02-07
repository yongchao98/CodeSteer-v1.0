import os
import json
import random
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class LetterLogicPuzzle:
    """
    A 7x7 letter-logic puzzle.

    Attributes:
      grid: 7x7 partially filled grid (None represents an empty cell).
      solution: 7x7 fully filled solution grid.
      complexity: Integer in [1..5], controls how many cells get removed.
    """
    grid: List[List[Optional[str]]]
    solution: List[List[str]]
    complexity: int


class LetterLogicPuzzleGenerator:
    def __init__(self):
        """
        Generator for a 7x7 puzzle that uses letters {a..g} such that:
         1) Each row has exactly one of each letter {a..g}.
         2) Each column has exactly one of each letter {a..g}.
         3) The minor diagonal (col = 6 - row) is uniform (all 'g' if unshuffled).
        """
        self.letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        self.n_rows = 7
        self.n_cols = 7

    def generate_dataset(
            self,
            output_dir: str,
            num_samples: int = 5,
            complexities: Optional[List[int]] = None
    ):
        """
        Generate a dataset of puzzles, each with:
         - A partially filled grid.
         - A valid solution grid.
         - A 'complexity' level controlling how many cells are removed.

        :param output_dir: Directory where puzzle samples are saved
        :param num_samples: Number of puzzles to generate
        :param complexities: Optional list of complexity values (1..5).
                            If not given, random complexities are used.
        """
        os.makedirs(output_dir, exist_ok=True)

        if complexities is None:
            complexities = [random.randint(1, 5) for _ in range(num_samples)]

        for i in range(num_samples):
            complexity = complexities[i % len(complexities)]

            # 1) Build a valid 7x7 solution that has a uniform minor diagonal
            full_solution = self._generate_full_solution()

            # 2) Remove some cells (up to max_removals) to create a puzzle
            puzzle_grid = self._remove_cells(full_solution, complexity)

            puzzle = LetterLogicPuzzle(
                grid=puzzle_grid,
                solution=full_solution,
                complexity=complexity
            )

            # 3) Save puzzle question and solution
            sample_dir = os.path.join(output_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)

            question_text = self._format_question_as_text(puzzle.grid)
            question_prompt = (
                    "Given a 7x7 grid of letters {a..g}, some cells are pre-filled. "
                    "Fill the rest so that:\n"
                    "1) Each row has exactly one of each letter {a..g}.\n"
                    "2) Each column has exactly one of each letter {a..g}.\n"
                    "3) All cells on the minor diagonal (top-right to bottom-left) contain the same letter.\n\n"
                    "Here is the puzzle (each row on its own line, cells separated by commas, empty cells blank):\n\n"
                    + question_text
                    + "\n\nReturn the answer with the format:\n"
                      "<<<\n"
                      "row1\n"
                      "row2\n"
                      "...  (7 rows total)\n"
                      ">>>\n\n"
                      "where each row has 7 letters separated by commas.\n"
            )

            # Write puzzle question
            with open(os.path.join(sample_dir, "question.txt"), "w") as f:
                f.write(question_prompt)

            # Write puzzle data in JSON
            puzzle_data = {
                "grid": puzzle.grid,
                "solution": puzzle.solution,
                "complexity": puzzle.complexity
            }
            with open(os.path.join(sample_dir, "solution.json"), "w") as f:
                json.dump(puzzle_data, f, indent=2)

            print(f"Generated sample_{i} with complexity={complexity}.")

    def _generate_full_solution(self) -> List[List[str]]:
        """
        Construct a 7x7 Latin square with a uniform minor diagonal.
        We exploit the group approach: row r, col c -> (r + c) mod 7
        Then rename 0..6 to letters {a..g} in some random order if desired.

        Because (r + (6-r)) mod 7 = 6 for all r, that means
        the minor diagonal is always mapped to 'letter_map[6]', ensuring uniform diagonal.
        """
        # Optionally shuffle letters to get variety
        # e.g. letter_perm = random.sample(self.letters, 7)
        # but let's do it for demonstration
        letter_perm = random.sample(self.letters, 7)

        # letter_map[i] = letter_perm[i], i in 0..6
        letter_map = {}
        for i, letter in enumerate(letter_perm):
            letter_map[i] = letter

        # Build the 7x7 grid: cell (r,c) = (r+c) mod 7
        solution = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                val = (r + c) % 7
                row.append(letter_map[val])
            solution.append(row)

        # Now (r, 6-r) always becomes letter_map[6], i.e. the same letter in each row.
        return solution

    def _remove_cells(self, full_solution: List[List[str]], complexity: int) -> List[List[Optional[str]]]:
        """
        Remove up to 'max_removals' cells from the solution to form a puzzle.
        The complexity determines how many cells are removed.
        """
        max_removals = min(10 + 5 * complexity, 30)

        puzzle_grid = [row[:] for row in full_solution]  # copy each row
        coords = [(r, c) for r in range(self.n_rows) for c in range(self.n_cols)]
        random.shuffle(coords)

        removed_count = 0
        for (r, c) in coords:
            if removed_count >= max_removals:
                break
            puzzle_grid[r][c] = None
            removed_count += 1

        return puzzle_grid

    def _format_question_as_text(self, grid: List[List[Optional[str]]]) -> str:
        """
        Format a partially filled 7x7 grid as lines of comma-separated values,
        with empty cells as blank.
        """
        lines = []
        for row in grid:
            row_str = ",".join(letter if letter is not None else "" for letter in row)
            lines.append(row_str)
        return "\n".join(lines)


###################################
# Checking an LLM's answer
###################################
def check_letter_logic_answer(
        llm_answer: str,
        puzzle: LetterLogicPuzzle
) -> bool:
    """
    Check if an LLM's answer is correct:

    1) Must have 7 lines, each with 7 letters {a..g} separated by commas.
    2) Must match puzzle.grid where puzzle.grid[r][c] is not None.
    3) Each row & column has unique letters a..g.
    4) Minor diagonal (col=6-r) is uniform.

    We try to parse from within <<< >>> or the entire text if not found.
    """
    import re

    # Extract lines within <<< >>>
    pattern = r'<<<([\s\S]*?)>>>'
    match = re.search(pattern, llm_answer)
    if match:
        extracted = match.group(1).strip()
    else:
        extracted = llm_answer.strip()

    lines = [line.strip() for line in extracted.split('\n') if line.strip()]
    if len(lines) != 7:
        print("Error: The answer does not have exactly 7 lines.")
        return False

    # Split each row by commas
    filled_grid = []
    for row_idx, line in enumerate(lines):
        cells = [cell.strip() for cell in line.split(',')]
        if len(cells) != 7:
            print(f"Error: Row {row_idx} does not have 7 comma-separated entries.")
            return False
        filled_grid.append(cells)

    allowed_letters = {'a', 'b', 'c', 'd', 'e', 'f', 'g'}

    # Check letters + pre-filled
    for r in range(7):
        for c in range(7):
            letter = filled_grid[r][c]
            if letter not in allowed_letters:
                print(f"Error: Invalid letter '{letter}' at ({r},{c}).")
                return False
            # If puzzle.grid[r][c] was given (not None), must match
            if puzzle.grid[r][c] is not None:
                if puzzle.grid[r][c] != letter:
                    print(
                        f"Error: Pre-filled cell mismatch at ({r},{c}). "
                        f"Expected '{puzzle.grid[r][c]}', got '{letter}'."
                    )
                    return False

    # Check row uniqueness
    for r in range(7):
        row_letters = filled_grid[r]
        if len(set(row_letters)) != 7:
            print(f"Error: Row {r} has duplicates.")
            return False

    # Check column uniqueness
    for c in range(7):
        col_letters = [filled_grid[r][c] for r in range(7)]
        if len(set(col_letters)) != 7:
            print(f"Error: Column {c} has duplicates.")
            return False

    # Minor diagonal uniform
    diag_letters = [filled_grid[r][6 - r] for r in range(7)]
    if len(set(diag_letters)) != 1:
        print("Error: Minor diagonal letters are not all the same.")
        return False

    return True


############################
# Example usage
############################
if __name__ == "__main__":
    # Generate 2 puzzles in "letter_logic_dataset"
    output_dir = "../dataset_gather/letter_logic_diagram"
    generator = LetterLogicPuzzleGenerator()
    generator.generate_dataset(output_dir=output_dir, num_samples=200)

    # Load one puzzle and check an example LLM answer
    sample_id = 0
    sample_dir = os.path.join(output_dir, f"sample_{sample_id}")
    with open(os.path.join(sample_dir, "solution.json"), "r") as f:
        data = json.load(f)
    puzzle = LetterLogicPuzzle(
        grid=data["grid"],
        solution=data["solution"],
        complexity=data["complexity"]
    )

    # Suppose we have an LLM's answer (fake example)
    llm_answer = """
<<<
a,b,c,d,e,f,g
b,c,d,e,f,g,a
c,d,e,f,g,a,b
d,e,f,g,a,b,c
e,f,g,a,b,c,d
f,g,a,b,c,d,e
g,a,b,c,d,e,f
>>>
"""
    result = check_letter_logic_answer(llm_answer, puzzle)
    print("Is the LLM's answer correct?", result)
