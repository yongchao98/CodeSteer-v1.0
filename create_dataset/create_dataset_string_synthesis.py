import random
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple


# ============================================
# Data structure to hold String Synthesis puzzle info
# ============================================
@dataclass
class StringSynthesisPuzzle:
    A_count: int
    B_count: int
    C_count: int
    # The final 9 blocks: [A], [B], [C], {A}, {B}, {C}, (A), (B), (C)
    final_counts: List[int]
    # Optional: some measure of puzzle complexity or steps
    complexity: int = 1


# ============================================
# The puzzle generator class
# ============================================
class StringSynthesisPuzzleGenerator:
    """
    Generates random puzzle setups and simulates the 6 methods repeatedly
    to find the final state.
    """

    def __init__(self, max_A=5, max_B=8, max_C=7, complexity_levels=(1, 5)):
        """
        :param max_A, max_B, max_C: Maximum starting quantity for [A], [B], [C].
        :param complexity_levels: A tuple indicating the min and max complexity.
        """
        self.max_A = max_A
        self.max_B = max_B
        self.max_C = max_C
        self.min_complexity, self.max_complexity = complexity_levels

    def generate_puzzle(self) -> StringSynthesisPuzzle:
        """
        Generate a single puzzle with random counts of [A], [B], [C].
        Then simulate the methods until no more transformations can occur.
        Returns a puzzle with final state.
        """
        # 1) Randomly pick how many [A], [B], and [C]
        A_count = random.randint(1, self.max_A)
        B_count = random.randint(1, self.max_B)
        C_count = random.randint(1, self.max_C)

        # 2) Simulate the combination steps until no more steps are possible
        final_counts = self._simulate_combinations(A_count, B_count, C_count)

        # 3) Assign some complexity measure (can be simple or more advanced)
        complexity = random.randint(self.min_complexity, self.max_complexity)

        return StringSynthesisPuzzle(
            A_count=A_count,
            B_count=B_count,
            C_count=C_count,
            final_counts=final_counts,
            complexity=complexity
        )

    def _simulate_combinations(self, A, B, C) -> List[int]:
        """
        Apply the methods in a cycle:
          M1 -> M2 -> M3 -> M4 -> M5 -> M6 -> M1 -> ...
        until no more transformations can be made.

        The possible blocks are in the order:
        [A], [B], [C], {A}, {B}, {C}, (A), (B), (C).

        We'll track counts in a list:
        counts = [num_[A], num_[B], num_[C], num_{A}, num_{B}, num_{C}, num_(A), num_(B), num_(C)]

        The methods are:
          1. One [A], one [B], and one [C] can be combined to form one {A}.
          2. One [A] and one [B] can be combined to form one {C}.
          3. One [B] and one [C] can be combined to form one {B}.
          4. Two [C] can be combined to form one {C}.
          5. One {A} and one {C} can be combined to form one (A) and one (B).
          6. Two {B} can be combined to form one (C).
        """
        # Initialize
        counts = [A, B, C, 0, 0, 0, 0, 0, 0]
        # Index reference
        iA, iB, iC = 0, 1, 2
        i_A, i_B, i_C = 3, 4, 5
        pA, pB, pC = 6, 7, 8

        # Method cycle
        methods = [1, 2, 3, 4, 5, 6]
        finished = False

        # We'll limit the number of cycles to prevent infinite loops
        # in case of buggy logic or a puzzle that cycles back.
        max_iterations = 1000
        iteration_count = 0

        while not finished and iteration_count < max_iterations:
            finished = True  # assume no changes; if a method can apply, we set it to False
            iteration_count += 1

            for m in methods:
                # Try to apply each method in order
                changed = True
                while changed:
                    changed = False
                    # M1. [A], [B], [C] -> {A}
                    if m == 1:
                        if counts[iA] > 0 and counts[iB] > 0 and counts[iC] > 0:
                            counts[iA] -= 1
                            counts[iB] -= 1
                            counts[iC] -= 1
                            counts[i_A] += 1
                            changed = True
                            finished = False

                    # M2. [A], [B] -> {C}
                    elif m == 2:
                        if counts[iA] > 0 and counts[iB] > 0:
                            counts[iA] -= 1
                            counts[iB] -= 1
                            counts[i_C] += 1
                            changed = True
                            finished = False

                    # M3. [B], [C] -> {B}
                    elif m == 3:
                        if counts[iB] > 0 and counts[iC] > 0:
                            counts[iB] -= 1
                            counts[iC] -= 1
                            counts[i_B] += 1
                            changed = True
                            finished = False

                    # M4. [C], [C] -> {C}
                    elif m == 4:
                        if counts[iC] > 1:
                            counts[iC] -= 2
                            counts[i_C] += 1
                            changed = True
                            finished = False

                    # M5. {A}, {C} -> (A), (B)
                    elif m == 5:
                        if counts[i_A] > 0 and counts[i_C] > 0:
                            counts[i_A] -= 1
                            counts[i_C] -= 1
                            counts[pA] += 1
                            counts[pB] += 1
                            changed = True
                            finished = False

                    # M6. {B}, {B} -> (C)
                    elif m == 6:
                        if counts[i_B] > 1:
                            counts[i_B] -= 2
                            counts[pC] += 1
                            changed = True
                            finished = False

        return counts


# ============================================
# Generate dataset
# ============================================
def generate_string_synthesis_dataset(output_dir: str, num_samples: int = 20):
    """
    Generate a dataset of string synthesis puzzles with varying complexity.
    Each puzzle has a question (question.txt) and a solution (solution.json).
    """
    os.makedirs(output_dir, exist_ok=True)
    generator = StringSynthesisPuzzleGenerator()

    for sample_id in range(num_samples):
        puzzle = generator.generate_puzzle()
        sample_dir = os.path.join(output_dir, f"sample_{sample_id}")
        os.makedirs(sample_dir, exist_ok=True)

        # =============== 1) Format the question prompt ===============
        # Example question prompt:
        """
        There are nine different blocks [A], [B], [C], {A}, {B}, {C}, (A), (B), (C) that satisfy:

        1. One [A], one [B], and one [C] can be combined to form one {A}.
        2. One [A] and one [B] can be combined to form one {C}.
        3. One [B] and one [C] can be combined to form one {B}.
        4. Two [C] can be combined to form one {C}.
        5. One {A} and one {C} can be combined to form one (A) and one (B).
        6. Two {B} can be combined to form one (C).

        Cycle through method 1..6 repeatedly until no more changes.
        Question: If we now have <X> [A], <Y> [B], and <Z> [C], what will the synthesized block be?
        Return your final answer with the format: 
        <<<a string of the number of certain block>>> 
        specifically in order: [A], [B], [C], {A}, {B}, {C}, (A), (B), (C).
        For example: <<<1 0 3 0 0 0 0 0 0>>> means 1 [A], 0 [B], 3 [C], 0 {A}, 0 {B}, 0 {C}, 0 (A), 0 (B), 0 (C).
        """

        question_prompt = (
            "There are nine different blocks [A], [B], [C], {A}, {B}, {C}, (A), (B), (C) that satisfy:\n"
            "1. One [A], one [B], and one [C] can be combined to form one {A}.\n"
            "2. One [A] and one [B] can be combined to form one {C}.\n"
            "3. One [B] and one [C] can be combined to form one {B}.\n"
            "4. Two [C] can be combined to form one {C}.\n"
            "5. One {A} and one {C} can be combined to form one (A) and one (B).\n"
            "6. Two {B} can be combined to form one (C).\n\n"
            "Cycle through Methods 1..6 repeatedly until no more changes can be made.\n\n"
            f"Question: If we now have {puzzle.A_count} [A], {puzzle.B_count} [B], and {puzzle.C_count} [C], "
            "what will the synthesized block be?\n\n"
            "Return your final answer with the exact format:\n"
            "<<<a string of the number of certain block>>>\n"
            "Your answer:\n"
        )

        # =============== 2) Save the question ===============
        with open(os.path.join(sample_dir, 'question.txt'), 'w') as fq:
            fq.write(question_prompt)

        # =============== 3) Format the solution ===============
        # We want the solution to be in the format [A, B, C, {A}, {B}, {C}, (A), (B), (C)].
        final_str = " ".join(str(x) for x in puzzle.final_counts)
        # The official "best" or "correct" solution (the gold reference).
        solution_dict = {
            "A_count": puzzle.A_count,
            "B_count": puzzle.B_count,
            "C_count": puzzle.C_count,
            "final_counts": puzzle.final_counts,
            "complexity": puzzle.complexity,
            "solution_string": f"<<<{final_str}>>>"
        }

        with open(os.path.join(sample_dir, 'solution.json'), 'w') as fs:
            json.dump(solution_dict, fs, indent=2)

        print(
            f"Generated sample_{sample_id}: Start=({puzzle.A_count}, {puzzle.B_count}, {puzzle.C_count}) -> Final={puzzle.final_counts}")


# ============================================
# A helper function to check correctness
# ============================================
def check_string_synthesis_answer(model_answer: str, ground_truth_counts: List[int]) -> bool:
    """
    Tries to parse the model's answer (which should be in the form
    <<<a string of 9 numbers>>>) for the 9 block counts.
    If the model does not follow the format, we attempt to parse out the numbers anyway.

    :param model_answer: The string that the LLM returned.
    :param ground_truth_counts: The correct final list of 9 block counts.
    :return: True if parsed model answer matches ground_truth_counts exactly, else False.
    """

    # We'll try to find a substring that starts with <<< and ends with >>> and parse numbers from it.
    import re

    pattern = r'<<<([\s\d]+)>>>'
    match = re.search(pattern, model_answer)
    if match:
        # We found something inside <<<  >>>
        answer_str = match.group(1).strip()
    else:
        # If there's no matching format, fallback to parse any sequence of digits in the entire string
        answer_str = model_answer

    # Now parse out the integer tokens
    tokens = re.findall(r'\d+', answer_str)
    if len(tokens) != 9:
        # If we do not have exactly 9 numbers, this is considered incorrect
        return False

    # Convert each to int
    parsed_counts = list(map(int, tokens))
    # Compare to ground_truth_counts
    return parsed_counts == ground_truth_counts


# ============================================
# Example usage
# ============================================
if __name__ == "__main__":
    # Example: generate 10 samples in an output directory
    output_dir = "../dataset_gather/string_synthesis"
    generate_string_synthesis_dataset(output_dir, num_samples=200)

    # Example usage of the check correctness function:
    # Suppose the gold final counts = [4, 2, 3, 1, 0, 0, 0, 0, 0]
    gold_counts = [4, 2, 3, 1, 0, 0, 0, 0, 0]

    # Two possible model answers:
    correct_answer = "<<<4 2 3 1 0 0 0 0 0>>>"
    incorrect_answer = "I think the answer is 4 A, 2 B, 2 C, 1 {A}, 0 {B}, 0 {C}, 0 (A), 0 (B), 0 (C)."

    print("Should be True: ", check_string_synthesis_answer(correct_answer, gold_counts))
    print("Should be False:", check_string_synthesis_answer(incorrect_answer, gold_counts))
