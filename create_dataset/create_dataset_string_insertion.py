import random
import os
import json
import re
from typing import List
from dataclasses import dataclass
import math


@dataclass
class StringInsertionSample:
    original_string: str
    question: str
    solution: str
    complexity: int  # Scale from 1-5 based on number of insertions and string length


class StringInsertionDatasetGenerator:
    def __init__(self):
        # Define the characters to be used in the strings
        self.characters = ['A', 'B', 'C', 'D', 'E']

        # Define the insertion rules as (substring, character to insert)
        self.insertion_rules = [
            ('ABCD', 'A'),
            ('BCDE', 'B'),
            ('CDEA', 'C'),
            ('DEAB', 'D'),
            ('EABC', 'E')
        ]

        # Ensure each generated string has exactly three insertions
        self.num_insertions = 3

    def generate_string(self) -> str:
        """
        Generate a string that contains exactly three substrings triggering insertions.
        """
        # Select three insertion rules, allowing repetitions
        selected_rules = [random.choice(self.insertion_rules) for _ in range(self.num_insertions)]

        # Initialize the string with the first substring
        generated = selected_rules[0][0]

        # Append the remaining substrings with fixed separators to avoid overlaps
        for rule in selected_rules[1:]:
            # Use a single random character as a separator
            separator = random.choice(self.characters)
            generated += separator + rule[0]

        # Optionally, add a fixed number of random characters before and after to keep string length reasonable
        prefix_length = random.randint(0, 2)
        suffix_length = random.randint(0, 2)
        prefix = ''.join(random.choices(self.characters, k=prefix_length))
        suffix = ''.join(random.choices(self.characters, k=suffix_length))

        full_string = prefix + generated + suffix

        # Final verification to ensure the string contains exactly three insertion substrings
        if self.count_trigger_substrings(full_string) != self.num_insertions:
            # If not, recursively generate a new string
            return self.generate_string()

        return full_string

    def count_trigger_substrings(self, s: str) -> int:
        """
        Count the number of trigger substrings present in the string.
        """
        count = 0
        for substr, _ in self.insertion_rules:
            count += s.count(substr)
        return count

    def apply_operation_T(self, s: str) -> str:
        """
        Apply operation T to the string s by performing insertions from left to right.
        """
        # List of tuples (substring, character to insert)
        rules = self.insertion_rules

        # To keep track of insertions, store indices where insertions should occur
        insertions = []
        i = 0
        while i <= len(s) - 4:
            substr = s[i:i + 4]
            for rule_substr, insert_char in rules:
                if substr == rule_substr:
                    # Record the position after the substring for insertion
                    insertions.append((i + 4, insert_char))
                    break  # Only one insertion per position
            i += 1

        # Perform insertions from left to right, adjusting positions for previous insertions
        offset = 0
        for pos, char in insertions:
            s = s[:pos + offset] + char + s[pos + offset:]
            offset += 1  # Increment offset due to the insertion

        return s

    def calculate_complexity(self, original: str, modified: str) -> int:
        """
        Calculate complexity based on number of insertions and string length.
        """
        num_insertions = len(modified) - len(original)
        length = len(original)

        # Complexity factors:
        # - Number of insertions contributes directly
        # - String length contributes logarithmically
        complexity = num_insertions + math.ceil(math.log2(length)) if length > 1 else num_insertions

        # Scale complexity to be within 1-5
        return min(max(complexity, 1), 5)

    def generate_sample(self) -> StringInsertionSample:
        """
        Generate a single dataset sample.
        """
        original = self.generate_string()
        solution = self.apply_operation_T(original)
        complexity = self.calculate_complexity(original, solution)

        question = (
            f"Given the string consisting of characters A, B, C, D, and E, define the T operation as follows:\n\n"
            f"1. If there is a substring ABCD in the string, insert the character A after that substring.\n"
            f"2. If there is a substring BCDE in the string, insert the character B after that substring.\n"
            f"3. If there is a substring CDEA in the string, insert the character C after that substring.\n"
            f"4. If there is a substring DEAB in the string, insert the character D after that substring.\n"
            f"5. If there is a substring EABC in the string, insert the character E after that substring.\n\n"
            f"Question: Given the string {original}, what will the string be after one operation T? String insertion from left to right.\n"
            f"Return the answer with the format <<<a string>>>."
        )

        return StringInsertionSample(
            original_string=original,
            question=question,
            solution=solution,
            complexity=complexity
        )

    def generate_dataset(self, output_dir: str, num_samples: int = 200):
        """
        Generate a dataset of string insertion samples and save them to the specified directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        for sample_id in range(num_samples):
            sample = self.generate_sample()
            sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
            os.makedirs(sample_dir, exist_ok=True)

            # Save question
            with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                f.write(sample.question)

            # Save solution
            with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                json.dump({
                    'original_string': sample.original_string,
                    'modified_string': sample.solution,
                    'complexity': sample.complexity
                }, f, indent=2)

            print(f"Generated sample {sample_id}: {sample.original_string} -> {sample.solution}")


def check_answer(taskllm_answer: str, expected_solution: str) -> bool:
    """
    Check the correctness of TaskLLM's answer.
    Extract the string from <<< and >>> if present, else assume the entire answer is the string.
    """
    # Use regex to extract the string within <<< and >>>
    match = re.search(r'<<<\s*(.*?)\s*>>>', taskllm_answer)
    if match:
        answer_str = match.group(1)
    else:
        # If format not followed, assume the entire answer is the string
        answer_str = taskllm_answer.strip()

    # Compare the extracted string with the expected solution
    return answer_str == expected_solution


def example_usage():
    """
    Example usage of the dataset generator and answer checker.
    """
    generator = StringInsertionDatasetGenerator()
    output_directory = '../dataset_gather/string_insertion'
    num_samples = 200  # Adjust as needed
    generator.generate_dataset(output_directory, num_samples)

    # Generate a single sample for demonstration
    sample = generator.generate_sample()
    print("\nSample Question:")
    print(sample.question)
    print("\nExpected Solution:")
    print(f"<<<{sample.solution}>>>")

    # Simulate TaskLLM's answer in different formats
    correct_answer = f"<<<{sample.solution}>>>"
    incorrect_format_answer = sample.solution  # Without <<< and >>>
    wrong_answer = "Some incorrect string"

    # Check correctness
    print("\nChecking Correct Answer:")
    print(check_answer(correct_answer, sample.solution))  # Should return True

    print("\nChecking Incorrect Format Answer:")
    print(check_answer(incorrect_format_answer, sample.solution))  # Should return True

    print("\nChecking Wrong Answer:")
    print(check_answer(wrong_answer, sample.solution))  # Should return False


if __name__ == "__main__":
    example_usage()
