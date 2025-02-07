import random
import os
import json
import itertools
from typing import List
from dataclasses import dataclass
import string
import re

@dataclass
class Password:
    numbers: List[int]
    letters: List[str]

    def __str__(self):
        return ''.join(map(str, self.numbers)) + ''.join(self.letters)

@dataclass
class GuessFeedback:
    guess: str
    feedback: str

@dataclass
class CryptanalysisProblem:
    password: Password
    guesses_feedback: List[GuessFeedback]
    complexity: int  # 1-5 scale based on number of guesses and feedback details

class CryptanalysisDatasetGenerator:
    def __init__(self):
        self.letters = list(string.ascii_uppercase)
        self.max_number = 9  # Maximum number allowed in passwords
        self.min_number = 0  # Minimum number allowed in passwords

    def generate_password(self) -> Password:
        """Generate a password with two unique numbers and two unique letters."""
        numbers = random.sample(range(self.min_number, self.max_number + 1), 2)
        letters = random.sample(self.letters, 2)
        return Password(numbers=numbers, letters=letters)

    def generate_guess(self, password: Password) -> str:
        """
        Generate a guess ensuring no repeated numbers or letters.
        The guess should not be the same as the password.
        """
        while True:
            guess_numbers = random.sample(range(self.min_number, self.max_number + 1), 2)
            guess_letters = random.sample(self.letters, 2)
            # Ensure the guess is not the password
            if guess_numbers != password.numbers or guess_letters != password.letters:
                return ''.join(map(str, guess_numbers)) + ''.join(guess_letters)

    def compute_feedback(self, password: Password, guess: str) -> str:
        """
        Compute feedback based on the guess compared to the password.
        Feedback includes information about correctness and positional accuracy.
        """
        guess_numbers = list(map(int, guess[:2]))
        guess_letters = list(guess[2:])

        password_numbers = password.numbers
        password_letters = password.letters

        # Numbers feedback
        numbers_feedback = self._get_numbers_feedback(password_numbers, guess_numbers)

        # Letters feedback
        letters_feedback = self._get_letters_feedback(password_letters, guess_letters)

        # Combine feedback
        feedback = f"{numbers_feedback}; {letters_feedback}."
        return feedback

    def _get_numbers_feedback(self, password_numbers: List[int], guess_numbers: List[int]) -> str:
        """Generate feedback for the number part of the guess."""
        feedback_parts = []
        correct = 0
        wrong_position = 0
        incorrect_descriptions = []

        for i, num in enumerate(guess_numbers):
            if num == password_numbers[i]:
                correct += 1
            elif num in password_numbers:
                wrong_position += 1
            else:
                if num < min(password_numbers):
                    incorrect_descriptions.append("too small")
                elif num > max(password_numbers):
                    incorrect_descriptions.append("too large")
                else:
                    incorrect_descriptions.append("incorrect")

        # Construct numbers feedback
        if correct > 0:
            if correct == 1:
                feedback_parts.append("one number is correct and in the correct position")
            elif correct == 2:
                feedback_parts.append("both numbers are correct and in the correct positions")

        if wrong_position > 0:
            if wrong_position == 1:
                feedback_parts.append("one number is correct but in the wrong position")
            elif wrong_position == 2:
                feedback_parts.append("both numbers are correct but in the wrong positions")

        # Handle incorrect descriptions
        if incorrect_descriptions:
            if len(incorrect_descriptions) == 2:
                # Check if both are too small or too large
                if all(desc == "too small" for desc in incorrect_descriptions):
                    feedback_parts.append("both numbers are incorrect and too small")
                elif all(desc == "too large" for desc in incorrect_descriptions):
                    feedback_parts.append("both numbers are incorrect and too large")
                else:
                    feedback_parts.append("both numbers are incorrect")
            elif len(incorrect_descriptions) == 1:
                feedback_parts.append(f"one number is incorrect and {incorrect_descriptions[0]}")

        if not feedback_parts:
            feedback_parts.append("numbers feedback unavailable")

        return "; ".join(feedback_parts)

    def _get_letters_feedback(self, password_letters: List[str], guess_letters: List[str]) -> str:
        """Generate feedback for the letter part of the guess."""
        feedback_parts = []
        correct = 0
        wrong_position = 0
        incorrect_descriptions = []

        for i, letter in enumerate(guess_letters):
            if letter == password_letters[i]:
                correct += 1
            elif letter in password_letters:
                wrong_position += 1
            else:
                pwd_positions = [self.letters.index(l) for l in password_letters]
                guess_pos = self.letters.index(letter)
                if guess_pos < min(pwd_positions):
                    incorrect_descriptions.append("too early in the alphabet")
                elif guess_pos > max(pwd_positions):
                    incorrect_descriptions.append("too late in the alphabet")
                else:
                    incorrect_descriptions.append("incorrect")

        # Construct letters feedback
        if correct > 0:
            if correct == 1:
                feedback_parts.append("one letter is correct and in the correct position")
            elif correct == 2:
                feedback_parts.append("both letters are correct and in the correct positions")

        if wrong_position > 0:
            if wrong_position == 1:
                feedback_parts.append("one letter is correct but in the wrong position")
            elif wrong_position == 2:
                feedback_parts.append("both letters are correct but in the wrong positions")

        # Handle incorrect descriptions
        if incorrect_descriptions:
            if len(incorrect_descriptions) == 2:
                # Check if both are too early or too late
                if all(desc == "too early in the alphabet" for desc in incorrect_descriptions):
                    feedback_parts.append("both letters are incorrect and too early in the alphabet")
                elif all(desc == "too late in the alphabet" for desc in incorrect_descriptions):
                    feedback_parts.append("both letters are incorrect and too late in the alphabet")
                else:
                    feedback_parts.append("both letters are incorrect")
            elif len(incorrect_descriptions) == 1:
                feedback_parts.append(f"one letter is incorrect and {incorrect_descriptions[0]}")

        if not feedback_parts:
            feedback_parts.append("letters feedback unavailable")

        return "; ".join(feedback_parts)

    def generate_guesses_feedback(self, password: Password, num_guesses: int) -> List[GuessFeedback]:
        """
        Generate a list of guesses along with their feedback.
        Ensures that the set of feedbacks leads to a unique solution.
        """
        guesses_feedback = []
        attempted_guesses = set()

        while len(guesses_feedback) < num_guesses:
            guess = self.generate_guess(password)
            if guess in attempted_guesses:
                continue  # Avoid duplicate guesses
            attempted_guesses.add(guess)
            feedback = self.compute_feedback(password, guess)
            guesses_feedback.append(GuessFeedback(guess=guess, feedback=feedback))

            # After adding each feedback, check if the current set leads to a unique password
            possible_passwords = self.find_possible_passwords(guesses_feedback)
            if len(possible_passwords) == 1:
                break  # Unique solution found

        return guesses_feedback

    def calculate_complexity(self, num_guesses: int) -> int:
        """
        Calculate complexity based on the number of guesses.
        Higher number of guesses increases complexity.
        """
        if num_guesses <= 3:
            return 1
        elif num_guesses <= 5:
            return 2
        elif num_guesses <= 7:
            return 3
        elif num_guesses <= 9:
            return 4
        else:
            return 5

    def find_possible_passwords(self, guesses_feedback: List[GuessFeedback]) -> List[Password]:
        """
        Find all possible passwords that satisfy all the feedbacks.
        """
        possible_passwords = []
        # Generate all possible unique passwords
        all_numbers = list(itertools.permutations(range(self.min_number, self.max_number + 1), 2))
        all_letters = list(itertools.permutations(self.letters, 2))

        total = len(all_numbers) * len(all_letters)
        count = 0

        for num_pair in all_numbers:
            for let_pair in all_letters:
                candidate = Password(numbers=list(num_pair), letters=list(let_pair))
                # Check if candidate satisfies all feedbacks
                if all(self._match_feedback(candidate, gf) for gf in guesses_feedback):
                    possible_passwords.append(candidate)
                count += 1
                # Optional: Progress indicator for large datasets
                if count % 1000 == 0:
                    print(f"Checked {count}/{total} possible passwords...", end='\r')

        return possible_passwords

    def _match_feedback(self, password: Password, guess_feedback: GuessFeedback) -> bool:
        """
        Check if a password matches the feedback for a single guess.
        """
        expected_feedback = guess_feedback.feedback
        actual_feedback = self.compute_feedback(password, guess_feedback.guess)
        return expected_feedback == actual_feedback

    def generate_unique_problem(self, num_guesses: int) -> CryptanalysisProblem:
        """
        Generate a cryptanalysis problem with a unique solution.
        Retries until a unique solution is found.
        """
        attempts = 0
        while True:
            password = self.generate_password()
            guesses_feedback = self.generate_guesses_feedback(password, num_guesses)
            possible_passwords = self.find_possible_passwords(guesses_feedback)
            if len(possible_passwords) == 1 and \
               possible_passwords[0].numbers == password.numbers and \
               possible_passwords[0].letters == password.letters:
                complexity = self.calculate_complexity(num_guesses)
                return CryptanalysisProblem(password=password, guesses_feedback=guesses_feedback, complexity=complexity)
            attempts += 1
            if attempts >= 100:
                raise Exception("Failed to generate a unique problem after 100 attempts. Consider increasing the number of guesses.")

def generate_dataset(output_dir: str, num_samples: int = 100):
    """
    Generate a dataset of cryptanalysis problems with unique solutions.

    Args:
        output_dir (str): Directory where the dataset will be saved.
        num_samples (int): Number of samples to generate.
    """
    os.makedirs(output_dir, exist_ok=True)
    generator = CryptanalysisDatasetGenerator()

    # Define different configurations for variety
    # Each tuple: (min_guesses, max_guesses)
    configurations = [
        (4, 6),
        (7, 9),
        (10, 12),
        (13, 15),
        (16, 18)
    ]

    samples_per_config = num_samples // len(configurations)
    sample_id = 0

    for config in configurations:
        min_guesses, max_guesses = config
        for _ in range(samples_per_config):
            num_guesses = random.randint(min_guesses, max_guesses)
            try:
                problem = generator.generate_unique_problem(num_guesses)
            except Exception as e:
                print(f"Skipping sample due to error: {e}")
                continue

            # Format the question prompt
            conditions = ""
            for idx, gf in enumerate(problem.guesses_feedback, 1):
                conditions += f"{idx}. Guess: {gf.guess}, Feedback: {gf.feedback}\n"

            question = (
                f"Please deduce the correct password based on the following {len(problem.guesses_feedback)} known conditions:\n"
                f"A certain combination lock consists of two numbers and two letters, with neither the numbers nor the letters repeating.\n"
                f"{conditions}"
                f"Please provide your answer in the format <<< [\"X\", \"Y\", \"Z\", \"W\"] >>> where X and Y are the numbers and Z and W are the letters of the password.\n"
            )

            # Define the solution as the correct password
            solution = ''.join(map(str, problem.password.numbers)) + ''.join(problem.password.letters)

            # Create sample directory
            sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
            os.makedirs(sample_dir, exist_ok=True)

            # Save question
            with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                f.write(question)

            # Save solution
            with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                json.dump({
                    'answer': [str(problem.password.numbers[0]),
                               str(problem.password.numbers[1]),
                               problem.password.letters[0],
                               problem.password.letters[1]]
                }, f, indent=2)

            sample_id += 1
            print(f"Generated sample {sample_id}: Password={solution}")

    print(f"Dataset generation complete. Total samples: {sample_id}")

def verify_answer(taskllm_output: str, correct_password: str) -> bool:
    """
    Verify the correctness of the TaskLLM's answer.

    Args:
        taskllm_output (str): The raw output from the TaskLLM.
        correct_password (str): The correct password to compare against.

    Returns:
        bool: True if the answer is correct, False otherwise.
    """
    # Use regex to extract the content within <<< and >>>
    match = re.search(r'<<<\s*(.*?)\s*>>>', taskllm_output, re.DOTALL)
    if not match:
        print("Answer format incorrect. Expected format <<< [\"X\", \"Y\", \"Z\", \"W\"] >>>.")
        return False

    extracted_answer = match.group(1).strip()

    # Attempt to parse the answer as JSON
    try:
        answer_list = json.loads(extracted_answer)
        if not isinstance(answer_list, list) or len(answer_list) != 4:
            print("Answer should be a list of four strings representing the password (two numbers and two letters).")
            return False
        deduced_password = ''.join(answer_list)
    except json.JSONDecodeError:
        # If not JSON, assume it's a plain string
        deduced_password = extracted_answer

    if deduced_password == correct_password:
        print("The TaskLLM's answer is correct.")
        return True
    else:
        print(f"The TaskLLM's answer is incorrect. Expected: {correct_password}, Got: {deduced_password}")
        return False

if __name__ == "__main__":
    # Example usage:
    # Generate a dataset of 100 samples in the specified directory
    output_directory = '../dataset_gather/cryptanalysis'
    generate_dataset(output_directory, num_samples=100)

    # Example verification
    # Suppose the correct password is '75LV' and the TaskLLM output is '<<< ["7", "5", "L", "V"] >>>'
    example_output = '<<< ["7", "5", "L", "V"] >>>'
    correct_pw = '75LV'
    verify_answer(example_output, correct_pw)
