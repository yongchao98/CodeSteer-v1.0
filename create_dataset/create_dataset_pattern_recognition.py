import random
import os
import json
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class PatternRecognitionSample:
    matrix: List[List[str]]
    solution: Tuple[int, int]
    complexity: int  # Complexity level from 1 to 5


class PatternRecognitionGenerator:
    def __init__(self):
        self.characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.min_size = 6  # Increased minimum matrix size for higher difficulty
        self.max_size = 15  # Increased maximum matrix size for higher difficulty
        self.min_square = 3  # Increased minimum square side length for higher difficulty

    def generate_unique_matrix(self, rows: int, cols: int, square_size: int) -> Tuple[List[List[str]], Tuple[int, int]]:
        """
        Generate a character matrix containing a single square of the specified size.
        Returns the matrix and the bottom-right corner coordinates (1-based indexing).
        """
        # Initialize matrix with random characters
        matrix = [[random.choice(self.characters) for _ in range(cols)] for _ in range(rows)]

        # Choose a character for the square
        square_char = random.choice(self.characters)

        # Choose the top-left corner for the square
        max_row = rows - square_size
        max_col = cols - square_size
        top_left_row = random.randint(0, max_row)
        top_left_col = random.randint(0, max_col)

        # Place the square in the matrix
        for r in range(top_left_row, top_left_row + square_size):
            for c in range(top_left_col, top_left_col + square_size):
                matrix[r][c] = square_char

        # Calculate bottom-right corner coordinates (0-based indexing)
        bottom_right = (top_left_row + square_size - 1, top_left_col + square_size - 1)

        # Ensure there are no other squares of size >=3 in the matrix
        for r in range(rows - self.min_square + 1):
            for c in range(cols - self.min_square + 1):
                if self.is_square(matrix, r, c):
                    # If another square is found, regenerate the matrix
                    return self.generate_unique_matrix(rows, cols, square_size)

        return matrix, bottom_right

    def is_square(self, matrix: List[List[str]], row: int, col: int) -> bool:
        """
        Check if there's a square of at least size 3 starting at (row, col).
        """
        char = matrix[row][col]
        size = 1
        rows = len(matrix)
        cols = len(matrix[0])
        while row + size < rows and col + size < cols:
            # Check the new row for the current size
            for c in range(col, col + size + 1):
                if matrix[row + size][c] != char:
                    return False
            # Check the new column for the current size
            for r in range(row, row + size + 1):
                if matrix[r][col + size] != char:
                    return False
            size += 1
        return size > self.min_square  # Ensure the square is at least min_square size

    def calculate_complexity(self, rows: int, cols: int, square_size: int) -> int:
        """
        Calculate the complexity based on matrix size and square size ratio.
        """
        area = rows * cols
        ratio = square_size / min(rows, cols)
        if area <= 81 and ratio <= 0.2:
            return 1
        elif area <= 144 and ratio <= 0.25:
            return 2
        elif area <= 225 and ratio <= 0.33:
            return 3
        elif area <= 324 and ratio <= 0.4:
            return 4
        else:
            return 5

    def generate_sample(self) -> PatternRecognitionSample:
        """
        Generate a single sample with varying complexity.
        """
        rows = random.randint(self.min_size, self.max_size)
        cols = random.randint(self.min_size, self.max_size)
        max_square_size = min(rows, cols) // 2  # Ensure square size is manageable
        square_size = random.randint(self.min_square, max_square_size)
        matrix, bottom_right = self.generate_unique_matrix(rows, cols, square_size)
        complexity = self.calculate_complexity(rows, cols, square_size)
        return PatternRecognitionSample(matrix=matrix, solution=bottom_right, complexity=complexity)


def format_matrix(matrix: List[List[str]]) -> str:
    """
    Format the matrix into a string for the question prompt.
    """
    return "\n".join([" ".join(row) for row in matrix])


def generate_dataset(output_dir: str, num_samples: int = 100):
    """
    Generate a dataset of pattern recognition samples with varying complexity.
    Each sample is saved in its own directory containing 'question.txt' and 'solution.json'.
    """
    os.makedirs(output_dir, exist_ok=True)
    generator = PatternRecognitionGenerator()
    dataset = []

    for sample_id in range(num_samples):
        sample = generator.generate_sample()
        matrix_str = format_matrix(sample.matrix)
        question = (
            f"Given a character matrix, your task is to identify the positions of all squares with a length of at least {generator.min_square} "
            f"and output the row and column numbers of the bottom-right corner of these squares. The squares are composed "
            f"of the same character, and each side has a length of at least {generator.min_square}.\n\n"
            f"If the matrix is\n\n{matrix_str}\n\n"
            f"Please provide the row and column of the bottom-right vertex of the square in the character matrix with a side length of at least {generator.min_square} "
            f"(the answer is unique). Notice: All coordinates start from 0.  \n\n"
            f"Your answer should be in the format <<<[row, column]>>>."
        )
        solution = sample.solution  # (row, column), 0-based indexing

        # Create sample directory
        sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
        os.makedirs(sample_dir, exist_ok=True)

        # Save question to 'question.txt'
        with open(os.path.join(sample_dir, 'question.txt'), 'w', encoding='utf-8') as f:
            f.write(question)

        # Save solution to 'solution.json'
        with open(os.path.join(sample_dir, 'solution.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'row': solution[0],
                'column': solution[1],
                'complexity': sample.complexity
            }, f, ensure_ascii=False, indent=2)

        # Optionally, append to dataset list for further use
        dataset.append({
            'question': question,
            'solution': {
                'row': solution[0],
                'column': solution[1],
                'complexity': sample.complexity
            }
        })

        if (sample_id + 1) % 10 == 0:
            print(f"Generated {sample_id + 1}/{num_samples} samples.")

    # Save the entire dataset to a JSON file if needed
    with open(os.path.join(output_dir, 'pattern_recognition_dataset.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Dataset generation completed. Saved to directory: {output_dir}")


def parse_answer(answer: str) -> Tuple[int, int]:
    """
    Parse the TaskLLM's answer and extract the row and column numbers.
    Expected format: <<<[row, column]>>>
    If the format does not match, attempt to extract numbers.
    """
    import re
    pattern = r'<<<\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*>>>'
    match = re.search(pattern, answer)
    if match:
        row, col = match.groups()
        return int(row), int(col)
    else:
        # Attempt to extract numbers if format does not match
        numbers = re.findall(r'\d+', answer)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        else:
            raise ValueError("Unable to parse the answer format.")


def check_answer(sample: PatternRecognitionSample, answer: str) -> bool:
    """
    Check the correctness of the TaskLLM's answer.
    Returns True if correct, False otherwise.
    """
    try:
        predicted = parse_answer(answer)
        return predicted == sample.solution
    except Exception as e:
        print(f"Error parsing answer: {e}")
        return False


def example_usage():
    """
    Example usage: Generate a dataset and check a sample answer.
    """
    # Generate dataset
    output_dir = '../dataset_gather/pattern_recognition'
    generate_dataset(output_dir, num_samples=100)

if __name__ == "__main__":
    example_usage()
