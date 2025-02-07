import random
import os
import json
from typing import List, Tuple
from dataclasses import dataclass
import string
import re

@dataclass
class MatrixPuzzle:
    matrix: List[List[str]]
    solution: List[List[str]]
    complexity: int  # 1-5 scale based on matrix size and content type

class MatrixRotationGenerator:
    def __init__(self):
        self.letters = string.ascii_uppercase
        self.numbers = string.digits[1:]  # excluding 0
        self.symbols = "!@#$%^&*"
        
    def rotate_matrix(self, matrix: List[List[str]]) -> List[List[str]]:
        """Rotate a matrix 90 degrees clockwise"""
        n = len(matrix)
        m = len(matrix[0])
        result = [['' for _ in range(n)] for _ in range(m)]
        
        for i in range(n):
            for j in range(m):
                result[j][n-1-i] = matrix[i][j]
                
        return result
    
    def format_solution(self, matrix: List[List[str]]) -> str:
        """Format the solution as a string with the required format"""
        flattened = []
        for row in matrix:
            flattened.extend(row)
        return ','.join(flattened)
    
    def generate_random_matrix(self, rows: int, cols: int, content_type: str) -> List[List[str]]:
        """Generate a random matrix with specified content type"""
        matrix = []
        for _ in range(rows):
            row = []
            for _ in range(cols):
                if content_type == 'letters':
                    char = random.choice(self.letters)
                elif content_type == 'numbers':
                    char = random.choice(self.numbers)
                elif content_type == 'mixed':
                    char = random.choice(self.letters + self.numbers)
                else:  # symbols
                    char = random.choice(self.symbols)
                row.append(char)
            matrix.append(row)
        return matrix
    
    def _calculate_complexity(self, rows: int, cols: int, content_type: str) -> int:
        """Calculate puzzle complexity on a scale of 1-5"""
        # Base complexity from matrix size
        size_complexity = min((rows * cols) // 4, 3)
        
        # Additional complexity from content type
        content_complexity = {
            'letters': 1,
            'numbers': 1,
            'mixed': 2,
            'symbols': 2
        }[content_type]
        
        return min(size_complexity + content_complexity, 5)
    
    def generate_puzzle(self, rows: int, cols: int, content_type: str) -> MatrixPuzzle:
        """Generate a puzzle with specified dimensions and content type"""
        matrix = self.generate_random_matrix(rows, cols, content_type)
        solution = self.rotate_matrix(matrix)
        complexity = self._calculate_complexity(rows, cols, content_type)
        
        return MatrixPuzzle(
            matrix=matrix,
            solution=solution,
            complexity=complexity
        )

def check_answer(answer: str, solution: List[List[str]]) -> Tuple[bool, str]:
    """Check if the provided answer matches the solution"""
    # Try to extract content between <<< and >>>
    match = re.search(r'<<<(.+?)>>>', answer)
    if not match:
        return False, "Answer not in required format <<<...>>>"
    
    try:
        # Extract and clean the answer content
        answer_content = match.group(1)
        # Remove any brackets and split by commas
        answer_content = answer_content.replace('[', '').replace(']', '')
        answer_values = [x.strip() for x in answer_content.split(',')]
        
        # Convert solution matrix to flat list
        solution_values = []
        for row in solution:
            solution_values.extend(row)
            
        return answer_values == solution_values, "Answer values don't match solution"
    except Exception as e:
        return False, f"Error processing answer: {str(e)}"

def generate_dataset(output_dir: str):
    """Generate a dataset of matrix rotation puzzles with varying complexity"""
    os.makedirs(output_dir, exist_ok=True)
    generator = MatrixRotationGenerator()
    
    # Define different configurations for variety
    configurations = [
        # (rows, cols, content_type, num_samples)
        (2, 2, 'letters', 20),
        (2, 3, 'numbers', 20),
        (3, 3, 'letters', 20),
        (3, 4, 'mixed', 20),
        (4, 4, 'symbols', 20),
        (2, 2, 'letters', 20),
        (2, 3, 'numbers', 20),
        (3, 3, 'letters', 20),
        (3, 4, 'mixed', 20),
        (4, 4, 'symbols', 20)
    ]
    
    sample_id = 0
    for rows, cols, content_type, num_samples in configurations:
        for _ in range(num_samples):
            puzzle = generator.generate_puzzle(rows, cols, content_type)
            
            # Create sample directory
            sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Format matrix as string
            matrix_str = '\n'.join(' '.join(row) for row in puzzle.matrix)
            
            # Format question
            question = (
                f"Given a 2D array of strings, which represents a letter matrix. "
                f"Your task is to rotate this matrix 90 degrees clockwise and output the result.\n"
                f"If the matrix is \n{matrix_str}.\n"
                f"What is the result?\n"
                f"Return your answer with format <<<list of values and symbols>>>, "
                f"For example, <<<A,B,C,D>>> represents a 2D array."
            )
            
            # Save question and solution
            with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                f.write(question)
                
            with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                json.dump({
                    'matrix': puzzle.matrix,
                    'solution': puzzle.solution,
                    'formatted_solution': generator.format_solution(puzzle.solution),
                    'complexity': puzzle.complexity
                }, f, indent=2)
                
            sample_id += 1
            print(f"Generated sample {sample_id}")

if __name__ == "__main__":
    output_dir = '../dataset_gather/matrix_transformation_dataset'
    generate_dataset(output_dir)