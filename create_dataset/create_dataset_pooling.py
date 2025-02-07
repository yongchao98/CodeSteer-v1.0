import random
import os
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
import re

@dataclass
class PoolingPuzzle:
    matrix: List[List[int]]
    window_size: int
    pooling_type: str  # 'max' or 'average'
    solution: List[List[float]]
    complexity: int  # 1-5 scale based on matrix size and window size

class PoolingPuzzleGenerator:
    def __init__(self):
        self.min_value = -10
        self.max_value = 20
        
    def generate_matrix(self, size: int) -> List[List[int]]:
        """Generate a random matrix of given size"""
        return [[random.randint(self.min_value, self.max_value) 
                for _ in range(size)] for _ in range(size)]
    
    def perform_max_pooling(self, matrix: List[List[int]], window_size: int) -> List[List[float]]:
        """Perform max pooling operation on the matrix"""
        n = len(matrix)
        result_size = n - window_size + 1
        result = [[0.0] * result_size for _ in range(result_size)]
        
        for i in range(result_size):
            for j in range(result_size):
                window = []
                for wi in range(window_size):
                    for wj in range(window_size):
                        window.append(matrix[i + wi][j + wj])
                result[i][j] = max(window)
        
        return result
    
    def perform_average_pooling(self, matrix: List[List[int]], window_size: int) -> List[List[float]]:
        """Perform average pooling operation on the matrix"""
        n = len(matrix)
        result_size = n - window_size + 1
        result = [[0.0] * result_size for _ in range(result_size)]
        
        for i in range(result_size):
            for j in range(result_size):
                window = []
                for wi in range(window_size):
                    for wj in range(window_size):
                        window.append(matrix[i + wi][j + wj])
                result[i][j] = sum(window) / (window_size * window_size)
        
        return result

    def _calculate_complexity(self, matrix_size: int, window_size: int) -> int:
        """Calculate puzzle complexity on a scale of 1-5"""
        # Base complexity from matrix size
        complexity = matrix_size // 2
        
        # Additional complexity from window size
        if window_size > 2:
            complexity += 1
            
        # Additional complexity if matrix is large relative to window
        if matrix_size - window_size > 3:
            complexity += 1
            
        return min(max(complexity, 1), 5)

    def generate_puzzle(self, matrix_size: int, window_size: int, 
                       pooling_type: str) -> PoolingPuzzle:
        """Generate a pooling puzzle with specified parameters"""
        matrix = self.generate_matrix(matrix_size)
        
        if pooling_type == 'max':
            solution = self.perform_max_pooling(matrix, window_size)
        else:
            solution = self.perform_average_pooling(matrix, window_size)
            
        complexity = self._calculate_complexity(matrix_size, window_size)
        
        return PoolingPuzzle(
            matrix=matrix,
            window_size=window_size,
            pooling_type=pooling_type,
            solution=solution,
            complexity=complexity
        )

def format_matrix_as_string(matrix: List[List[int]]) -> str:
    """Format matrix as comma-separated string with line breaks"""
    return '\n'.join([', '.join(map(str, row)) for row in matrix])

def check_llm_answer(answer: str, expected_solution: List[List[float]], 
                    tolerance: float = 1e-6) -> Tuple[bool, str]:
    """Check if LLM's answer matches the expected solution"""
    # Try to extract content between <<< and >>>
    match = re.search(r'<<<(.+?)>>>', answer, re.DOTALL)
    if not match:
        return False, "Answer not in required format <<<...>>>"
    
    content = match.group(1)
    
    # Try to parse the content as a 2D array
    try:
        # Remove any whitespace and split by commas
        values = content.replace(' ', '').replace('\n', '').split(',')
        
        # Convert to floats
        parsed_values = [float(v) for v in values]
        
        # Reshape into 2D array
        size = int(len(parsed_values) ** 0.5)
        if len(parsed_values) != size * size:
            return False, "Invalid array dimensions"
            
        parsed_solution = [parsed_values[i:i+size] for i in range(0, len(parsed_values), size)]
        
        # Check dimensions
        if len(parsed_solution) != len(expected_solution) or \
           len(parsed_solution[0]) != len(expected_solution[0]):
            return False, "Solution dimensions don't match"
        
        # Check values within tolerance
        for i in range(len(expected_solution)):
            for j in range(len(expected_solution[0])):
                if abs(parsed_solution[i][j] - expected_solution[i][j]) > tolerance:
                    return False, f"Value mismatch at position ({i},{j})"
        
        return True, "Correct solution"
        
    except Exception as e:
        return False, f"Error parsing answer: {str(e)}"

def generate_dataset(output_dir: str):
    """Generate a dataset of pooling puzzles with varying complexity"""
    os.makedirs(output_dir, exist_ok=True)
    generator = PoolingPuzzleGenerator()
    
    # Define different configurations for variety
    configurations = [
        # (matrix_size, window_size, pooling_type, num_samples)
        (3, 2, 'max', 18),
        (4, 2, 'max', 18),
        (5, 3, 'max', 14),
        (4, 2, 'average', 18),
        (5, 3, 'average', 18),
        (6, 3, 'average', 14),
        (3, 2, 'max', 18),
        (4, 2, 'max', 18),
        (5, 3, 'max', 14),
        (4, 2, 'average', 18),
        (5, 3, 'average', 18),
        (6, 3, 'average', 14),
    ]
    
    sample_id = 0
    for matrix_size, window_size, pooling_type, num_samples in configurations:
        for _ in range(num_samples):
            puzzle = generator.generate_puzzle(matrix_size, window_size, pooling_type)
            
            # Create sample directory
            sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Format question
            question = (
                f"In an N*N grid, there are N^2 numbers, with numbers in the same row separated by commas. "
                f"We define the \"{pooling_type} pooling\" operation: suppose that there is an n*n (n<N) sliding window, "
                f"which slides from left to right or from top to bottom in the matrix, "
                f"{'finding the maximum value' if pooling_type == 'max' else 'calculating the average value'} "
                f"in each sliding window. Then, the {'maximum' if pooling_type == 'max' else 'average'} values "
                f"are arranged according to their original positions to form a new matrix for output.\n"
                f"Now, please perform {pooling_type} pooling on the following matrix by using a {window_size}*{window_size} sliding window:\n"
                f"{format_matrix_as_string(puzzle.matrix)}\n"
                f"For example, <<<0,0,0,0>>> represents a 2D array."
            )
            
            # Save question and solution
            with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                f.write(question)
                
            with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                json.dump({
                    'matrix': puzzle.matrix,
                    'window_size': window_size,
                    'pooling_type': pooling_type,
                    'solution': puzzle.solution,
                    'complexity': puzzle.complexity
                }, f, indent=2)
                
            sample_id += 1
            print(f"Generated sample {sample_id}")

if __name__ == "__main__":
    output_dir = '../dataset_gather/pooling_dataset'
    generate_dataset(output_dir)