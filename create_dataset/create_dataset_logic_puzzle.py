import random
import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
import re
import time
from functools import lru_cache

@dataclass
class GridPuzzle:
    grid: List[List[int]]
    num_selections: int
    constraints: dict
    start_position: Tuple[int, int]
    start_value: int
    solution: List[Tuple[int, int]]
    complexity: int  # 1-5 scale
    coordinate_system: str  # 'top-left' or 'bottom-left'

class GridPuzzleGenerator:
    def __init__(self):
        self.complexity_configs = {
            1: {'size': (2, 2), 'num_select': 3, 'value_range': (1, 3), 'constraint': 'sum_le_4'},
            2: {'size': (3, 3), 'num_select': 4, 'value_range': (1, 4), 'constraint': 'sum_le_4'},
            3: {'size': (4, 4), 'num_select': 6, 'value_range': (-5, 8), 'constraint': 'product_gt_0'},
            4: {'size': (5, 5), 'num_select': 9, 'value_range': (-8, 8), 'constraint': 'product_gt_0'},  # Reduced from 8
            5: {'size': (5, 5), 'num_select': 10, 'value_range': (-10, 10), 'constraint': 'product_gt_0'},  # Reduced size from 6x6
            6: {'size': (6, 6), 'num_select': 12, 'value_range': (-12, 12), 'constraint': 'product_gt_0'},
            7: {'size': (7, 7), 'num_select': 16, 'value_range': (-14, 14), 'constraint': 'product_gt_0'},
            8: {'size': (8, 8), 'num_select': 18, 'value_range': (-16, 16), 'constraint': 'product_gt_0'},
            9: {'size': (9, 9), 'num_select': 20, 'value_range': (-18, 18), 'constraint': 'product_gt_0'},
            10: {'size': (10, 10), 'num_select': 25, 'value_range': (-20, 20), 'constraint': 'product_gt_0'}
        }
        self.max_attempts = 50  # Maximum attempts to generate a valid puzzle
        self.timeout = 5  # Timeout in seconds for solution finding

    def generate_grid(self, size: Tuple[int, int], value_range: Tuple[int, int]) -> List[List[int]]:
        # Generate grid with more balanced distribution of positive/negative numbers
        grid = []
        for _ in range(size[0]):
            row = []
            for _ in range(size[1]):
                if value_range[0] < 0:  # If we have negative numbers
                    # 70% chance of positive number to make solutions more likely
                    if random.random() < 0.7:
                        value = random.randint(1, value_range[1])
                    else:
                        value = random.randint(value_range[0], -1)
                else:
                    value = random.randint(*value_range)
                row.append(value)
            grid.append(row)
        return grid

    @lru_cache(maxsize=1024)
    def check_sum_constraint(self, positions_tuple: Tuple[Tuple[int, int], ...], max_sum: int) -> bool:
        positions = list(positions_tuple)
        rows = {}
        cols = {}
        for r, c in positions:
            val = self.grid[r][c]  # Access grid as instance variable
            rows[r] = rows.get(r, 0) + val
            cols[c] = cols.get(c, 0) + val
        return all(sum <= max_sum for sum in rows.values()) and all(sum <= max_sum for sum in cols.values())

    @lru_cache(maxsize=1024)
    def check_product_constraint(self, positions_tuple: Tuple[Tuple[int, int], ...]) -> bool:
        positions = list(positions_tuple)
        rows = {}
        cols = {}
        for r, c in positions:
            val = self.grid[r][c]  # Access grid as instance variable
            rows[r] = rows.get(r, 1) * val
            cols[c] = cols.get(c, 1) * val
        return all(prod > 0 for prod in rows.values()) and all(prod > 0 for prod in cols.values())

    def find_valid_solution(self, puzzle: GridPuzzle) -> Optional[List[Tuple[int, int]]]:
        self.grid = puzzle.grid  # Store grid as instance variable for caching
        grid =  self.grid
        start_time = time.time()
        
        def get_candidate_positions(grid: List[List[int]], positions: Set[Tuple[int, int]], values: Set[int]) -> List[Tuple[int, int]]:
            candidates = []
            rows_used = {r for r, _ in positions}
            cols_used = {c for _, c in positions}
            
            # Prioritize positions that help satisfy constraints
            for r in range(len(grid)):
                for c in range(len(grid[0])):
                    if (r, c) not in positions and grid[r][c] not in values:
                        # Prioritize unused rows and columns
                        priority = (r not in rows_used) + (c not in cols_used)
                        # For product constraint, prioritize positive numbers
                        if puzzle.constraints['type'] == 'product_gt_0':
                            priority += (grid[r][c] > 0)
                        candidates.append(((r, c), priority))
            
            # Sort by priority (higher first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [pos for pos, _ in candidates]

        def backtrack(positions: Set[Tuple[int, int]], values: Set[int]) -> Optional[List[Tuple[int, int]]]:
            if time.time() - start_time > self.timeout:
                return None
                
            if len(positions) == puzzle.num_selections:
                positions_tuple = tuple(sorted(positions))
                if puzzle.constraints['type'] == 'sum_le_4':
                    if self.check_sum_constraint(positions_tuple, 4):
                        return list(positions)
                elif puzzle.constraints['type'] == 'product_gt_0':
                    if self.check_product_constraint(positions_tuple):
                        return list(positions)
                return None

            candidates = get_candidate_positions(grid, positions, values)
            for r, c in candidates:
                val = grid[r][c]
                
                # Quick check for product constraint
                if puzzle.constraints['type'] == 'product_gt_0':
                    row_vals = [grid[r][c2] for _, c2 in positions if r == _] + [val]
                    if 0 in row_vals or (len(row_vals) > 1 and all(v < 0 for v in row_vals)):
                        continue

                positions.add((r, c))
                values.add(val)
                result = backtrack(positions, values)
                if result:
                    return result
                positions.remove((r, c))
                values.remove(val)
            return None

        # Start with the given position
        initial_positions = {puzzle.start_position}
        initial_values = {puzzle.grid[puzzle.start_position[0]][puzzle.start_position[1]]}
        return backtrack(initial_positions, initial_values)

    def generate_puzzle(self, complexity: int) -> Optional[GridPuzzle]:
        config = self.complexity_configs[complexity]
        
        for _ in range(self.max_attempts):
            grid = self.generate_grid(config['size'], config['value_range'])
            
            # Choose start position (for this example, always use the top-right corner)
            start_r, start_c = (0, config['size'][1]-1)
            start_value = grid[start_r][start_c]
            
            puzzle = GridPuzzle(
                grid=grid,
                num_selections=config['num_select'],
                constraints={'type': config['constraint']},
                start_position=(start_r, start_c),
                start_value=start_value,
                solution=[],
                complexity=complexity,
                coordinate_system='top-left'
            )
            
            solution = self.find_valid_solution(puzzle)
            if solution:
                puzzle.solution = solution
                return puzzle
                
        return None

    def format_question(self, puzzle: GridPuzzle) -> str:
        # Same as before
        size = len(puzzle.grid)
        grid_str = '\n'.join([', '.join(map(str, row)) for row in puzzle.grid])
        
        question = (
            f"Given a {size}*{size} grid where each cell contains a number "
            f"(between {min(min(row) for row in puzzle.grid)} to {max(max(row) for row in puzzle.grid)}), "
            f"you must select {puzzle.num_selections} different numbers from these cells. "
        )
        
        if puzzle.constraints['type'] == 'sum_le_4':
            question += "The sum of the selected numbers in each row and each column should be less than or equal to 4. "
        elif puzzle.constraints['type'] == 'product_gt_0':
            question += "The product of each row and column of the selected number must be greater than 0. "
            
        question += (
            f"The numbers in the same row are separated by ',', and the coordinate system uses the "
            f"top-left cell as the coordinate (1,1), with the horizontal coordinate under the top-left cell "
            f"increasing and the vertical coordinate to the right of the top-left cell increasing.\n"
            f"Using the number {puzzle.start_value} in the top-rightmost square (0,{size-1}) as the first selected number\n"
            f"{grid_str}\n\n"
            f"Return your answer with the format <<<list of selected number positions>>>, where each position "
            f"is given as row column (0-based indices). For example, <<<0 3, 1 0, 2 4>>> means numbers selected "
            f"are at positions (0,3), (1,0), and (2,4)."
        )
        
        return question

def validate_llm_answer(answer: str, solution: List[Tuple[int, int]]) -> bool:
    # Same as before
    try:
        match = re.search(r'<<<(.+?)>>>', answer)
        if not match:
            return False
            
        positions_str = match.group(1)
        positions = []
        for pos in positions_str.split(','):
            row, col = map(int, pos.strip().split())
            positions.append((row, col))
            
        solution_set = set(solution)
        return set(positions) == solution_set
    except:
        return False

def generate_dataset(output_dir: str, samples_per_complexity: int = 1):
    os.makedirs(output_dir, exist_ok=True)
    generator = GridPuzzleGenerator()
    
    sample_id = 0
    for complexity in range(1, 11):
        if complexity == 7 or complexity == 8 or complexity == 9 or complexity == 10:
            samples_per_complexity = 13 * 2
        else:
            samples_per_complexity = 8 * 2
        successful_samples = 0
        attempts = 0
        max_attempts = samples_per_complexity * 3  # Allow more attempts for harder puzzles
        
        while successful_samples < samples_per_complexity and attempts < max_attempts:
            puzzle = generator.generate_puzzle(complexity)
            attempts += 1
            
            if puzzle:
                # Create sample directory
                sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
                os.makedirs(sample_dir, exist_ok=True)
                
                # Generate question
                question = generator.format_question(puzzle)
                
                # Save question and solution
                with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                    f.write(question)
                    
                with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                    json.dump({
                        'grid': puzzle.grid,
                        'solution': puzzle.solution,
                        'complexity': puzzle.complexity,
                        'constraints': puzzle.constraints
                    }, f, indent=2)
                    
                sample_id += 1
                successful_samples += 1
                print(f"Generated sample {sample_id} with complexity {complexity}")
            else:
                print(f"Failed to generate puzzle with complexity {complexity}, attempt {attempts}")

if __name__ == "__main__":
    output_dir = '../dataset_gather/logic_puzzle_dataset'
    generate_dataset(output_dir)