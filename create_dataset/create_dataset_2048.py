import random
import os
import json
from typing import List, Tuple
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class Game2048Puzzle:
    grid: List[List[int]]
    moves: str
    start_pos: Tuple[int, int]
    solution: str  # Changed from List[List[int]] to str
    complexity: int  

class Game2048Generator:
    def __init__(self):
        self.valid_powers = [2 ** i for i in range(1, 11)]  # 2 to 1024
        
    def apply_move(self, grid: List[List[int]], pos: Tuple[int, int], move: str) -> Tuple[List[List[int]], Tuple[int, int]]:
        """Apply a single move to the grid and return new grid and position"""
        n = len(grid)
        new_grid = deepcopy(grid)
        x, y = pos
        
        move_map = {
            'u': (-1, 0),
            'd': (1, 0),
            'l': (0, -1),
            'r': (0, 1)
        }
        
        if move not in move_map:
            return new_grid, pos
            
        dx, dy = move_map[move]
        new_x, new_y = x + dx, y + dy
        
        # Check if move is valid
        if 0 <= new_x < n and 0 <= new_y < n:
            # If empty space, just move
            if new_grid[new_x][new_y] == 0:
                new_grid[new_x][new_y] = new_grid[x][y]
                new_grid[x][y] = 0
                return new_grid, (new_x, new_y)
            # If same number, combine
            elif new_grid[new_x][new_y] == new_grid[x][y]:
                new_grid[new_x][new_y] *= 2
                new_grid[x][y] = 0
                return new_grid, (new_x, new_y)
                
        return new_grid, pos

    def apply_moves(self, grid: List[List[int]], start_pos: Tuple[int, int], moves: str) -> Tuple[List[List[int]], Tuple[int, int]]:
        """Apply a sequence of moves and return final grid and position"""
        current_grid = deepcopy(grid)
        current_pos = start_pos
        
        for move in moves:
            current_grid, current_pos = self.apply_move(current_grid, current_pos, move)
            
        return current_grid, current_pos

    def grid_to_string(self, grid: List[List[int]]) -> str:
        """Convert grid to string representation"""
        return "\n".join(", ".join(str(x) for x in row) for row in grid)

    def generate_puzzle(self, grid_size: int, num_moves: int, min_complexity: int = 1, max_complexity: int = 5) -> Game2048Puzzle:
        """Generate a puzzle with specified grid size and number of moves"""
        # Generate random grid with powers of 2
        grid = [[0] * grid_size for _ in range(grid_size)]
        num_numbers = random.randint(grid_size, grid_size * 2)
        
        positions = random.sample([(i, j) for i in range(grid_size) for j in range(grid_size)], num_numbers)
        for pos in positions:
            grid[pos[0]][pos[1]] = random.choice(self.valid_powers[:5])
            
        # Generate random start position
        start_x = random.randint(0, grid_size - 1)
        start_y = random.randint(0, grid_size - 1)
        
        # Generate random moves
        moves = ''.join(random.choice('udlr') for _ in range(num_moves))
        
        # Calculate solution and convert to string
        solution_grid, _ = self.apply_moves(grid, (start_x, start_y), moves)
        solution = self.grid_to_string(solution_grid)
        
        # Calculate complexity
        complexity = self._calculate_complexity(grid_size, num_moves, len(positions))
        
        return Game2048Puzzle(
            grid=grid,
            moves=moves,
            start_pos=(start_x, start_y),
            solution=solution,
            complexity=complexity
        )

    def _calculate_complexity(self, grid_size: int, num_moves: int, num_numbers: int) -> int:
        """Calculate puzzle complexity on a scale of 1-5"""
        complexity = 1
        
        if grid_size > 3:
            complexity += 1
        if grid_size > 4:
            complexity += 1
            
        if num_moves > 3:
            complexity += 1
        if num_moves > 5:
            complexity += 1
            
        if num_numbers > grid_size * 1.5:
            complexity += 1
            
        return min(max(complexity, 1), 5)

def validate_llm_answer(answer: str, expected: str) -> bool:
    """Validate LLM's answer against expected solution"""
    try:
        # Extract content between <<< and >>>
        start_idx = answer.find('<<<')
        end_idx = answer.find('>>>')
        if start_idx == -1 or end_idx == -1:
            return False
            
        content = answer[start_idx + 3:end_idx].strip()
        
        # Normalize both strings by removing extra whitespace and standardizing separators
        def normalize_grid_string(s: str) -> str:
            lines = s.strip().split('\n')
            normalized_lines = []
            for line in lines:
                numbers = [n.strip() for n in line.split(',')]
                normalized_lines.append(', '.join(numbers))
            return '\n'.join(normalized_lines)
        
        normalized_content = normalize_grid_string(content)
        normalized_expected = normalize_grid_string(expected)
        
        return normalized_content == normalized_expected
    except:
        return False

def generate_dataset(output_dir: str):
    """Generate a dataset of 2048 puzzles with varying complexity"""
    os.makedirs(output_dir, exist_ok=True)
    generator = Game2048Generator()
    
    configurations = [
        # (grid_size, num_moves, min_complexity, max_complexity, num_samples)
        (2, 2, 1, 2, 20),
        (3, 3, 2, 3, 20),
        (4, 4, 3, 4, 20),
        (5, 5, 4, 5, 20),
        (5, 6, 4, 5, 20),
        (2, 2, 1, 2, 20),
        (3, 3, 2, 3, 20),
        (4, 4, 3, 4, 20),
        (5, 5, 4, 5, 20),
        (5, 6, 4, 5, 20)
    ]
    
    sample_id = 0
    for grid_size, num_moves, min_complexity, max_complexity, num_samples in configurations:
        for _ in range(num_samples):
            puzzle = generator.generate_puzzle(grid_size, num_moves, min_complexity, max_complexity)
            
            sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Format grid as string
            grid_str = generator.grid_to_string(puzzle.grid)
            
            # Format question
            question = (
                f"In an {grid_size}x{grid_size} grid, there are several squares containing powers of 2. "
                f"We define the '2048' game: The numbers in the grid can move in any direction. "
                f"When they encounter a number that is the same power of 2, they can combine to form "
                f"a larger number that is the next power of 2. For example: 2 meets 2 -> 4; 4 meets 4 -> 8; "
                f"16 meets 16 -> 32. When they encounter a different number, they remain unchanged.\n"
                f"u: represents moving up one step\n"
                f"d: represents moving down one step\n"
                f"l: represents moving left one step\n"
                f"r: represents moving right one step\n\n"
                f"Now, in the following matrix, move the number in position ({puzzle.start_pos[0] + 1}, {puzzle.start_pos[1] + 1}) "
                f"using the sequence '{puzzle.moves}':\n"
                f"{grid_str}\n\n"
                f"What is the matrix after moving?\n"
                f"Return your answer with format <<<matrix>>>."
            )
            
            # Save question and solution
            with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                f.write(question)
                
            with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                json.dump({
                    'grid': puzzle.grid,
                    'moves': puzzle.moves,
                    'start_pos': puzzle.start_pos,
                    'solution': puzzle.solution,
                    'complexity': puzzle.complexity
                }, f, indent=2)
                
            sample_id += 1
            print(f"Generated sample {sample_id}")

if __name__ == "__main__":
    output_dir = '../dataset_gather/2048_dataset'
    generate_dataset(output_dir)
