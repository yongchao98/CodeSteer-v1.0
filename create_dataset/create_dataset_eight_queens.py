import random
import os
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class QueensPuzzle:
    initial_board: List[str]  # Board as list of strings, each string represents a row
    blocked_positions: List[Tuple[int, int]]  # Positions marked with 'X'
    placed_queens: List[Tuple[int, int]]  # Initially placed queens
    solution: List[Tuple[int, int]]  # Complete solution including initial queens
    complexity: int  # 1-5 scale based on constraints

class QueensPuzzleGenerator:
    def __init__(self):
        self.board_size = 8
        
    def is_valid_position(self, queens: List[Tuple[int, int]], new_pos: Tuple[int, int], blocked: List[Tuple[int, int]]) -> bool:
        """Check if a new queen position is valid given current queens and blocked positions"""
        if new_pos in blocked:
            return False
            
        row, col = new_pos
        for q_row, q_col in queens:
            if (row == q_row or  # Same row
                col == q_col or  # Same column
                abs(row - q_row) == abs(col - q_col)):  # Same diagonal
                return False
        return True
        
    def solve_board(self, initial_queens: List[Tuple[int, int]], blocked: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Find a solution given initial queens and blocked positions"""
        def backtrack(queens: List[Tuple[int, int]], row: int) -> bool:
            if row == self.board_size:
                return True
                
            # Skip rows that already have queens
            while row < self.board_size and any(q[0] == row for q in initial_queens):
                row += 1
            if row == self.board_size:
                return True
                
            for col in range(self.board_size):
                new_pos = (row, col)
                if self.is_valid_position(queens, new_pos, blocked):
                    queens.append(new_pos)
                    if backtrack(queens, row + 1):
                        return True
                    queens.pop()
            return False
            
        solution = initial_queens.copy()
        backtrack(solution, 0)
        return solution
        
    def generate_puzzle(self, min_complexity: int = 1, max_complexity: int = 5) -> QueensPuzzle:
        """Generate a puzzle with specified complexity"""
        while True:
            # Generate random initial queens and blocked positions based on complexity
            num_initial_queens = random.randint(0, 3)
            num_blocked = random.randint(0, 4)
            
            # Place initial queens
            initial_queens = []
            for _ in range(num_initial_queens):
                while True:
                    pos = (random.randint(0, 7), random.randint(0, 7))
                    if self.is_valid_position(initial_queens, pos, []):
                        initial_queens.append(pos)
                        break
                        
            # Place blocked positions
            blocked = []
            for _ in range(num_blocked):
                while True:
                    pos = (random.randint(0, 7), random.randint(0, 7))
                    if pos not in initial_queens and pos not in blocked:
                        blocked.append(pos)
                        break
                        
            # Try to solve
            solution = self.solve_board(initial_queens, blocked)
            if solution and len(solution) == 8:
                # Calculate complexity
                complexity = self._calculate_complexity(initial_queens, blocked)
                if min_complexity <= complexity <= max_complexity:
                    # Create board representation
                    board = self._create_board(initial_queens, blocked)
                    return QueensPuzzle(
                        initial_board=board,
                        blocked_positions=blocked,
                        placed_queens=initial_queens,
                        solution=solution,
                        complexity=complexity
                    )
                    
    def _calculate_complexity(self, queens: List[Tuple[int, int]], blocked: List[Tuple[int, int]]) -> int:
        """Calculate puzzle complexity on a scale of 1-5"""
        # Base complexity from constraints
        complexity = 1
        
        # Additional complexity from initial queens
        complexity += len(queens)
        
        # Additional complexity from blocked positions
        complexity += len(blocked) // 2
        
        return min(max(complexity, 1), 5)
        
    def _create_board(self, queens: List[Tuple[int, int]], blocked: List[Tuple[int, int]]) -> List[str]:
        """Create string representation of the board"""
        board = [['0' for _ in range(self.board_size)] for _ in range(self.board_size)]
        
        # Place queens
        for row, col in queens:
            board[row][col] = '1'
            
        # Place blocked positions
        for row, col in blocked:
            board[row][col] = 'X'
            
        return [''.join(row) for row in board]

def check_solution(llm_response: str, solution: List[Tuple[int, int]], blocked: List[Tuple[int, int]]) -> bool:
    """Check if LLM's response is correct, handling various formats"""
    try:
        # Extract content between <<< and >>>
        start_idx = llm_response.find('<<<')
        end_idx = llm_response.find('>>>')
        if start_idx == -1 or end_idx == -1:
            return False
            
        content = llm_response[start_idx+3:end_idx].strip()
        
        # Convert response to board representation
        board = [['0' for _ in range(8)] for _ in range(8)]
        
        # Handle different possible formats
        if '[' in content and ']' in content:
            # List format
            content = content.replace('[', '').replace(']', '')
            positions = content.split(',')
            for pos in positions:
                try:
                    row, col = map(int, pos.strip().split())
                    if (row, col) in blocked:
                        return False
                    board[row][col] = '1'
                except:
                    return False
        else:
            # Direct board format
            rows = content.split()
            if len(rows) != 8:
                return False
            for i, row in enumerate(rows):
                if len(row) != 8:
                    return False
                for j, char in enumerate(row):
                    if char == '1':
                        board[i][j] = '1'
                    elif char not in '0X':
                        return False
        
        # Convert board to queen positions
        response_queens = []
        for i in range(8):
            for j in range(8):
                if board[i][j] == '1':
                    response_queens.append((i, j))
        
        # Check if number of queens is correct
        if len(response_queens) != 8:
            return False
            
        # Check if queens are in valid positions
        checker = QueensPuzzleGenerator()
        for pos in response_queens:
            if not checker.is_valid_position(
                [p for p in response_queens if p != pos],
                pos,
                blocked
            ):
                return False
                
        return True
    except:
        return False

def generate_dataset(output_dir: str):
    """Generate a dataset of Eight Queens puzzles with varying complexity"""
    os.makedirs(output_dir, exist_ok=True)
    generator = QueensPuzzleGenerator()
    
    # Define different configurations for variety
    configurations = [
        # (min_complexity, max_complexity, num_samples)
        (1, 2, 60),  # Easy puzzles
        (2, 3, 90),  # Medium puzzles
        (3, 4, 90),  # Hard puzzles
        (4, 5, 60)   # Very hard puzzles
    ]
    
    sample_id = 0
    for min_complexity, max_complexity, num_samples in configurations:
        for _ in range(num_samples):
            puzzle = generator.generate_puzzle(min_complexity, max_complexity)
            
            # Create sample directory
            sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Format question
            board_str = '\n'.join(puzzle.initial_board)
            question = (
                f"Given an 8*8 grid, cells marked with 1 have a queen placed, and cells marked with 0 "
                f"are empty. A total of eight queens need to be placed such that no two queens can attack "
                f"each other, meaning no two queens can be in the same row, column, or diagonal. "
                f"Additionally, positions marked with X are not allowed for placing queens. Please help me "
                f"complete the placement of the remaining queens on the partially filled board.\n\n"
                f"{board_str}\n\n"
                f"Return your answer with the format <<<list of queen positions>>>, where each position "
                f"is given as row column (0-based indices). For example, <<<0 3, 1 0, 2 4>>> means queens "
                f"are placed at positions (0,3), (1,0), and (2,4)."
            )
            
            # Save question and solution
            with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                f.write(question)
                
            with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                json.dump({
                    'initial_board': puzzle.initial_board,
                    'blocked_positions': puzzle.blocked_positions,
                    'placed_queens': puzzle.placed_queens,
                    'solution': puzzle.solution,
                    'complexity': puzzle.complexity
                }, f, indent=2)
                
            sample_id += 1
            print(f"Generated sample {sample_id} with complexity {puzzle.complexity}")

if __name__ == "__main__":
    #output_dir = '/Users/yongchaochen/Robot_NLP/Codesteer/ICLR_Code/dataset_gather/eight_queens_dataset_2'
    output_dir = 'dataset_gather/eight_queens_dataset'
    generate_dataset(output_dir)
