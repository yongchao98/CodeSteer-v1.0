import random
import os
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class ReversiPuzzle:
    board_size: int
    initial_state: List[List[str]]
    moves: List[Tuple[int, int]]
    player_color: str  # '0' for black, '1' for white
    solution: str  # Changed from List[str] to str
    complexity: int  # 1-5 scale based on number of moves and flips

class ReversiGameState:
    def __init__(self, size: int):
        self.size = size
        self.board = [['*' for _ in range(size)] for _ in range(size)]
        # Initialize center pieces
        mid = size // 2
        self.board[mid-1][mid-1] = '1'
        self.board[mid-1][mid] = '0'
        self.board[mid][mid-1] = '0'
        self.board[mid][mid] = '1'

    def copy(self):
        new_state = ReversiGameState(self.size)
        new_state.board = deepcopy(self.board)
        return new_state

    def is_valid_move(self, row: int, col: int, player: str) -> bool:
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        if self.board[row][col] != '*':
            return False
        
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        opponent = '1' if player == '0' else '0'
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if not (0 <= r < self.size and 0 <= c < self.size):
                continue
            if self.board[r][c] != opponent:
                continue
                
            r, c = r + dr, c + dc
            while 0 <= r < self.size and 0 <= c < self.size:
                if self.board[r][c] == '*':
                    break
                if self.board[r][c] == player:
                    return True
                r, c = r + dr, c + dc
        return False

    def make_move(self, row: int, col: int, player: str) -> int:
        if not self.is_valid_move(row, col, player):
            return 0
            
        flips = 0
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        opponent = '1' if player == '0' else '0'
        self.board[row][col] = player
        
        for dr, dc in directions:
            to_flip = []
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size:
                if self.board[r][c] == '*':
                    break
                if self.board[r][c] == opponent:
                    to_flip.append((r, c))
                if self.board[r][c] == player:
                    for flip_r, flip_c in to_flip:
                        self.board[flip_r][flip_c] = player
                        flips += 1
                    break
                r, c = r + dr, c + dc
                
        return flips

    def to_solution_string(self) -> str:
        # Convert board to a single string with commas between each cell
        cells = []
        for row in self.board:
            cells.extend(row)
        return ','.join(cells)

class ReversiPuzzleGenerator:
    def __init__(self):
        self.min_moves = 1
        self.max_moves = 4
        
    def generate_puzzle(self, board_size: int, num_moves: int, min_complexity: int = 1, max_complexity: int = 5) -> ReversiPuzzle:
        game = ReversiGameState(board_size)
        moves = []
        current_player = '0'  # Black starts
        
        # Generate random valid moves
        for _ in range(num_moves):
            valid_moves = []
            for r in range(board_size):
                for c in range(board_size):
                    if game.is_valid_move(r, c, current_player):
                        valid_moves.append((r, c))
                        
            if not valid_moves:
                break
                
            move = random.choice(valid_moves)
            moves.append(move)
            flips = game.make_move(move[0], move[1], current_player)
            current_player = '1' if current_player == '0' else '0'
        
        # Calculate complexity based on number of moves and flips
        complexity = min(max(len(moves), 1), 5)
        
        return ReversiPuzzle(
            board_size=board_size,
            initial_state=game.board,
            moves=moves,
            player_color='0',  # Always generate puzzles for black player
            solution=game.to_solution_string(),  # Now returns a single string
            complexity=complexity
        )

def validate_llm_response(response: str, expected_solution: str) -> bool:
    """Validate LLM's response against the expected solution"""
    try:
        # Extract content between <<< and >>>
        start_idx = response.find('<<<')
        end_idx = response.find('>>>')
        if start_idx == -1 or end_idx == -1:
            return False
            
        content = response[start_idx+3:end_idx].strip()
        
        # Remove any extra whitespace/quotes and compare directly
        actual_solution = content.strip().strip('"').strip("'")
        
        # Split both solutions into lists and compare
        actual_cells = [cell.strip() for cell in actual_solution.split(',')]
        expected_cells = expected_solution.split(',')
        
        return actual_cells == expected_cells
    except Exception:
        return False

def generate_dataset(output_dir: str):
    """Generate a dataset of Reversi puzzles with varying complexity"""
    os.makedirs(output_dir, exist_ok=True)
    generator = ReversiPuzzleGenerator()
    
    # Define different configurations for variety
    configurations = [
        # (board_size, num_moves, min_complexity, max_complexity, num_samples)
        (4, 1, 1, 2, 20),
        (4, 2, 2, 3, 20),
        (4, 3, 3, 4, 20),
        (6, 2, 2, 3, 20),
        (6, 3, 3, 4, 20),
        (4, 1, 1, 2, 20),
        (4, 2, 2, 3, 20),
        (4, 3, 3, 4, 20),
        (6, 2, 2, 3, 20),
        (6, 3, 3, 4, 20)
    ]
    
    sample_id = 0
    for board_size, num_moves, min_complexity, max_complexity, num_samples in configurations:
        for _ in range(num_samples):
            puzzle = generator.generate_puzzle(board_size, num_moves, min_complexity, max_complexity)
            
            # Create sample directory
            sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Format question
            moves_description = "\n".join([f"Round {i+1}: {'Your' if i % 2 == 0 else 'The opponent'}'s landing position is ({move[0]+1},{move[1]+1})"
                                         for i, move in enumerate(puzzle.moves)])
            
            initial_board = "\n".join(["".join(row) for row in puzzle.initial_state])
            
            question = (
                f"There is a game similar to Reversi. On an n*n grid (where n is even), "
                f"four pieces are placed in the center, two black and two white, arranged diagonally. "
                f"Black pieces are represented by 0, white pieces by 1, and empty spaces by *.\n"
                f"Rules:\n"
                f"1. Two players take turns placing pieces, one playing black and the other playing white.\n"
                f"2. Each player places one piece on an empty space during their turn.\n"
                f"3. After placing a piece, if there are any of the opponent's pieces between two of "
                f"the player's pieces (in the same row, column, or diagonal), those pieces will be "
                f"flipped to the player's color. That is, 0 becomes 1, or 1 becomes 0.\n"
                f"All coordinates start from the first row and first column, and the coordinate (a, b) "
                f"represents the cell in the a-th row and b-th column.\n\n"
                f"Given: n={board_size}, you play black chess.\n\n"
                f"The previous round of the game was: \n{initial_board}\n\n"
                f"{moves_description}\n\n"
                f"Please write down the game situation after this round based on the situation of "
                f"the previous round.\nFor example, <<<*,*,*,*,*,*,*,*,*,*,*,*,*,*,*,*>>> represents "
                f"the empty grid."
            )
            
            # Save question and solution
            with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                f.write(question)
                
            with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                json.dump({
                    'board_size': board_size,
                    'moves': [(m[0]+1, m[1]+1) for m in puzzle.moves],  # Convert to 1-based indexing
                    'solution': puzzle.solution,
                    'complexity': puzzle.complexity
                }, f, indent=2)
                
            sample_id += 1
            print(f"Generated sample {sample_id}")

if __name__ == "__main__":
    output_dir = '../dataset_gather/reversi_dataset'
    generate_dataset(output_dir)