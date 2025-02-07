import random
import os
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class LightPuzzle:
    n: int  # grid size
    initial_state: List[List[int]]  # initial light configuration
    buttons: Dict[str, List[List[int]]]  # button configurations
    operations: List[str]  # sequence of button presses
    solution: List[int]  # final state in flattened format
    complexity: int  # 1-5 scale based on grid size and number of buttons

class LightPuzzleGenerator:
    def __init__(self):
        self.max_grid_size = 5
        self.max_buttons = 3
        
    def generate_initial_state(self, n: int) -> List[List[int]]:
        """Generate random initial light configuration"""
        return [[random.randint(0, 1) for _ in range(n)] for _ in range(n)]
    
    def generate_button_controls(self, n: int, num_buttons: int) -> Dict[str, List[List[int]]]:
        """Generate random button configurations"""
        buttons = {}
        available_positions = [(i+1, j+1) for i in range(n) for j in range(n)]
        
        for button_id in range(num_buttons):
            button_name = chr(65 + button_id)  # A, B, C, etc.
            num_controls = random.randint(2, n*n//2)  # Each button controls 2 to n²/2 lights
            controlled_positions = random.sample(available_positions, num_controls)
            buttons[button_name] = [[pos[0], pos[1]] for pos in controlled_positions]
            
        return buttons
    
    def apply_button(self, state: List[List[int]], button_positions: List[List[int]]) -> List[List[int]]:
        """Apply button press effect on the light configuration"""
        new_state = [row[:] for row in state]
        for pos in button_positions:
            row, col = pos[0]-1, pos[1]-1  # Convert to 0-based indexing
            new_state[row][col] = 1 - new_state[row][col]  # Toggle light state
        return new_state
    
    def flatten_state(self, state: List[List[int]]) -> List[int]:
        """Convert 2D grid to 1D list"""
        return [val for row in state for val in row]
    
    def generate_puzzle(self, n: int, num_buttons: int, num_operations: int) -> LightPuzzle:
        """Generate a complete light puzzle"""
        initial_state = self.generate_initial_state(n)
        buttons = self.generate_button_controls(n, num_buttons)
        
        # Generate random sequence of button presses
        operations = [chr(65 + random.randint(0, num_buttons-1)) for _ in range(num_operations)]
        
        # Calculate final state
        current_state = initial_state
        for op in operations:
            current_state = self.apply_button(current_state, buttons[op])
        
        # Calculate complexity
        complexity = min(5, max(1, (n + num_buttons + num_operations) // 3))
        
        return LightPuzzle(
            n=n,
            initial_state=initial_state,
            buttons=buttons,
            operations=operations,
            solution=self.flatten_state(current_state),
            complexity=complexity
        )

def format_grid(grid: List[List[int]]) -> str:
    """Format grid for question text"""
    return '\n'.join(''.join(map(str, row)) for row in grid)

def generate_dataset(output_dir: str):
    """Generate a dataset of light puzzles with varying complexity"""
    os.makedirs(output_dir, exist_ok=True)
    generator = LightPuzzleGenerator()
    
    # Define different configurations for variety
    configurations = [
        # (grid_size, num_buttons, num_operations, num_samples)
        # (2, 1, 1, 1),   # Simple cases
        # (3, 2, 2, 1),   # Medium complexity
        # (4, 2, 3, 1),   # More complex
        # (4, 3, 3, 2),   # High complexity
        # (5, 3, 4, 25),    # Very complex
        (5, 3, 4, 50),    # Very complex
        (2, 1, 1, 14),   # Simple cases
        (3, 2, 2, 14),   # Medium complexity
        (4, 2, 3, 14),   # More complex
        (4, 3, 3, 8),   # High complexity
        (5, 3, 4, 50),    # Very complex
        (5, 3, 4, 50)    # Very complex
    ]
    
    sample_id = 0
    for n, num_buttons, num_operations, num_samples in configurations:
        for _ in range(num_samples):
            puzzle = generator.generate_puzzle(n, num_buttons, num_operations)
            
            # Create sample directory
            sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Format button controls for question
            button_desc = []
            for button, positions in puzzle.buttons.items():
                positions_str = ', '.join([f'["{pos[0]}", "{pos[1]}"]' for pos in positions])
                button_desc.append(f'Button {button} controls the lights at {positions_str}')
            
            # Format operations
            operations_str = '\n'.join([f'Round {i+1}: Button {op}' 
                                      for i, op in enumerate(puzzle.operations)])
            
            # Format question
            question = (
                f"There is an n*n grid with some lights randomly distributed. A lit light is "
                f"represented by 1, and an unlit light is represented by 0. There are some buttons "
                f"which can change the state of the lights in certain positions when they are pressed, "
                f"like turning some lights on or off. Different buttons may control the state of the same light.\n"
                f"Given n={n} and the number of buttons={num_buttons}, the initial network is:\n"
                f"{format_grid(puzzle.initial_state)}\n\n"
                f"{chr(10).join(button_desc)}\n\n"
                f"{operations_str}\n\n"
                f"Please write the result after all rounds of operation based on the lights controlled "
                f"by the button.\nFor example, <<<{','.join(map(str, [0]*n*n))}>>> represents a "
                f"fully unlit light network."
            )
            
            # Save question and solution
            with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                f.write(question)
                
            with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                json.dump({
                    'n': n,
                    'initial_state': puzzle.initial_state,
                    'buttons': puzzle.buttons,
                    'operations': puzzle.operations,
                    'solution': puzzle.solution,
                    'complexity': puzzle.complexity
                }, f, indent=2)
                
            sample_id += 1
            print(f"Generated sample {sample_id}")

def check_answer(answer: str, solution: List[int]) -> bool:
    """Check if the TaskLLM's answer is correct"""
    try:
        # Extract values between <<< and >>>
        start_idx = answer.find('<<<')
        end_idx = answer.find('>>>')
        if start_idx == -1 or end_idx == -1:
            return False
        
        # Extract and clean the answer string
        answer_str = answer[start_idx+3:end_idx].strip()
        # Convert string of comma-separated values to list of integers
        answer_values = [int(x.strip()) for x in answer_str.split(',')]
        
        # Compare with solution
        return answer_values == solution
    except:
        return False

if __name__ == "__main__":
    output_dir = '../dataset_gather/light_puzzles_dataset'
    generate_dataset(output_dir)