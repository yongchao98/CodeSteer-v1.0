import random
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Union

@dataclass
class SynthesisProblem:
    initial_state: Dict[str, int]
    methods: List[Dict[str, Dict[str, int]]]
    solution: Dict[str, Union[List[str], List[List[str]]]]
    complexity: int  # 1-5 scale

class SynthesisPuzzleGenerator:
    def __init__(self):
        self.crops = ['A', 'B', 'C']
        self.products = ['X', 'Y']
        
    def apply_method(self, state: Dict[str, int], method: Dict[str, Dict[str, int]]) -> Union[Dict[str, int], None]:
        """Apply a synthesis method if possible"""
        new_state = state.copy()
        
        # Check if method can be applied
        for resource, amount in method['input'].items():
            if state.get(resource, 0) < amount:
                return None
                
        # Apply method
        for resource, amount in method['input'].items():
            new_state[resource] = new_state.get(resource, 0) - amount
        for resource, amount in method['output'].items():
            new_state[resource] = new_state.get(resource, 0) + amount
            
        return new_state

    def solve_puzzle(self, initial_state: Dict[str, int], methods: List[Dict[str, Dict[str, int]]]) -> Dict[str, Union[List[str], List[List[str]]]]:
        """Solve the synthesis puzzle"""
        current_state = initial_state.copy()
        process = []
        method_index = 0
        
        while True:
            method = methods[method_index]
            new_state = self.apply_method(current_state, method)
            
            if new_state:
                current_state = new_state
                # Record state after method application
                state_list = [str(current_state.get('A', 0)), 
                            str(current_state.get('B', 0)),
                            str(current_state.get('C', 0)),
                            str(current_state.get('X', 0)),
                            str(current_state.get('Y', 0))]
                process.append(state_list)
            
            method_index = (method_index + 1) % len(methods)
            
            # Check if no more methods can be applied
            can_continue = False
            for m in methods:
                if self.apply_method(current_state, m):
                    can_continue = True
                    break
            
            if not can_continue:
                break
        
        return {
            "answer": [str(current_state.get('A', 0)), 
                      str(current_state.get('B', 0)),
                      str(current_state.get('C', 0)),
                      str(current_state.get('X', 0)),
                      str(current_state.get('Y', 0))],
            "process": process
        }

    def generate_puzzle(self, complexity: int) -> SynthesisProblem:
        """Generate a synthesis puzzle based on complexity"""
        if complexity <= 2:
            # Simple case with just X product
            initial_state = {
                'A': random.randint(3, 6),
                'B': random.randint(2, 4)
            }
            methods = [
                {'input': {'A': 1, 'B': 1}, 'output': {'X': 1}},
                {'input': {'A': 3}, 'output': {'X': 2}}
            ]
        elif complexity <= 4:
            # Medium case with three crops and X
            initial_state = {
                'A': random.randint(4, 8),
                'B': random.randint(3, 6),
                'C': random.randint(2, 5)
            }
            methods = [
                {'input': {'A': 1, 'B': 1, 'C': 1}, 'output': {'X': 1}},
                {'input': {'A': 1, 'B': 1}, 'output': {'C': 2}},
                {'input': {'C': 2}, 'output': {'X': 1}}
            ]
        else:
            # Complex case with both X and Y products
            initial_state = {
                'A': random.randint(5, 8),
                'B': random.randint(2, 4),
                'C': random.randint(4, 7)
            }
            methods = [
                {'input': {'A': 2, 'B': 1, 'C': 1}, 'output': {'X': 3}},
                {'input': {'A': 1, 'B': 1}, 'output': {'C': 2}},
                {'input': {'C': 2, 'X': 1}, 'output': {'Y': 1}},
                {'input': {'A': 1, 'X': 2}, 'output': {'Y': 1}}
            ]
        
        solution = self.solve_puzzle(initial_state, methods)
        return SynthesisProblem(
            initial_state=initial_state,
            methods=methods,
            solution=solution,
            complexity=complexity
        )

def format_method(method: Dict[str, Dict[str, int]], method_num: int) -> str:
    """Format method description for question prompt"""
    input_str = ' and '.join([f"{amount} unit{'s' if amount > 1 else ''} of {'crop' if resource in ['A','B','C'] else 'product'} {resource}" 
                             for resource, amount in method['input'].items()])
    output_str = ' and '.join([f"{amount} unit{'s' if amount > 1 else ''} of {'product'} {resource}" 
                              for resource, amount in method['output'].items()])
    return f"Method {method_num}: {input_str} can be exchanged for {output_str}, and this can be done only once per cycle."

def generate_dataset(output_dir: str):
    """Generate dataset with varying complexity"""
    os.makedirs(output_dir, exist_ok=True)
    generator = SynthesisPuzzleGenerator()
    
    configurations = [
        (1, 10),  # (complexity, num_samples)
        (2, 20),
        (3, 30),
        (4, 40),
        (5, 100)  # More samples for complex cases
    ]
    
    sample_id = 0
    for complexity, num_samples in configurations:
        for _ in range(num_samples):
            puzzle = generator.generate_puzzle(complexity)
            
            sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Format question
            initial_state_str = ', '.join([f"{amount} units of {'crop' if crop in ['A','B','C'] else 'product'} {crop}" 
                                         for crop, amount in puzzle.initial_state.items()])
            methods_str = '\n\n'.join([format_method(method, i+1) 
                                     for i, method in enumerate(puzzle.methods)])
            
            products_str = 'X and Y' if complexity > 4 else 'X'
            crops_str = 'A, B, and C' if complexity > 2 else 'A and B'
            
            question = (
                f"Xiao Li is a farm owner who grows {crops_str}. He can exchange these crops "
                f"for agricultural product{'s' if complexity > 4 else ''} {products_str}. "
                f"He has discovered the following methods to increase the farm's yield:\n\n"
                f"{methods_str}\n\n"
                f"Xiao Li continuously cycles through these methods in the given order, executing "
                f"one method at a time. After Method {len(puzzle.methods)}, start again from Method 1. "
                f"Continue this cycle until no method can be executed.\n\n"
                f"Initially there are {initial_state_str}.\n\n"
                f"First output the entire reasoning process step by step, showing each synthesis step's result. "
                f"Then wrap your final answer in <<<>>> format as a list showing "
                f"[remaining_A, remaining_B{', remaining_C' if complexity > 2 else ''}, obtained_X{', obtained_Y' if complexity > 4 else ''}]. "
                f"For example, if there are 1 unit of crop A, 0 units of crop B{', 2 units of crop C' if complexity > 2 else ''}, "
                f"and 3 units of product X{', 1 unit of product Y' if complexity > 4 else ''} remaining, output: "
                f"<<<[1, 0{', 2' if complexity > 2 else ''}, 3{', 1' if complexity > 4 else ''}]>>>"
            )
            
            # Save files
            with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                f.write(question)
                
            with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                json.dump({
                    'initial_state': puzzle.initial_state,
                    'solution': puzzle.solution,
                    'complexity': puzzle.complexity
                }, f, indent=2)
                
            sample_id += 1

def check_llm_answer(llm_response: str, solution: Dict[str, Union[List[str], List[List[str]]]], complexity: int) -> bool:
    """Check if LLM's answer matches the solution with the new <<<>>> format requirement"""
    try:
        # Find content between <<< and >>>
        pattern = r'<<<\s*\[(.*?)\]\s*>>>'
        match = re.search(pattern, llm_response, re.DOTALL)
        if not match:
            return False
        
        # Extract and parse the answer
        answer_str = match.group(1)
        # Split by comma and clean up each element
        answer_parts = [part.strip().strip('"\'') for part in answer_str.split(',')]
        
        # Check number of elements based on complexity
        expected_length = 5 if complexity > 4 else (4 if complexity > 2 else 3)
        if len(answer_parts) != expected_length:
            return False
            
        # Compare with solution
        return answer_parts == solution['answer'][:expected_length]
        
    except Exception:
        return False

def format_expected_output(complexity: int) -> str:
    """Generate example of expected output format"""
    if complexity <= 2:
        return '<<<[1, 0, 2]>>>'
    elif complexity <= 4:
        return '<<<[1, 0, 2, 3]>>>'
    else:
        return '<<<[1, 0, 2, 3, 1]>>>'

if __name__ == "__main__":
    output_dir = '../dataset_gather/synthesis_decomposition_dataset'
    generate_dataset(output_dir)