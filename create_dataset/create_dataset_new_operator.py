import random
import os
import json
from typing import List, Dict, Union, Tuple
from dataclasses import dataclass
import re

@dataclass
class OperatorPuzzle:
    question: str
    solution: Union[float, int]
    solution_steps: List[str]
    complexity: int  # 1-5 scale
    operator_definitions: Dict[str, str]

class NewOperatorGenerator:
    def __init__(self):
        self.operators = ['#', '@', '$', '&', '%', '~']
        self.complexity_weights = {
            1: 0.2,  # Simple single operation
            2: 0.3,  # Single operation with more complex numbers
            3: 0.2,  # Multiple operations
            4: 0.2,  # Operations with variables and substitution
            5: 0.1   # Complex operations with multiple rules
        }
    
    def generate_simple_operator(self, op1=None,op2=None) -> Tuple[str, str, Dict[str, float]]:
        """Generate a simple operator definition with coefficients"""
        while True:
            operator = random.choice(self.operators)
            if operator != op1 and operator != op2: 
                break
        coef1 = random.randint(1, 5)
        coef2 = random.randint(1, 5)
        
        templates = [
            (f"{coef1}*a{'+' if coef2 >= 0 else ''}{coef2}*b", {'coef1': coef1, 'coef2': coef2}),
            (f"{coef1}*a*b", {'coef1': coef1}),
            (f"(a + b)/b", {})
        ]
        
        formula, coefficients = random.choice(templates)
        definition = f"Let a and b be numbers. Define a{operator}b to be {formula}."
        
        return operator, definition, coefficients

    def evaluate_single_operation(self, a: float, b: float, formula: str) -> float:
        """Evaluate a single operation with two numbers"""
        try:
            result = eval(formula.replace('a', str(a)).replace('b', str(b)))
            return float(result)
        except Exception as e:
            raise ValueError(f"Error evaluating {formula} with a={a}, b={b}: {str(e)}")

    def parse_and_evaluate(self, expr: str, operator_defs: Dict[str, Dict]) -> float:
        """Parse and evaluate an expression with parentheses and custom operators"""
        # First handle parentheses
        while '(' in expr:
            # Find innermost parentheses
            start = expr.rindex('(')
            end = expr.find(')', start)
            if end == -1:
                raise ValueError("Mismatched parentheses")
            
            # Evaluate inside parentheses
            inner_result = self.parse_and_evaluate(expr[start+1:end], operator_defs)
            # Replace parentheses expression with result
            expr = expr[:start] + str(inner_result) + expr[end+1:]
        
        # Now handle operators
        for op in self.operators:
            if op in expr:
                # Find operator application
                pattern = r'(-?\d+\.?\d*)\s*' + re.escape(op) + r'\s*(-?\d+\.?\d*)'
                match = re.search(pattern, expr)
                if match:
                    a = float(match.group(1))
                    b = float(match.group(2))
                    formula = operator_defs[op]['formula']
                    result = self.evaluate_single_operation(a, b, formula)
                    return result
        
        # If no operators found, it should be a single number
        try:
            return float(expr)
        except ValueError:
            raise ValueError(f"Invalid expression: {expr}")

    def evaluate_expression(self, expr: str, operator_defs: Dict[str, Dict]) -> float:
        """Evaluate a complete expression with custom operators"""
        try:
            return self.parse_and_evaluate(expr, operator_defs)
        except Exception as e:
            raise ValueError(f"Error evaluating expression {expr}: {str(e)}")

    def generate_puzzle(self, complexity: int) -> OperatorPuzzle:
        """Generate a puzzle with specified complexity"""
        if complexity == 1:
            # Simple single operation
            operator, definition, coeffs = self.generate_simple_operator()
            a, b = random.randint(1, 10), random.randint(1, 10)
            question = f"{definition}\nCalculate {a}{operator}{b}."
            formula = definition.split("to be ")[1].strip(".")
            solution = self.evaluate_expression(f"{a}{operator}{b}", 
                                             {operator: {'formula': formula, 'coefficients': coeffs}})
            return OperatorPuzzle(
                question=question,
                solution=solution,
                solution_steps=[],
                complexity=complexity,
                operator_definitions={operator: formula}
            )
            
        elif complexity >= 4:
            # Complex operations with multiple rules
            op1, def1, coeffs1 = self.generate_simple_operator()
            op2, def2, coeffs2 = self.generate_simple_operator(op1=op1)
            op3, def3, coeffs3 = self.generate_simple_operator(op1=op1, op2=op2)
            
            question = (f"Let x and y represent two numbers. The new operations {op1}, {op2} and {op3} "
                       f"are defined as follows:\n{def1}\n{def2}\n{def3}\n"
                       f"Calculate (3{op1}4){op2}5{op3}6.")
                       
            operator_defs = {
                op1: {'formula': def1.split("to be ")[1].strip("."),
                      'coefficients': coeffs1},
                op2: {'formula': def2.split("to be ")[1].strip("."),
                      'coefficients': coeffs2},
                op3: {'formula': def3.split("to be ")[1].strip("."),
                      'coefficients': coeffs3}
            }
            
            solution = self.evaluate_expression(f"(3{op1}4){op2}5{op3}6", operator_defs)
            return OperatorPuzzle(
                question=question,
                solution=solution,
                solution_steps=[],
                complexity=complexity,
                operator_definitions={op: def_dict['formula'] 
                                    for op, def_dict in operator_defs.items()}
            )
        
        else:
            # Medium complexity
            operator, definition, coeffs = self.generate_simple_operator()
            nums = [random.randint(1, 10) for _ in range(3)]
            question = f"{definition}\nCalculate ({nums[0]}{operator}{nums[1]}){operator}{nums[2]}."
            formula = definition.split("to be ")[1].strip(".")
            solution = self.evaluate_expression(f"({nums[0]}{operator}{nums[1]}){operator}{nums[2]}", 
                                             {operator: {'formula': formula, 'coefficients': coeffs}})
            return OperatorPuzzle(
                question=question,
                solution=solution,
                solution_steps=[],
                complexity=complexity,
                operator_definitions={operator: formula}
            )

def verify_llm_answer(llm_response: str, expected_solution: float, tolerance: float = 1e-6) -> bool:
    """Verify LLM's answer considering potential format issues"""
    try:
        # Try to extract answer from <<<>>> format
        match = re.search(r'<<<(.+?)>>>', llm_response)
        if match:
            # Extract the list of values and symbols
            content = match.group(1)
            # Remove list brackets if present
            content = content.strip('[]')
            # Split by commas if present, otherwise by spaces
            elements = content.split(',') if ',' in content else content.split()
            # Try to find any number in the response
            numbers = re.findall(r'-?\d*\.?\d+', content)
            if numbers:
                for num in numbers:
                    if abs(float(num) - expected_solution) <= tolerance:
                        return True
    except:
        # If format is incorrect, try to find any number in the response
        numbers = re.findall(r'-?\d*\.?\d+', llm_response)
        if numbers:
            for num in numbers:
                if abs(float(num) - expected_solution) <= tolerance:
                    return True
    return False

def generate_dataset(output_dir: str, num_samples: int = 100):
    """Generate a dataset of new operator puzzles"""
    os.makedirs(output_dir, exist_ok=True)
    generator = NewOperatorGenerator()
    
    for i in range(num_samples):
        try:
            # Select complexity based on weights
            if i % 2 == 0:
                complexity = 5#random.choices(list(generator.complexity_weights.keys()),
                                    #   list(generator.complexity_weights.values()))[0]
            else:
                complexity = 2
            
            puzzle = generator.generate_puzzle(complexity)
            
            # Create sample directory
            sample_dir = os.path.join(output_dir, f'sample_{i}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save question and solution
            with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                f.write(puzzle.question + "\nReturn your answer with format <<<list of values and symbols>>>")
                
            with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                json.dump({
                    'solution': puzzle.solution,
                    'complexity': puzzle.complexity,
                    'operator_definitions': puzzle.operator_definitions
                }, f, indent=2)
                
            print(f"Generated sample {i}: Complexity {puzzle.complexity}, Solution: {puzzle.solution}")
        except Exception as e:
            print(f"Error generating sample {i}: {str(e)}")
            continue

if __name__ == "__main__":
    output_dir = '../dataset_gather/new_operator_dataset'
    generate_dataset(output_dir, num_samples=200)
