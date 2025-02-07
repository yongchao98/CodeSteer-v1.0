import random
import os
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

@dataclass
class StringSplittingProblem:
    initial_A: int
    initial_B: int
    initial_C: int
    description: str
    complexity: int  # 1-5 scale based on number of methods and initial counts

@dataclass
class StringSplittingSolution:
    final_A: int
    final_B: int
    final_C: int
    final_X: int
    final_Y: int
    final_Z: int

class StringSplittingGenerator:
    def __init__(self):
        # Define multiple templates for each method description to introduce variability
        self.method_descriptions_templates = {
            1: [
                "Method 1: One unit of machine A can be dismantled into two units of part X and one unit of part Y per round.",
                "Method 1: Dismantling a single machine A yields two part X and one part Y each cycle.",
                "Method 1: Each round, one machine A breaks down into two part X and one part Y."
            ],
            2: [
                "Method 2: Two units of machine B can be dismantled into one unit of part X per round.",
                "Method 2: Dismantling two machine B units produces one part X each cycle.",
                "Method 2: Every round, dismantling two machines B results in one part X."
            ],
            3: [
                "Method 3: Two units of machine C can be dismantled into one unit of part Y per round.",
                "Method 3: Dismantling two machine C units yields one part Y each cycle.",
                "Method 3: Each round, two machines C break down into one part Y."
            ],
            4: [
                "Method 4: One unit of machine B and one unit of machine C can be combined into one unit of machine A per round.",
                "Method 4: Combining one machine B and one machine C results in one machine A each cycle.",
                "Method 4: Every round, merging one machine B with one machine C creates one machine A."
            ],
            5: [
                "Method 5: One unit of part X and one unit of part Y can be combined into one unit of part Z per round.",
                "Method 5: Combining one part X with one part Y produces one part Z each cycle.",
                "Method 5: Each round, one part X and one part Y merge to form one part Z."
            ]
        }

    def generate_problem(self, complexity: int) -> Tuple[StringSplittingProblem, StringSplittingSolution]:
        """
        Generate a String Splitting problem instance. Complexity determines the range of initial machine counts.
        """
        # Define initial machine counts based on complexity
        if complexity == 1:
            initial_A = random.randint(1, 3)
            initial_B = random.randint(1, 3)
            initial_C = random.randint(1, 3)
        elif complexity == 2:
            initial_A = random.randint(2, 4)
            initial_B = random.randint(2, 4)
            initial_C = random.randint(2, 4)
        elif complexity == 3:
            initial_A = random.randint(3, 6)
            initial_B = random.randint(3, 6)
            initial_C = random.randint(3, 6)
        elif complexity == 4:
            initial_A = random.randint(4, 7)
            initial_B = random.randint(4, 7)
            initial_C = random.randint(4, 7)
        else:
            initial_A = random.randint(5, 10)
            initial_B = random.randint(5, 10)
            initial_C = random.randint(5, 10)

        # Select a random description template for each method to ensure variability
        selected_methods_descriptions = {}
        for method_num, templates in self.method_descriptions_templates.items():
            selected_methods_descriptions[method_num] = random.choice(templates)

        # Construct problem description
        description = (
            f"There is a dismantling engineer who has old machines A, B, and C. "
            f"He discovered that he can obtain a batch of new parts X, Y, and Z through the following methods:\n"
        )
        for method_num in sorted(selected_methods_descriptions.keys()):
            description += f"{selected_methods_descriptions[method_num]}\n"

        problem_description = (
            f"{description}\n"
            f"Xiao Li continuously cycles through these methods in the given order, executing one method at a time. "
            f"After Method 5, start again from Method 1. Continue this cycle until no method can be executed. "
            f"Initially, there are {initial_A} units of Machine A, {initial_B} units of Machine B, and {initial_C} units of Machine C. "
            f"How many parts X, Y, and Z can Li ultimately obtain, and how many machines A, B, and C will remain?"
        )

        problem = StringSplittingProblem(
            initial_A=initial_A,
            initial_B=initial_B,
            initial_C=initial_C,
            description=problem_description,
            complexity=complexity
        )

        solution = self.compute_solution(problem)

        return problem, solution

    def compute_solution(self, problem: StringSplittingProblem) -> StringSplittingSolution:
        """
        Simulate the execution of methods to compute the final counts of machines and parts.
        """
        A = problem.initial_A
        B = problem.initial_B
        C = problem.initial_C
        X = 0
        Y = 0
        Z = 0

        method_order = [1, 2, 3, 4, 5]
        method_index = 0

        # To track the number of executions
        execution_steps = []

        while True:
            current_method = method_order[method_index % len(method_order)]
            executed = False

            if current_method == 1:
                if A >= 1:
                    A -= 1
                    X += 2
                    Y += 1
                    executed = True
                    execution_steps.append(f"Method 1 executed: A=1 -> A={A}, X=+2 -> X={X}, Y=+1 -> Y={Y}")
            elif current_method == 2:
                if B >= 2:
                    B -= 2
                    X += 1
                    executed = True
                    execution_steps.append(f"Method 2 executed: B=2 -> B={B}, X=+1 -> X={X}")
            elif current_method == 3:
                if C >= 2:
                    C -= 2
                    Y += 1
                    executed = True
                    execution_steps.append(f"Method 3 executed: C=2 -> C={C}, Y=+1 -> Y={Y}")
            elif current_method == 4:
                if B >= 1 and C >= 1:
                    B -= 1
                    C -= 1
                    A += 1
                    executed = True
                    execution_steps.append(f"Method 4 executed: B=1 -> B={B}, C=1 -> C={C}, A=+1 -> A={A}")
            elif current_method == 5:
                if X >= 1 and Y >= 1:
                    X -= 1
                    Y -= 1
                    Z += 1
                    executed = True
                    execution_steps.append(f"Method 5 executed: X=1 -> X={X}, Y=1 -> Y={Y}, Z=+1 -> Z={Z}")

            if executed:
                method_index += 1
            else:
                method_index += 1
                # Check if any method can still be executed
                can_execute_any = False
                for m in method_order:
                    if self.can_execute(m, A, B, C, X, Y):
                        can_execute_any = True
                        break
                if not can_execute_any:
                    break

        # Optional: Print the execution steps for debugging
        # for step in execution_steps:
        #     print(step)

        return StringSplittingSolution(
            final_A=A,
            final_B=B,
            final_C=C,
            final_X=X,
            final_Y=Y,
            final_Z=Z
        )

    def can_execute(self, method_num: int, A: int, B: int, C: int, X: int, Y: int) -> bool:
        """
        Check if a specific method can be executed given the current counts.
        """
        if method_num == 1:
            return A >= 1
        elif method_num == 2:
            return B >= 2
        elif method_num == 3:
            return C >= 2
        elif method_num == 4:
            return B >= 1 and C >= 1
        elif method_num == 5:
            return X >= 1 and Y >= 1
        return False

def generate_dataset(output_dir: str, num_samples_per_complexity: int = 50):
    """
    Generate a dataset of String Splitting problems with varying complexities.
    Each sample contains a question prompt and its solution.
    """
    os.makedirs(output_dir, exist_ok=True)
    generator = StringSplittingGenerator()

    # Define complexity levels and number of samples for each
    complexity_levels = {
        1: num_samples_per_complexity,  # Simple
        2: num_samples_per_complexity,
        3: num_samples_per_complexity,
        4: num_samples_per_complexity,
        5: num_samples_per_complexity  # Complex
    }

    sample_id = 0
    for complexity, num_samples in complexity_levels.items():
        for _ in range(num_samples):
            problem, solution = generator.generate_problem(complexity)

            # Create sample directory
            sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
            os.makedirs(sample_dir, exist_ok=True)

            # Format question
            question = (
                f"{problem.description}\n\n"
                f"Generate the answer in the format <<<a string representing the outcome in the order of machines A, B, C, parts X, Y, Z>>>. "
                f"For example, if Machine A has 1 unit, Machine B has 0 units, Machine C has 1 unit, Part X has 1 unit, Part Y has 1 unit, and Part Z has 1 unit, it should be represented as [\"1\", \"0\", \"1\", \"1\", \"1\", \"1\"].\n"
                f"Your answer:"
            )

            # Prepare expected answer
            expected_answer = [
                str(solution.final_A),
                str(solution.final_B),
                str(solution.final_C),
                str(solution.final_X),
                str(solution.final_Y),
                str(solution.final_Z)
            ]

            # Save question and solution
            with open(os.path.join(sample_dir, 'question.txt'), 'w', encoding='utf-8') as f:
                f.write(question)

            with open(os.path.join(sample_dir, 'solution.json'), 'w', encoding='utf-8') as f:
                json.dump({
                    'expected_answer': expected_answer,
                    'final_counts': {
                        'A': solution.final_A,
                        'B': solution.final_B,
                        'C': solution.final_C,
                        'X': solution.final_X,
                        'Y': solution.final_Y,
                        'Z': solution.final_Z
                    },
                    'complexity': problem.complexity
                }, f, ensure_ascii=False, indent=2)

            sample_id += 1
            if sample_id % 100 == 0:
                print(f"Generated {sample_id} samples...")

    print(f"Dataset generation complete. Total samples: {sample_id}")

def check_answer(expected: List[str], actual: str) -> bool:
    """
    Check the correctness of the TaskLLM's answer.
    The answer should be in the format <<<["A", "B", "C", "X", "Y", "Z"]>>>
    This function parses the actual answer and compares it with the expected answer.
    """
    import re

    # Extract the JSON-like list from the answer using regex
    match = re.search(r'<<<\s*(\[[^\]]+\])\s*>>>', actual)
    if not match:
        print("Answer format is incorrect. Missing <<<>>> delimiters or list.")
        return False

    list_str = match.group(1)

    try:
        # Safely evaluate the list string
        parsed_answer = json.loads(list_str.replace("'", '"'))
    except json.JSONDecodeError:
        print("Failed to parse the list inside <<<>>>.")
        return False

    if not isinstance(parsed_answer, list):
        print("Parsed answer is not a list.")
        return False

    if len(parsed_answer) != 6:
        print(f"Expected 6 elements in the list, got {len(parsed_answer)}.")
        return False

    # Compare each element as strings
    for exp, act in zip(expected, parsed_answer):
        if exp != act:
            print(f"Mismatch: expected {exp}, got {act}.")
            return False

    return True

if __name__ == "__main__":
    # Define the output directory for the dataset
    output_directory = '../dataset_gather/string_splitting'
    # Generate the dataset
    generate_dataset(output_directory, num_samples_per_complexity=40)

    # Example of how to use the check_answer function
    # Assuming you have an LLM's response stored in 'llm_response'
    # and the expected answer is ['1', '0', '1', '1', '1', '1']
    """
    llm_response = '<<<["1", "0", "1", "1", "1", "1"]>>>'
    expected = ["1", "0", "1", "1", "1", "1"]
    is_correct = check_answer(expected, llm_response)
    print("Is the LLM's answer correct?", is_correct)
    """
