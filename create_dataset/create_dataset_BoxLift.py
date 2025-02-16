import random
import math
import json
from typing import List, Tuple, Dict
import os

def create_test_case(num_boxes: int, num_lifters: int, min_box_weight: int, max_box_weight: int,
                     min_lifter_capacity: int, max_lifter_capacity: int) -> Tuple[List[int], List[int]]:
    """
    Create a test case for the BoxLift task.
    
    :return: A tuple containing a list of box weights and a list of lifter capacities.
    """
    boxes = [random.randint(min_box_weight, max_box_weight) for _ in range(num_boxes)]
    lifters = [random.randint(min_lifter_capacity, max_lifter_capacity) for _ in range(num_lifters)]
    return boxes, lifters

def create_prompt(boxes: List[int], lifters: List[int]) -> str:
    """
    Create a prompt for the LLM based on the given boxes and lifters.
    
    :param boxes: List of box weights.
    :param lifters: List of lifter capacities.
    :return: A string containing the prompt for the LLM.
    """
    prompt = f"""Task: BoxLift

You are given a list of boxes with the following weights: {boxes}
And a list of lifters with the following maximum lifting capacities: {lifters}

Your task is to assign the lifters to lift all the boxes in multiple steps, following these rules:
1. Multiple boxes can be lifted in each step.
2. Each lifter can only lift one box at a time.
3. Each lifting agent can be used only once in each step.
4. Multiple lifters can combine together to lift one box if the box is too heavy for a single lifter.
5. Try to lift all the boxes using the minimum number of steps possible.

Please provide your solution in the following format:
Step 1: [(Box weight, [Lifter indices]), (Box weight, [Lifter indices]), ...]
Step 2: [(Box weight, [Lifter indices]), (Box weight, [Lifter indices]), ...]
...

For example:
Step 1: [(50, [0, 2]), (30, [1]), (20, [3])]
This means in Step 1, lifters 0 and 2 are lifting a box weighing 50, lifter 1 is lifting a box weighing 30, and lifter 3 is lifting a box weighing 20.

Ensure all boxes are lifted and provide the most efficient solution possible.
"""
    return prompt

def verify_solution(boxes: List[int], lifters: List[int], solution: str) -> Tuple[bool, List[int]]:
    """
    Verify the correctness of the LLM's solution.
    
    :param boxes: List of box weights.
    :param lifters: List of lifter capacities.
    :return: A tuple containing a boolean indicating if the solution is correct and a list of remaining boxes.
    """
    remaining_boxes = boxes.copy()
    
    steps = solution.split("Step")[1:]  # Split the solution into steps
    
    for step in steps:
        used_lifters = set()
        assignments = eval(step.split(":")[1].strip())
        
        for box_weight, lifter_indices in assignments:
            # Check if the box weight is valid
            if box_weight not in remaining_boxes:
                return False, remaining_boxes
            
            # Check if lifters are used only once per step
            if any(index in used_lifters for index in lifter_indices):
                return False, remaining_boxes
            
            # Check if lifters can lift the box
            if sum(lifters[i] for i in lifter_indices) < box_weight:
                return False, remaining_boxes
            
            remaining_boxes.remove(box_weight)
            used_lifters.update(lifter_indices)
    
    return len(remaining_boxes) == 0, remaining_boxes


def estimate_steps(boxes: List[int], lifters: List[int]) -> int:
    """
    Estimate a reasonable number of steps to accomplish the BoxLift task using a greedy heuristic.

    :param boxes: List of box weights.
    :param lifters: List of lifter capacities.
    :return: Estimated number of steps.
    """
    remaining_boxes = sorted(boxes, reverse=True)  # Sort boxes in descending order
    steps = 0

    while remaining_boxes:
        steps += 1
        available_lifters = lifters.copy()

        i = 0
        while i < len(remaining_boxes) and available_lifters:
            box = remaining_boxes[i]
            combined_strength = sum(available_lifters)

            if combined_strength >= box:
                # Lift the box using as many lifters as needed
                lift_strength = 0
                used_lifters = []
                for j, lifter in enumerate(available_lifters):
                    lift_strength += lifter
                    used_lifters.append(j)
                    if lift_strength >= box:
                        break

                # Remove the used lifters and the lifted box
                for j in reversed(used_lifters):
                    available_lifters.pop(j)
                remaining_boxes.pop(i)
            else:
                i += 1  # Move to the next box if we can't lift this one

    return steps

def write_test_case(filename: str, boxes: List[int], lifters: List[int]):
    """
    Write the test case (boxes and lifters) to a JSON file.
    
    :param filename: Name of the file to write to.
    :param boxes: List of box weights.
    :param lifters: List of lifter capacities.
    """
    data = {
        "boxes": boxes,
        "lifters": lifters
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

def read_test_case(filename: str) -> Tuple[List[int], List[int]]:
    """
    Read the test case (boxes and lifters) from a JSON file.
    
    :param filename: Name of the file to read from.
    :return: A tuple containing a list of box weights and a list of lifter capacities.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["boxes"], data["lifters"]


if __name__ == "__main__":
    repeat_num = 20
    Saving_path = f'../dataset_gather/BoxLift_dataset/'
    if not os.path.exists(Saving_path):
        os.makedirs(Saving_path)

    for num_boxes, num_lifters, min_box_weight, max_box_weight, min_lifter_capacity, max_lifter_capacity in \
            [(10, 3, 10, 100, 40, 80), (15, 4, 20, 200, 30, 120), (20, 5, 30, 300, 40, 160), (25, 6, 40, 400, 50, 200),
             (10, 4, 10, 100, 40, 80), (15, 5, 20, 200, 30, 120), (20, 6, 30, 300, 40, 160), (25, 7, 40, 400, 50, 200),
             (16, 5, 20, 200, 30, 120), (21, 6, 30, 300, 40, 160)]:
        print(f'Creating dataset for num_boxes = {num_boxes}, num_lifters = {num_lifters}, repeat_num = {repeat_num}')
        if not os.path.exists(Saving_path+f'/BoxLift_{num_boxes}_{num_lifters}'):
            os.makedirs(Saving_path+f'/BoxLift_{num_boxes}_{num_lifters}')

        for iteration_num in range(repeat_num):
            boxes, lifters = create_test_case(num_boxes, num_lifters, min_box_weight, max_box_weight, min_lifter_capacity, max_lifter_capacity)
            estimated_steps = estimate_steps(boxes, lifters)
            print(f"Estimated number of steps: {estimated_steps}")
            os.makedirs(Saving_path+f'/BoxLift_{num_boxes}_{num_lifters}/BoxLift{iteration_num}')
            write_test_case(Saving_path+f'/BoxLift_{num_boxes}_{num_lifters}/BoxLift{iteration_num}/BoxLift.json', boxes, lifters)
