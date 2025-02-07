import json
import re
import pandas as pd
import os
import subprocess
import sys
from openai import OpenAI
from generation_models import message_construct_func, message_construct_llama_func, GPT_response, count_total_tokens, extract_code, extract_and_check, LLM_answer_code_checker, save_file_func, paraphrase_with_GPT4, log_run_info
import random
import math
import json
from typing import List, Tuple, Dict
import time
import numpy as np
from prompt import *
from argparse import ArgumentParser
from symbolic_code_check import analyze_computational_approach, analyze_code_and_explain

def save_file_func_baselines(save_code_dir, response_list, user_prompt_list, question, system_message):
    data = {
        'question': question,
        'response_list': response_list,
        'user_prompt_list': user_prompt_list,
        'system_message': system_message
    }

    output_file = os.path.join(save_code_dir, 'conversation_data.json')

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving data: {str(e)}")

def extract_equation_with_GPT4_BoxLift(response):
    prompt = 'Your task is to extract the final answer from the given answer by another LLM:\n' \
             'Note that the equation should be in the form like <<<answer>>>, <<<Step 1: [(185, [0, 1]), (108, [0, 1])]\nStep 2: [(184, [0, 1]), (75, [0, 1])]\nStep 3: [(174, [0, 1]), (70, [0, 1])]\nStep 4: [(171, [0, 1]), (63, [0]), (34, [0])]\nStep 5: [(157, [0, 1]), (32, [0]), (31, [0])]>>>, \n' \
             'Here is the reponse, return your answer with the format <<<equation>>>, like <<<Step 1: [(185, [0, 1]), (108, [0, 1])]\nStep 2: [(184, [0, 1]), (75, [0, 1])]>>>. ' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list=[prompt + response], response_total_list=[], logprobs=False)
    return extract_equation

def create_prompt(boxes: List[int], lifters: List[int], estimated_steps) -> str:
    prompt = f"""Task: BoxLift

You are given a list of boxes with the following weights: {boxes}
And a list of lifters with the following maximum lifting capacities: {lifters}

Your task is to assign the lifters to lift all the boxes in multiple steps, following these rules:
1. Multiple boxes can be lifted in each step.
2. Each lifter can only lift one box at a time.
3. Each lifting agent can be used only once in each step.
4. Multiple lifters can combine together to lift one box if the box is too heavy for a single lifter.
5. Try to lift all the boxes using the minimum number of steps possible.
6. You need to lift all the boxes in less than or equal to {estimated_steps} steps.

Please provide your solution in the following format:
Step 1: [(Box weight, [Lifter indices]), (Box weight, [Lifter indices]), ...]
Step 2: [(Box weight, [Lifter indices]), (Box weight, [Lifter indices]), ...]
...

For example:
Step 1: [(50, [0, 2]), (30, [1]), (20, [3])]
This means in Step 1, lifters 0 and 2 are lifting a box weighing 50, lifter 1 is lifting a box weighing 30, and lifter 3 is lifting a box weighing 20.

Surround the answer with <<<content>>>.

For example, <<<Step 1: [(50, [0, 2]), (30, [1]), (20, [3])]\nStep 2: [(40, [0, 1]), (20, [2]), (20, [3])]\nStep 3:...>>>

Ensure all boxes are lifted and provide the most efficient solution possible.

Your answer:\n
"""
    return prompt


def verify_solution(boxes: List[int], lifters: List[int], solution: str, estimated_steps) -> Tuple[bool, List[int]]:
    remaining_boxes = boxes.copy()
    success_failure_list = []

    steps = solution.split("Step")[1:]  # Split the solution into steps
    if len(steps) > estimated_steps:
        success_failure = 'Too many steps'
        success_failure_list.append(success_failure)
        #return False, remaining_boxes, success_failure

    for index in range(min(estimated_steps, len(steps))):
        step = steps[index]
        used_lifters = set()
        try:
            assignments = eval(step.split(":")[1].strip())
            for box_weight, lifter_indices in assignments:
                # Check if the box weight is valid
                if box_weight not in remaining_boxes:
                    success_failure = 'Invalid box weight'
                    success_failure_list.append(success_failure)
                    #return False, remaining_boxes, success_failure

                elif any(index >= len(lifters) for index in lifter_indices):
                    success_failure = 'Invalid lifter index'
                    success_failure_list.append(success_failure)
                    #return False, remaining_boxes, success_failure

                # Check if lifters are used only once per step
                elif any(index in used_lifters for index in lifter_indices):
                    success_failure = 'Lifter used more than once'
                    success_failure_list.append(success_failure)
                    #return False, remaining_boxes, success_failure

                # Check if lifters can lift the box
                elif sum(lifters[i] for i in lifter_indices) < box_weight:
                    success_failure = 'Insufficient lifter strength'
                    success_failure_list.append(success_failure)
                    # return False, remaining_boxes, success_failure
                    #pass
                else:
                    remaining_boxes.remove(box_weight)
                    used_lifters.update(lifter_indices)
        except:
            success_failure = 'Invalid format'
            success_failure_list.append(success_failure)
            return False, remaining_boxes, success_failure

    return len(remaining_boxes) == 0, remaining_boxes, success_failure_list


def estimate_steps(boxes: List[int], lifters: List[int]) -> int:
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


def read_test_case(filename: str) -> Tuple[List[int], List[int]]:
    """
    Read the test case (boxes and lifters) from a JSON file.

    :param filename: Name of the file to read from.
    :return: A tuple containing a list of box weights and a list of lifter capacities.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["boxes"], data["lifters"]

def run_boxlift_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path):
    print('\n' + '*'*30)
    print(f'BoxLift, Model_name: {model_name}\n')
    base_save_code_dir = save_input_dir + f'/result_boxlift_{baseline_method_name}_{model_name}'

    if not os.path.exists(base_save_code_dir):
        os.makedirs(base_save_code_dir)

    lifted_box_ratio_list = []
    lifted_weight_ratio_list = []
    total_sample_num = 0
    total_correct_num = 0
    dataset_input_dir = './dataset_gather/BoxLift_dataset'

    for num_boxes, num_lifters, min_box_weight, max_box_weight, min_lifter_capacity, max_lifter_capacity in \
            [(10, 3, 10, 100, 40, 80), (15, 4, 20, 200, 30, 120), (20, 5, 30, 300, 40, 160), (25, 6, 40, 400, 50, 200)]:
        #for iteration_num in range(10):
        for iteration_num in range(5):
            total_sample_num += 1
            print(f'\n\nNum_boxes = {num_boxes}, Num_lifters = {num_lifters}, Iteration_num = {iteration_num}')
            save_code_dir = os.path.join(base_save_code_dir, f"{num_boxes}_{num_lifters}_{iteration_num}/")
            if not os.path.exists(save_code_dir):
                os.makedirs(save_code_dir)

            boxes, lifters = read_test_case(dataset_input_dir + f'/BoxLift_{num_boxes}_{num_lifters}/BoxLift{iteration_num}/BoxLift.json')
            print(f"Initial boxes: {boxes}")
            print(f"Initial lifters: {lifters}")

            estimated_steps = estimate_steps(boxes, lifters)
            print(f"Estimated number of steps: {estimated_steps}")
            question = create_prompt(boxes, lifters, estimated_steps)

            system_message = ''
            if baseline_method_name == '1_only_ques':
                user_prompt_list = [question]

            if model_name in ['o1', "o1-preview", 'o1-mini', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo',
                              "claude-3-sonnet-20240229",
                              "claude-3-opus-20240229", "claude-3-haiku-20240307"]:
                response = GPT_response(system_message, user_prompt_list[0], model_name=model_name,
                                        code_interpreter=False,
                                        user_prompt_list=user_prompt_list, response_total_list=[], logprobs=False)
            else:
                messages = message_construct_llama_func(user_prompt_list, [])
                response = run_response(messages, args_path)
            response_list = []
            response_list.append(response)

            save_file_func_baselines(save_code_dir, response_list, user_prompt_list, question, system_message)

            response = response_list[-1]
            original_response = response

            code_block_list = extract_code(response)
            for index, code_string in enumerate(code_block_list):
                with open(save_code_dir + f"/code_1_{index}.py", "w") as f:
                    f.write(code_string)

            # Test the generated code
            if not os.path.exists(save_code_dir + f"/code_1_0.py"):
                pass
            else:
                try:
                    result = subprocess.run(
                        ["python3", "-c", f"exec(open('{save_code_dir}/code_1_0.py').read()); print(result)"],
                        capture_output=True,
                        text=True,
                        timeout=15
                    )

                    response = result.stdout
                    errors = result.stderr
                except Exception as e:
                    print('Code execution error')
                    response = ""
                    errors = str(e)

            response_answer, _ = extract_and_check(response)
            print(f"Response: {response_answer}")

            if response_answer == '':
                response_answer = extract_equation_with_GPT4_BoxLift(response)

            is_correct, remaining, success_failure_list = verify_solution(boxes, lifters, response_answer,
                                                                          estimated_steps)

            print(f"Response: {response}")
            print(f"Response_answer: {response_answer}")
            print(f"Response is valid: {is_correct}")
            print(f'Initial boxes: {boxes}')
            print(f"Remaining boxes: {remaining}")
            print(f"Lifted box ratio: {(len(boxes) - len(remaining)) / len(boxes)}")
            print(f"Lifted weight ratio: {(sum(boxes) - sum(remaining)) / sum(boxes)}")
            lifted_box_ratio_list.append((len(boxes) - len(remaining)) / len(boxes))
            lifted_weight_ratio_list.append((sum(boxes) - sum(remaining)) / sum(boxes))

            with open(save_code_dir + "/Lifted_box_ratio_1.txt", "w") as f:
                f.write(str((len(boxes) - len(remaining)) / len(boxes)))
            with open(save_code_dir + "/Lifted_weight_ratio_1.txt", "w") as f:
                f.write(str((sum(boxes) - sum(remaining)) / sum(boxes)))
            with open(save_code_dir + "/response_answer.txt", "w") as f:
                f.write(response_answer)
            with open(save_code_dir + "/success_failure.txt", "w") as f:
                f.write(str(is_correct))

            if is_correct:
                print('True')
                with open(save_code_dir + f"/success_failure.txt", "w") as f:
                    f.write('True')
                total_correct_num += 1
            else:
                print('False')
                with open(save_code_dir + f"/success_failure.txt", "w") as f:
                    f.write('False')

            print(f'Total lifted weight ratio: {np.mean(lifted_weight_ratio_list)}')
            print(f'Total lifted box ratio: {np.mean(lifted_box_ratio_list)}')
            print(f'Correct/all: {total_correct_num}/{total_sample_num}')

            with open(base_save_code_dir + f"/total_lifted_weight_ratio.txt", "w") as f:
                f.write(str(np.mean(lifted_weight_ratio_list)))
            with open(base_save_code_dir + f"/total_lifted_box_ratio.txt", "w") as f:
                f.write(str(np.mean(lifted_box_ratio_list)))
            with open(base_save_code_dir + f"/lifted_weight_ratio_list.txt", "w") as f:
                f.write(str(lifted_weight_ratio_list))
            with open(base_save_code_dir + f"/lifted_box_ratio_list.txt", "w") as f:
                f.write(str(lifted_box_ratio_list))

    run_info = f"CodeSteer, BoxLift, {baseline_method_name}, {model_name}\n"
    run_info_result = f'total_lifted_weight_ratio: {np.mean(lifted_weight_ratio_list)}, total_lifted_box_ratio: {np.mean(lifted_box_ratio_list)}, correct/all:{total_correct_num}/{total_sample_num}\n'
    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}.txt")
    log_run_info(log_file_result, run_info + run_info_result)