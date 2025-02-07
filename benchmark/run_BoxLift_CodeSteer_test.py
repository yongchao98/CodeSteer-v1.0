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
from LLaMA_Factory.src.llamafactory.chat.chat_model import run_response

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

def run_boxlift(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM):
    print('\n' + '*'*30)
    print(f'BoxLift, Model_name: {model_name}, CodeSteer\n')
    base_save_code_dir = save_input_dir + f'/result_boxlift_{CodeSteer_LLM}_{model_name}_MTD_{max_tree_depth}_CodeSteer_1'

    if not os.path.exists(base_save_code_dir):
        os.makedirs(base_save_code_dir)

    lifted_box_ratio_list = []
    lifted_weight_ratio_list = []
    total_sample_num = 0
    total_correct_num = 0

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

            response_list = [];
            CodeSteer_output_prompt_guidance_list = [];
            CodeSteer_input_prompt_list = [code_text_choice_prompt + question];
            CodeSteer_input_prompt_training_list = [code_text_choice_prompt + question]

            ############ Starting first guidance ############
            #starting_prompt_choice = GPT_response("", code_text_choice_prompt + question, model_name=model_name, code_interpreter=False,
            #                                    user_prompt_list=[code_text_choice_prompt + question], response_total_list=[], logprobs=False)
            messages = message_construct_llama_func([code_text_choice_prompt + question], [])
            starting_prompt_choice = run_response(messages, args_path)

            print(f'Starting prompt choice: {starting_prompt_choice}')
            user_prompt_list = [starting_prompt_choice + question]
            CodeSteer_output_prompt_guidance_list.append(starting_prompt_choice)
            response = GPT_response('', user_prompt_list[0], model_name=model_name, code_interpreter=False,
                                    user_prompt_list=user_prompt_list, response_total_list=response_list,
                                    logprobs=False)
            response_list.append(response)
            # print(f'\nResponse_0: {response}\n')

            ############ Further rounds of guidance ############
            for tree_depth in range(max_tree_depth):
                code_block_list = extract_code(response)
                if len(code_block_list) > 0:
                    code_complexity_summary, code_complexity_score = analyze_code_and_explain(code_block_list[0])
                    if code_complexity_score <= 2:
                        code_complexity_summary += '\nThe generated code may not be complex enough to carry out symbolic computing for solving the task.'
                    with open(save_code_dir + f"/code_1_{tree_depth}.py", "w") as f:
                        f.write(code_block_list[0])

                    try:
                        result = subprocess.run(
                            ["python3", save_code_dir + f"/code_1_{tree_depth}.py"],
                            capture_output=True, text=True, timeout=10
                        )
                        output = result.stdout
                        errors = result.stderr
                    except subprocess.TimeoutExpired as e:
                        output = e.stdout if e.stdout else ""
                        errors = e.stderr if e.stderr else ""
                        errors += f"\nTimeoutExpired: Command '{e.cmd}' timed out after {e.timeout} seconds"
                    response = response + f'\nThe execution result from the generated code is:\noutput: {output}, errors: {errors}'

                check_code_saving_path = save_code_dir + f"/check_code_1_{tree_depth}.py"
                check_result = LLM_answer_code_checker(question, response, check_code_saving_path)

                CodeSteer_input_prompt_head = f'''{decision_prompt} {question}\n'''
                if len(code_block_list) > 0:
                    print('\n############True#############\n')
                    CodeSteer_input_prompt = f'''The response from TaskLLM is: {response}\n\nThe feedback from the checking agent is:\n{check_result}\n\nThe summary of generated code complexity is: {code_complexity_summary}\n\n''' + \
                                             f'''The final returned guidance prompt should be of the format <<<guidance prompt content>>>.'''

                else:
                    CodeSteer_input_prompt = f'''The response from TaskLLM is: {response}\n\nThe feedback from the checking agent is:\n{check_result}\n\n''' + \
                                             f'''The final returned guidance prompt should be of the format <<<guidance prompt content>>>.'''

                CodeSteer_input_prompt_total = CodeSteer_input_prompt_head + CodeSteer_input_prompt
                CodeSteer_input_prompt_list.append(CodeSteer_input_prompt_total)
                CodeSteer_input_prompt_training_list.append(CodeSteer_input_prompt)
                #response_text = GPT_response("", '', model_name=model_name, code_interpreter=False,
                #                             user_prompt_list=CodeSteer_input_prompt_list,
                #                             response_total_list=CodeSteer_output_prompt_guidance_list, logprobs=False)
                #matches = re.findall(r'<<<(.*?)>>>', response_text, re.DOTALL)
                #guidance_prompt = matches[-1] if matches else response_text

                messages = message_construct_llama_func(CodeSteer_input_prompt_training_list, CodeSteer_output_prompt_guidance_list)
                guidance_prompt = run_response(messages, args_path)

                print(f'\nGuidance prompt_{tree_depth + 1}: {guidance_prompt}\n')

                CodeSteer_output_prompt_guidance_list.append(guidance_prompt)
                if '<<<Code>>>' in guidance_prompt:
                    guidance_prompt = with_COT_code_output_prompt
                elif '<<<Text>>>' in guidance_prompt:
                    guidance_prompt = text_output_prompt
                elif '<<<Return Answer>>>' in guidance_prompt or 'Return Answer' in guidance_prompt or '<<<Terminate>>>' in guidance_prompt or 'Terminate' in guidance_prompt:
                    break
                user_prompt_list.append(guidance_prompt)

                response = GPT_response('', user_prompt_list[0], model_name=model_name, code_interpreter=False,
                                        user_prompt_list=user_prompt_list, response_total_list=response_list,
                                        logprobs=False)

                response_list.append(response)
                # print(f'\nResponse_{tree_depth}: {response}\n')
            save_file_func(save_code_dir, response_list, user_prompt_list, question, CodeSteer_input_prompt_list,
                           CodeSteer_input_prompt_training_list, CodeSteer_output_prompt_guidance_list)

            ## Evaluation
            estimated_steps = estimate_steps(boxes, lifters)
            print(f"Estimated number of steps: {estimated_steps}")
            response = response_list[-1]

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

    run_info = f"CodeSteer, BoxLift, {CodeSteer_LLM}, {model_name}, MTD_{max_tree_depth}_CodeSteer_1\n"
    run_info_result = f'total_lifted_weight_ratio: {np.mean(lifted_weight_ratio_list)}, total_lifted_box_ratio: {np.mean(lifted_box_ratio_list)}, correct/all:{total_correct_num}/{total_sample_num}\n'
    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}.txt")
    log_run_info(log_file_result, run_info + run_info_result)