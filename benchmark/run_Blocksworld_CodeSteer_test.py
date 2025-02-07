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
from models.src.llamafactory.chat.chat_model import run_response

# Define types
State = List[List[str]]
Action = Tuple[str, str, str]  # (block, from, to)

def state_to_prompt(state: State, goal: State) -> str:
    """
    Convert a Blocksworld state to a prompt description for the LLM.
    """
    prompt = "Blocksworld Task:\n\nInitial State:\n"
    for stack, blocks in state.items():
        prompt += f"{stack}: {' '.join(blocks)}\n"

    prompt += "\nGoal State:\n"
    for stack, blocks in goal.items():
        prompt += f"{stack}: {' '.join(blocks)}\n"

    prompt += "\nPlease provide a series of moves to reach the goal state. " \
              "You can only move one block at a time. And that box should be the top box of the stack. " \
              "Note that from the left to the right in each stack is the order from the bottom to the top of boxes. " \
              "For example, in stack A B C D, A is the bottom box and D is the top box so that you can only move D in this case. "
    prompt += "***Be careful that you can only pick up the top box in each stack. Check this rule before your move!***. "
    prompt += "\nEach move should be in the format: 'Move [block] from [source] to [destination]'. "
    prompt += "You cannot create new stacks but only move among the existing stacks. "
    prompt += "Separate each move with a newline. Surround the answer with <<<content>>>. "
    prompt += "Answer with the required format like the example: <<<Move B from 2 to table\nMove A from 1 to 2\nMove C from 3 to 1\nMove D from 3 to 2\nMove B from 1 to 2>>>\n"
    prompt += "Each action should be separated by separate line. Your answer: \n"

    return prompt

def read_state_from_file(filename: str) -> Tuple[State, State]:
    """
    Read the initial and goal states from a text file.
    """
    initial_state = {}
    goal_state = {}
    current_state = initial_state

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line == "Initial State:":
            current_state = initial_state
        elif line == "Goal State:":
            current_state = goal_state
        elif line:
            parts = line.split(': ')
            stack = parts[0] if len(parts) > 1 else parts[0][:-1]
            blocks = parts[1].split() if len(parts) > 1 else []
            current_state[stack] = blocks

    return initial_state, goal_state

def extract_equation_with_GPT4(response):
    prompt = 'Your task is to extract the final answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<list>>>, like <<<Yes>>>, <<<No>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<No answer found>>>.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs=False)
    return extract_equation

def validate_response(initial_state: State, goal_state: State, response: str) -> Tuple[bool, str]:
    """
    Validate the LLM's response and check if it reaches the goal state.
    """
    current_state = {stack: blocks.copy() for stack, blocks in initial_state.items()}
    #print('current_state:', current_state)
    moves = response.strip().split('\n')
    #print('goal state:', goal_state)
    for move in moves:
        parts = move.split()
        if len(parts) != 6 or parts[0] != "Move" or parts[2] != "from" or parts[4] != "to":
            return False, f"Invalid move format: {move}"

        block, source, destination = parts[1], parts[3], parts[5]
        if 'stack' not in source:
            source = 'stack' + source
        if 'stack' not in destination:
            destination = 'stack' + destination

        if source not in current_state or destination not in current_state:
            return False, f"Invalid source or destination stack: {move}"

        if not current_state[source] or current_state[source][-1] != block:
            return False, f"Invalid move: {move}. Block {block} is not at the top of the source stack."

        # Move the block
        moved_block = current_state[source].pop()
        current_state[destination].append(moved_block)
        #print('current_state:', current_state)

    def compare_states(state1, state2):
        state1_non_empty = {k: v for k, v in state1.items() if v}
        state2_non_empty = {k: v for k, v in state2.items() if v}
        return state1_non_empty == state2_non_empty

    # Check if the final state matches the goal state
    if compare_states(current_state, goal_state):
        return True, "Goal state reached successfully!"
    else:
        return False, "The final state does not match the goal state."

def run_blocksworld(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM):
    print('\n' + '*'*30)
    print(f'Blocksworld, Model_name: {model_name}, CodeSteer\n')
    base_save_code_dir = save_input_dir + f'/result_blocksworld_{CodeSteer_LLM}_{model_name}_MTD_{max_tree_depth}_CodeSteer_1'

    if not os.path.exists(base_save_code_dir):
        os.makedirs(base_save_code_dir)

    total_sample_num = 0
    total_correct_num = 0

    for num_blocks, initial_stacks, goal_stacks in [
        (2, 3, 2), (2, 3, 3), (2, 4, 2), (2, 4, 3), (2, 4, 4), (2, 5, 2), (2, 5, 3), (2, 5, 4),
        (3, 3, 2), (3, 3, 3), (3, 4, 2), (3, 4, 3), (3, 4, 4), (3, 5, 2), (3, 5, 3), (3, 5, 4),
        (4, 3, 2), (4, 3, 3), (4, 4, 2), (4, 4, 3), (4, 4, 4), (4, 5, 2), (4, 5, 3), (4, 5, 4)
    ]:
        #for index in range(5):
        for index in range(2):
            total_sample_num += 1
            dataset_base_dir_sample = os.path.join(dataset_input_dir, f"{num_blocks}_{initial_stacks}_{goal_stacks}_{index}/")
            # Read states from file
            initial_state, goal_state = read_state_from_file(dataset_base_dir_sample + f"blocksworld_task.txt")
            print(f'num_blocks: {num_blocks}, initial_stacks: {initial_stacks}, goal_stacks: {goal_stacks}, index: {index}')

            save_code_dir = os.path.join(base_save_code_dir, f"{num_blocks}_{initial_stacks}_{goal_stacks}_{index}/")
            if not os.path.exists(save_code_dir):
                os.makedirs(save_code_dir)

            # Generate prompt from the read states
            question = state_to_prompt(initial_state, goal_state)

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
            response = response_list[-1]
            original_response = response

            code_block_list = extract_code(response)
            for index, code_string in enumerate(code_block_list):
                with open(save_code_dir + f"/code_1_{index}.py", "w") as f:
                    f.write(code_string)
                #print(f'code_{index}:\n {code_string}')

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

                    '''
                    result = subprocess.run(
                        ["python3", save_code_dir + f"/code_1_0.py"],
                        capture_output=True, text=True, timeout=15)
                    '''
                    response = result.stdout
                    errors = result.stderr
                except Exception as e:
                    print('Code execution error')
                    response = ""
                    errors = str(e)

            extracted_text_1, _ = extract_and_check(response)
            if extracted_text_1 == '':
                print('\n*****Response:', response)
                extracted_text_1 = extract_equation_with_GPT4(response)

            extracted_text_2, _ = extract_and_check(original_response)
            if extracted_text_2 == '':
                print('\n*****Response:', response)
                extracted_text_2 = extract_equation_with_GPT4(original_response)

            is_valid_1, message_1 = validate_response(initial_state, goal_state, extracted_text_1)
            is_valid_2, message_2 = validate_response(initial_state, goal_state, extracted_text_2)

            print(f'True_false_result from response: {is_valid_1}')
            print(f'True_false_result from original_response: {is_valid_2}')
            print(f'extracted_text from response: {extracted_text_1}')
            print(f'extracted_text from original_response: {extracted_text_2}')

            print(f"Message: {message_1}")
            print(f"Message: {message_2}")

            with open(save_code_dir + f"/True_false_result_1.txt", "w") as f:
                f.write(str(is_valid_1))
            with open(save_code_dir + f"/True_false_result_2.txt", "w") as f:
                f.write(str(is_valid_2))
            with open(save_code_dir + f"/extracted_answer_1.txt", "w") as f:
                f.write(extracted_text_1)
            with open(save_code_dir + f"/extracted_answer_2.txt", "w") as f:
                f.write(extracted_text_2)

            if is_valid_1 == True or is_valid_2 == True:
                print('True')
                with open(save_code_dir + f"/success_failure.txt", "w") as f:
                    f.write('True')
                total_correct_num += 1

            else:
                print('False')
                with open(save_code_dir + f"/success_failure.txt", "w") as f:
                    f.write('False')

            print(f'Correct/all: {total_correct_num}/{total_sample_num}')

    run_info = f"CodeSteer, Blocksworld, {CodeSteer_LLM}, {model_name}, MTD_{max_tree_depth}_CodeSteer_1\n"
    run_info_result = f'correct/all:{total_correct_num}/{total_sample_num}\n'
    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}.txt")
    log_run_info(log_file_result, run_info + run_info_result)