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
import copy
import ast

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

def action_from_response(pg_dict_input, original_response_dict_list):
    pg_dict_current = copy.deepcopy(pg_dict_input)

    for original_response_dict in original_response_dict_list:
        transformed_dict = {}
        for key, value in original_response_dict.items():
            coordinates = tuple(map(float, re.findall(r"\d+\.?\d*", key)))
            match = re.match(r"move\((.*?),\s(.*?)\)", value)
            if match:
                item, location = match.groups()
                if "square" in location:
                    location = tuple(map(float, re.findall(r"\d+\.?\d*", location)))
                transformed_dict[coordinates] = [item, location]

        # Process each move with the current state
        for key, value in transformed_dict.items():
            current_pos = f"{key[0]}_{key[1]}"

            # Check if this is a box-target matching move
            if (value[0] in pg_dict_current[current_pos] and
                    isinstance(value[1], str) and
                    value[1] in pg_dict_current[current_pos] and
                    value[0].startswith('box_') and
                    value[1].startswith('target_') and
                    value[0][4:] == value[1][7:]):
                # Remove both box and target when matched
                pg_dict_current[current_pos].remove(value[0])
                pg_dict_current[current_pos].remove(value[1])

            # Check if this is a movement to another square
            elif (value[0] in pg_dict_current[current_pos] and
                  isinstance(value[1], tuple)):  # Only check coordinates for square movements
                # Calculate if move is to adjacent square
                if ((np.abs(key[0] - value[1][0]) == 0 and np.abs(key[1] - value[1][1]) == 1) or
                        (np.abs(key[0] - value[1][0]) == 1 and np.abs(key[1] - value[1][1]) == 0)):
                    # Move box to new location
                    target_pos = f"{value[1][0]}_{value[1][1]}"
                    pg_dict_current[current_pos].remove(value[0])
                    pg_dict_current[target_pos].append(value[0])

    return pg_dict_current


def score_in_training_set(pg_dict, response):
    success_failure = ''
    try:
        original_response_dict_list = json.loads(response)
        for original_response_dict in original_response_dict_list:
            for key, value in original_response_dict.items():
                coordinates = tuple(map(float, re.findall(r"\d+\.?\d*", key)))
                # match the item and location in the value
                match = re.match(r"move\((.*?),\s(.*?)\)", value)
    except:
        success_failure = 'response in the wrong format'

    if success_failure == 'response in the wrong format':
        print('\nResponse in the wrong format!\n')
        return pg_dict, success_failure
    elif success_failure == '':
        pg_dict_returned = action_from_response(pg_dict, original_response_dict_list)
        count = 0
        for key, value in pg_dict_returned.items():
            count += len(value)
        if count == 0:
            success_failure = 'success'
        elif success_failure == '':
            success_failure = 'failure after full execution'
        return pg_dict_returned, success_failure

def create_prompt(pg_row_num, pg_column_num, pg_dict):
    state_update_prompt = state_update_func(pg_row_num, pg_column_num, pg_dict)
    prompt = '''
You are a central planner tasked with directing agents in a grid-like field to move colored boxes to their corresponding color-coded targets. Each agent occupies a 1x1 square and can only interact with objects within its square. Agents can move a box to an adjacent square or directly to a target square of the same color. A square may contain multiple boxes and targets.



The squares are identified by their center coordinates (e.g., square[0.5, 0.5]). Actions are formatted as: move(box_color, destination), where box_color is the color of the box and destination is either a target of the same color or an adjacent square.



Your objective is to create a sequence of action plans that instructs each agent to match all boxes to their color-coded targets in the most efficient manner.



Please adhere to the following rules when specifying your action plan:



1. **Single Action per Agent**: Assign only one action to each agent at a time. However, the final answer shoule be a list of action plans for multiple steps.



2. **Unique Agent Keys**: Use unique keys for each agent in the JSON format action plan. The key should be the agent's coordinates in the format "Agent[x, y]".



3. **Prioritize Matching Boxes to Targets**: Always prioritize actions that will match a box to its target over moving a box to an adjacent square.



4. **Sequential Action Planning**: The whole returned answer should be a list of action plans for multiple steps, do not just return one step plan.



5. **Clear Formatting**: Ensure the action plan is clearly formatted in JSON, with each agent's action specified as a key-value pair.



6. **Conflict Resolution**: Ensure that no two agents are assigned actions that would interfere with each other.



7. **Optimize Efficiency**: Aim to minimize the number of moves required to match all boxes with their targets.



Here is the format for your action plan:
```json
[{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_red, target_red)"}, {"Agent[0.5, 1.5]":"move(box_blue, target_blue)", "Agent[2.5, 0.5]":"move...}, {...}...]
```
Include an agent in the action plan only if it has a task to perform next.

'''
    prompt += "Surround the answer with <<<content>>>. \n"
    prompt += f'''
    The current left boxes and agents are:
    {state_update_prompt}\n
    '''
    prompt += '''
    Please respond in the format: <<<list of action dictionary>>>, such as <<<[{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}, {"Agent[0.5, 1.5]":"move(box_blue, target_blue)"}, {...}...]>>>.\n
    Your answer:\n
    '''
    return prompt

def extract_equation_with_GPT4(response):
    prompt = 'Your task is to extract the final answer from the given answer by another LLM:\n' \
             'Note that the equation should be in the form like <<<answer>>>, <<<[{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}, {"Agent[0.5, 1.5]":"move(box_blue, target_blue)}, {...}...]>>>, \n' \
             'Here is the reponse, return your answer with the format <<<equation>>>, like <<<[{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}, {"Agent[0.5, 1.5]":"move(box_blue, target_blue)}, {...}...]>>>. ' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list=[prompt + response], response_total_list=[], logprobs=False)
    return extract_equation

def surround_index_func(row_num, coloum_num, row_index, coloum_index):
  surround_index_list = []
  for i, j in ([row_index-1, coloum_index], [row_index+1, coloum_index], [row_index, coloum_index-1], [row_index, coloum_index+1]):
    if i>=0 and i<=row_num-1 and j>=0 and j<=coloum_num-1 and not (i == row_index and j == coloum_index):
      surround_index_list.append([i+0.5,j+0.5])
  return surround_index_list

def state_update_func(pg_row_num, pg_column_num, pg_dict):
  pg_dict_copy = copy.deepcopy(pg_dict)
  state_update_prompt = ''
  for i in range(pg_row_num):
    for j in range(pg_column_num):
      square_item_list = pg_dict_copy[str(i+0.5)+'_'+str(j+0.5)]
      square_item_only_box = [item for item in square_item_list if item[:3]=='box']
      surround_index_list = surround_index_func(pg_row_num, pg_column_num, i, j)
      state_update_prompt += f'Agent[{i+0.5}, {j+0.5}]: I am in square[{i+0.5}, {j+0.5}], I can observe {square_item_list}, I can do '
      action_list = []
      for box in square_item_only_box:
        for surround_index in surround_index_list:
          action_list.append(f'move({box}, square{surround_index})')
        if 'target'+box[3:] in square_item_list:
          action_list.append(f'move({box}, target{box[3:]})')
      state_update_prompt += f'{action_list}\n'
  return state_update_prompt

def run_boxnet1_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path):
    print('\n' + '*'*30)
    print(f'BoxNet1, Model_name: {model_name}\n')
    base_save_code_dir = save_input_dir + f'/result_boxnet1_{baseline_method_name}_{model_name}'

    if not os.path.exists(base_save_code_dir):
        os.makedirs(base_save_code_dir)

    lifted_ratio_list = []
    total_sample_num = 0
    total_correct_num = 0
    dataset_input_dir = './dataset_gather/BoxNet1_dataset'

    for pg_row_num, pg_column_num in [(1, 2), (2, 2), (2, 4)]:
        for iteration_num in range(10):
            total_sample_num += 1

            print('-------###-------###-------###-------')
            print(f'Row num is: {pg_row_num}, Column num is: {pg_column_num}, Iteration num is: {iteration_num}\n\n')

            save_code_dir = os.path.join(base_save_code_dir, f"{pg_row_num}_{pg_column_num}_{iteration_num}/")
            if not os.path.exists(save_code_dir):
                os.makedirs(save_code_dir)

            with open(
                    dataset_input_dir + f'/env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/pg_state{iteration_num}.json',
                    'r') as file:
                pg_dict = json.load(file)
            file.close()

            pg_dict_initial = copy.deepcopy(pg_dict)
            prompt = create_prompt(pg_row_num, pg_column_num, pg_dict)
            question = prompt

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

                    response = result.stdout
                    errors = result.stderr
                except Exception as e:
                    pass

            extracted_text_1, _ = extract_and_check(response)
            if extracted_text_1 == '':
                extracted_text_1 = extract_equation_with_GPT4(response)

            extracted_text_2, _ = extract_and_check(original_response)
            if extracted_text_2 == '':
                extracted_text_2 = extract_equation_with_GPT4(original_response)

            remaining_box_dict_1, success_failure_1 = score_in_training_set(pg_dict, extracted_text_1)
            remaining_box_dict_2, success_failure_2 = score_in_training_set(pg_dict, extracted_text_2)

            boxes_all_list = [item for items in pg_dict_initial.values() for item in items if item.startswith('box')]
            boxes_remaining_list_1 = [item for items in remaining_box_dict_1.values() for item in items if
                                    item.startswith('box')]
            lifted_ratio_1 = (len(boxes_all_list) - len(boxes_remaining_list_1)) / len(boxes_all_list)
            boxes_remaining_list_2 = [item for items in remaining_box_dict_2.values() for item in items if
                                    item.startswith('box')]
            lifted_ratio_2 = (len(boxes_all_list) - len(boxes_remaining_list_2)) / len(boxes_all_list)

            lifted_ratio_list.append(max(lifted_ratio_1, lifted_ratio_2))

            print(f"\nResponse 1: {extracted_text_1}")
            print(f"Response 2: {extracted_text_2}")
            print('pg_dict_initial:', pg_dict_initial)
            print(f'Initial boxes: {boxes_all_list}')
            print(f"Remaining boxes 1: {boxes_remaining_list_1}")
            print(f"Remaining boxes 2: {boxes_remaining_list_2}")
            print(f"Lifted ratio: {max(lifted_ratio_1, lifted_ratio_2)}")

            with open(save_code_dir + "/Lifted_ratio.txt", "w") as f:
                f.write(str(max(lifted_ratio_1, lifted_ratio_2)))

            with open(save_code_dir + "/response_answer_1.txt", "w") as f:
                f.write(extracted_text_1)

            with open(save_code_dir + "/response_answer_2.txt", "w") as f:
                f.write(extracted_text_2)

            if success_failure_1 == 'success' or success_failure_2 == 'success':
                print('True')
                with open(save_code_dir + f"/success_failure.txt", "w") as f:
                    f.write('True')
                total_correct_num += 1
            else:
                print('False')
                with open(save_code_dir + f"/success_failure.txt", "w") as f:
                    f.write('False')

            print(f'\ntotal_sample_num: {total_sample_num}')
            print(f'total_correct_num: {total_correct_num}')
            print(f'Total lifted ratio: {np.mean(lifted_ratio_list)}')
            print(f'Correct/all: {total_correct_num}/{total_sample_num}')

            with open(base_save_code_dir + f"/total_lifted_ratio.txt", "w") as f:
                f.write(str(np.mean(lifted_ratio_list)))
            with open(base_save_code_dir + f"/lifted_ratio_list.txt", "w") as f:
                f.write(str(lifted_ratio_list))
            with open(base_save_code_dir + f"/total_sample_num.txt", "w") as f:
                f.write(str(total_sample_num))
            with open(base_save_code_dir + f"/total_correct_num.txt", "w") as f:
                f.write(str(total_correct_num))

    run_info = f"CodeSteer, BoxNet1, {baseline_method_name}, {model_name}\n"
    run_info_result = f'total_lifted_ratio: {np.mean(lifted_ratio_list)}, correct/all:{total_correct_num}/{total_sample_num}\n'
    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}.txt")
    log_run_info(log_file_result, run_info + run_info_result)