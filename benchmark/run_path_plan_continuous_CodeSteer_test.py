import json
import re
import os
import subprocess
import io
import sys
from openai import OpenAI
from typing_extensions import override
from openai import AssistantEventHandler
from generation_models import message_construct_func, message_construct_llama_func, GPT_response, count_total_tokens, extract_code, extract_and_check, LLM_answer_code_checker, save_file_func, paraphrase_with_GPT4, log_run_info
import copy
import argparse
from datasets import load_dataset
import time
import numpy as np
from prompt import *
from argparse import ArgumentParser
from symbolic_code_check import analyze_computational_approach, analyze_code_and_explain
from LLaMA_Factory.src.llamafactory.chat.chat_model import run_response
import pandas as pd
from post_process_path_plan_continuous_func import check_trajectory

def True_false_func(box_dict, starting_position, ending_position, extracted_waypoint_list_code, index):
    if extracted_waypoint_list_code == '':
        extracted_waypoint_list_code = '[]'
    try:
        waypoints = eval(extracted_waypoint_list_code)
    except:
        waypoints = []
        print('Wrong format of waypoints: ', extracted_waypoint_list_code)

    if waypoints == []:
        print('Extracted_waypoint_list_code: ', extracted_waypoint_list_code)

    try:
        True_false_result, feedback_reason = check_trajectory(box_dict, starting_position, ending_position, waypoints,
                                                                index)
    except:
        True_false_result = False
        print('Error in trajectory checking')
    return True_false_result


def extract_equation_with_GPT4(response):
    prompt = 'Your task is to extract the waypoints of the robot trajectory from the given answer by another LLM:\n' \
             'Note that the waypoints should include a list of (x, y) coordinates like [(-1.3, -1.3), (-0.85, 0.125), (-0.75, -1.1), (0.45, -0.75), (0.45, 1.0), (1.3, 1.3)], ' \
             '[(1.3, -1.3), (0.45, 1.0), (0.45, -1.0), (-0.85, 0.1), (1.3, 1.3)] \n' \
             'Here is the response, return your answer with the format <<<list>>>, like <<<[(1.3, -1.3), (0.65, 0.9), (0.45, 0.0), (1.3, 1.3)]>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no waypoints are answered, then directly answer empty list <<<[]>>>.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def run_path_plan(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM):
    print('\n' + '*'*30)
    print(f'Path Plan, Model_name: {model_name}, CodeSteer\n')
    base_save_code_dir = save_input_dir + f'/result_path_plan_{CodeSteer_LLM}_{model_name}_MTD_{max_tree_depth}_CodeSteer_1'
    if not os.path.exists(base_save_code_dir):
        os.makedirs(base_save_code_dir)

    total_sample_num = 0
    total_correct_num = 0

    walls = [(-1.5, -1.4, -1.5, 1.5), (1.4, 1.5, -1.5, 1.5), (-1.5, 1.5, -1.5, -1.4), (-1.5, 1.5, 1.4, 1.5)]

    # First Environment
    box_dict_1 = {'yellow': (-1, -0.7, -0.25, 0.5), 'red': (0, 0.9, -1, -0.5), 'green': (0.2, 0.7, 0.8, 1.2),
                'blue': (-0.5, 0.5, -0.5, 0.5), 'pink': (-1.2, 0.0, 0.6, 0.9), 'purple': (-1.0, -0.5, -1.3, -0.9), 'orange': (0.6, 1.2, -0.5, 0.5)}
    starting_position_1 = (-1.3, -1.3); ending_position_1 = (1.3, 1.3)
    Instruction_simple_houseworld_1_list = \
        ['Navigate into the green room and go to the yellow room.',
        'Navigate into the yellow room, go to the purple room, and to the red room, and to the green room.',
        'Navigate into the yellow room, go to the purple room, and to the red room, and to the green room. Remember do not enter the blue area all the time.',
        'Navigate into the green room and go to the yellow room.',
        'Visit blue region during your trajectory, but never enter all the other colored boxes.',
        'Navigate into the green room and go to the yellow room. Remember do not enter the pink area all the time.',
        'Navigate into the yellow room and go to the orange room. Remember do not enter the blue, purple, and red areas all the time.',
        'Navigate into the green room and go to the yellow room and go to the red room. Remember do not enter the blue area all the time.',
        'Visit purple, orange, and pink areas, but remember never enter all the other colored areas.',
        'At some point go to the yellow box, and at some point go to the red box, and enter the green box, and always do not enter the blue area.']

    # Second Environment
    box_dict_2 = box_dict_1; starting_position_2 = (1.3, -1.3); ending_position_2 = (1.3, 1.3); Instruction_simple_houseworld_2_list = Instruction_simple_houseworld_1_list

    # Third Environment
    box_dict_3 = {'yellow': (-1, -0.7, -0.25, 0.5), 'red': (0, 0.9, -1, -0.5), 'green': (0.2, 0.7, 0.8, 1.2),
                'blue': (-0.2, 0.2, -0.2, 0.2), 'pink': (-1.2, 0.0, 0.6, 0.9), 'purple': (-1.0, -0.5, -1.3, -0.9), 'orange': (0.6, 1.2, -0.5, 0.5)}
    starting_position_3 = starting_position_1; ending_position_3 = ending_position_1; Instruction_simple_houseworld_3_list = Instruction_simple_houseworld_1_list

    # Forth Environment
    box_dict_4 = box_dict_1; starting_position_4 = (-1.3, 1.3); ending_position_4 = (1.3, 1.3); Instruction_simple_houseworld_4_list = Instruction_simple_houseworld_1_list

    # Fifth Environment
    box_dict_5 = {'red': (-1, -0.7, -0.25, 0.5), 'yellow': (0, 0.9, -1.3, -0.8), 'green': (0.2, 0.7, 0.8, 1.2),
                'blue': (-0.3, 0.3, -0.3, 0.3), 'orange': (-1.2, 0.0, 0.6, 0.9), 'purple': (-1.0, -0.5, -1.3, -0.7), 'pink': (0.6, 1.2, -0.5, 0.5)}
    starting_position_5 = (1.3, -1.3); ending_position_5 = (1.3, 1.3); Instruction_simple_houseworld_5_list = Instruction_simple_houseworld_1_list

    # Sixth Environment
    box_dict_6 = {'red': (-1, -0.7, -0.25, 0.5), 'yellow': (0, 0.9, -1.3, -0.6), 'green': (0.1, 0.9, 0.7, 1.4),
                'blue': (-0.3, 0.3, -0.3, 0.3), 'orange': (-1.2, 0.0, 0.6, 0.9), 'purple': (-1.0, -0.25, -1.3, -0.5),
                'pink': (0.6, 1.2, -0.5, 0.5)}
    starting_position_6 = (1.3, -1.3); ending_position_6 = (1.3, 1.3); Instruction_simple_houseworld_6_list = Instruction_simple_houseworld_1_list

    # Seventh Environment
    box_dict_7 = {'pink': (-1, -0.7, -0.25, 0.5), 'purple': (0, 0.9, -1.3, -0.6), 'green': (0.1, 0.9, 0.7, 1.2),
                'blue': (-0.3, 0.3, -0.3, 0.3), 'orange': (-1.2, 0.0, 0.6, 1.4), 'yellow': (-1.0, -0.25, -1.3, -0.5),
                'red': (0.6, 1.2, -0.5, 0.5)}
    starting_position_7 = (1.1, 1.3); ending_position_7 = (-1.3, 1.1); Instruction_simple_houseworld_7_list = Instruction_simple_houseworld_1_list

    # Eighth Environment
    box_dict_8 = {'pink': (-1.3, -0.9, -0.25, 0.5), 'purple': (0, 0.9, -1.3, -0.7), 'green': (0.1, 0.9, 0.7, 1.2),
                'blue': (-0.55, 0.55, -0.55, 0.55), 'orange': (-1.2, 0.0, 0.7, 1.3),
                'yellow': (-1.0, -0.25, -1.3, -0.7), 'red': (0.7, 1.2, -0.5, 0.5)}
    starting_position_8 = (1.3, 1.3); ending_position_8 = (-1.3, -1.3); Instruction_simple_houseworld_8_list = Instruction_simple_houseworld_1_list

    # Ninth Environment
    box_dict_9 = {'pink': (-1.3, -0.9, -0.25, 0.5), 'purple': (-0.3, 0.6, -1.3, -0.7), 'green': (0.1, 0.9, 0.7, 1.2),
                'blue': (-0.55, 1.4, -0.3, 0.5), 'orange': (-1.2, 0.0, 0.7, 1.3), 'yellow': (-1.3, -0.4, -1.3, -0.7),
                'red': (0.7, 1.2, -1.2, -0.5)}
    starting_position_9 = (1.35, 1.35); ending_position_9 = (-1.35, -1.35); Instruction_simple_houseworld_9_list = Instruction_simple_houseworld_1_list

    # Tenth Environment
    box_dict_10 = {'pink': (-1.3, -0.9, -0.25, 0.5), 'purple': (-0.3, 0.6, -1.3, -0.7), 'green': (0.1, 0.9, 0.7, 1.2),
                'blue': (-0.55, 1.4, -0.3, 0.5), 'orange': (-1.2, 0.0, 0.7, 1.3), 'yellow': (-1.3, -0.4, -1.3, -0.7),
                'red': (0.7, 1.2, -1.2, -0.5)}
    starting_position_10 = (-1.35, 1.35); ending_position_10 = (1.35, -1.35); Instruction_simple_houseworld_10_list = Instruction_simple_houseworld_1_list

    # Eleventh Environment
    box_dict_11 = {'green': (-1.3, -0.8, -0.25, 0.5), 'purple': (-0.3, 0.6, -1.1, -0.5), 'pink': (0.1, 0.9, 0.7, 1.2),
                'blue': (-0.55, 1.4, -0.3, 0.5), 'orange': (-1.2, 0.0, 0.7, 1.3), 'yellow': (-1.3, -0.4, -1.1, -0.5),
                'red': (0.7, 1.2, -1.2, -0.7)}
    starting_position_11 = (1.35, -1.35); ending_position_11 = (-1.35, 1.35); Instruction_simple_houseworld_11_list = Instruction_simple_houseworld_1_list

    # Twelfth Environment
    box_dict_12 = {'green': (-1.3, -0.8, -0.25, 0.5), 'purple': (-0.3, 0.6, -1.1, -0.5), 'pink': (0.1, 0.9, 0.7, 1.2),
                'blue': (-0.55, 1.4, -0.3, 0.5), 'orange': (-1.2, 0.0, 0.7, 1.3), 'yellow': (-1.3, -0.4, -1.1, -0.5),
                'red': (0.7, 1.2, -1.2, -0.7)}
    starting_position_12 = (1.2, 0.9); ending_position_12 = (-1.35, -1.35); Instruction_simple_houseworld_12_list = Instruction_simple_houseworld_1_list

    # Thirteenth Environment
    box_dict_13 = {'green': (-1.3, -0.8, -0.25, 0.5), 'purple': (-0.3, 0.6, -1.1, -0.5), 'pink': (0.1, 0.9, 0.7, 1.2),
                'blue': (-0.55, 1.4, -0.3, 0.5), 'orange': (-1.2, 0.0, 0.7, 1.3), 'yellow': (-1.3, -0.4, -1.1, -0.5),
                'red': (0.7, 1.2, -1.2, -0.7)}
    starting_position_13 = (-0.7, 0.0); ending_position_13 = (1.3, -0.5); Instruction_simple_houseworld_13_list = Instruction_simple_houseworld_1_list

    # Fourteenth Environment
    box_dict_14 = {'pink': (-1.3, -0.9, -0.25, 0.5), 'purple': (0, 0.9, -1.3, -0.7), 'green': (0.1, 0.9, 0.7, 1.2),
                'blue': (-0.55, 0.55, -0.55, 0.55), 'orange': (-1.2, 0.0, 0.7, 1.3),
                'yellow': (-1.0, -0.25, -1.3, -0.7), 'red': (0.7, 1.2, -0.5, 0.5)}
    starting_position_14 = (-0.7, 0.0); ending_position_14 = (-1.3, -1.3); Instruction_simple_houseworld_14_list = Instruction_simple_houseworld_1_list

    box_dict_list = [box_dict_1, box_dict_2, box_dict_3, box_dict_4, box_dict_5, box_dict_6, box_dict_7, box_dict_8, box_dict_9, box_dict_10, box_dict_11, box_dict_12, box_dict_13, box_dict_14]
    starting_position_list = [starting_position_1, starting_position_2, starting_position_3, starting_position_4, starting_position_5, starting_position_6, starting_position_7, starting_position_8, starting_position_9, starting_position_10, starting_position_11, starting_position_12, starting_position_13, starting_position_14]
    ending_position_list = [ending_position_1, ending_position_2, ending_position_3, ending_position_4, ending_position_5, ending_position_6, ending_position_7, ending_position_8, ending_position_9, ending_position_10, ending_position_11, ending_position_12, ending_position_13, ending_position_14]
    Instruction_simple_houseworld_list_gather = [Instruction_simple_houseworld_1_list, Instruction_simple_houseworld_2_list, Instruction_simple_houseworld_3_list,
                                                 Instruction_simple_houseworld_4_list, Instruction_simple_houseworld_5_list, Instruction_simple_houseworld_6_list,
                                                 Instruction_simple_houseworld_7_list, Instruction_simple_houseworld_8_list, Instruction_simple_houseworld_9_list,
                                                 Instruction_simple_houseworld_10_list, Instruction_simple_houseworld_11_list, Instruction_simple_houseworld_12_list,
                                                 Instruction_simple_houseworld_13_list, Instruction_simple_houseworld_14_list]

    for env_index in range(14):
        box_dict = box_dict_list[env_index]; starting_position = starting_position_list[env_index]; ending_position = ending_position_list[env_index]
        Instruction_simple_houseworld_list = Instruction_simple_houseworld_list_gather[env_index]

        num_wall = len(walls)
        Environment_description_houseworld = f'First, there are four black colored walls with position {walls[0]}'
        for i in range(1, num_wall):
          Environment_description_houseworld += f', {walls[i]}'
        Environment_description_houseworld += '. Then the boxes are:'
        for color, box in box_dict.items():
          Environment_description_houseworld += f' a {color} colored box {box};'
        Environment_description_houseworld = Environment_description_houseworld[:-1]
        Environment_description_houseworld += '.'

        for index in range(len(Instruction_simple_houseworld_list)):
            total_sample_num += 1

            Instruction_simple_houseworld = Instruction_simple_houseworld_list[index]
            question=f'''
            I am a mobile robot and I hope you can help me plan the trajectory to fulfill the navigation instruction. I will give you the position and size of each box in the whole environment and the instruction is to enter or avoid the box. Each box is of square shape, I will give you (x_start, x_end, y_start, y_end) to describe the shape of squares/boxes. x_start, x_end denote the boundary of squares/boxes in x coordinates. y_start, y_end denote the boundary of squares/boxes in y coordinates. 
            \n{Environment_description_houseworld}
            \nDuring navigation, the trajectory should always not touch the black wall. The starting position of the car is {starting_position}, the ending position of the car is {ending_position}.
            \nThe instruction is '{Instruction_simple_houseworld}'. The car should fulfill the instruction and in the end stop at the end position. Could you give me the trajectory to realize the above instruction? Return your answer with a list of waypoints [(x0, y0), (x1, y1)...].
            \nNote that the whole trajectory is acquired by directly connecting the near waypoints with straight lines. These straight lines should also fulfill the instruction requirements. Particularly check whether the interpolated straight lines enter into the areas that should avoid.
            \nNote that entering in the area means the trajectory intersects with the box, including the boundary of the box. Hence, the trajectory should not intersect with the boundary of the box if the instruction is to avoid the box.
            '''

            save_code_dir = os.path.join(base_save_code_dir, f'sample_{env_index}_{index}')
            if not os.path.exists(save_code_dir):
                os.makedirs(save_code_dir)

            print(f'\nsample_{env_index}_{index}')
            response_list = [];
            CodeSteer_output_prompt_guidance_list = [];
            CodeSteer_input_prompt_list = [code_text_choice_prompt + question];
            CodeSteer_input_prompt_training_list = [code_text_choice_prompt + question]

            ############ Starting first guidance ############
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
                '''
                response_text, perplexity_score, response_text_tokens, logprobs = GPT_response("", '',
                                                                                               model_name=model_name,
                                                                                               code_interpreter=False,
                                                                                               user_prompt_list=CodeSteer_input_prompt_list,
                                                                                               response_total_list=CodeSteer_output_prompt_guidance_list,
                                                                                               logprobs=True)

                matches = re.findall(r'<<<(.*?)>>>', response_text, re.DOTALL)
                guidance_prompt = matches[-1] if matches else response_text
                '''

                messages = message_construct_llama_func(CodeSteer_input_prompt_training_list,
                                                        CodeSteer_output_prompt_guidance_list)
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


            response = response_list[-1]
            original_response = response

            print(f'\nSample_{env_index}_{index}, model_name_{model_name}:')

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
                        capture_output=True, text=True, timeout=15)
                    if result.stdout == '':
                        result = subprocess.run(
                            ["python3", "-c", f"exec(open('{save_code_dir}/code_1_0.py').read()); print(Waypoints)"],
                            capture_output=True, text=True, timeout=15)
                    if result.stdout == '':
                        result = subprocess.run(
                            ["python3", "-c", f"exec(open('{save_code_dir}/code_1_0.py').read()); print(waypoints)"],
                            capture_output=True, text=True, timeout=15)
                    if result.stdout == '':
                        result = subprocess.run(
                            ["python3", "-c", f"exec(open('{save_code_dir}/code_1_0.py').read()); print(trajectory)"],
                            capture_output=True, text=True, timeout=15)

                    response = result.stdout
                    errors = result.stderr
                except Exception as e:
                    pass

            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4(response)
            extracted_text_1, _ = extract_and_check(output_1)

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4(original_response)
            extracted_text_2, _ = extract_and_check(output_2)

            True_false_result_1 = True_false_func(box_dict, starting_position, ending_position, extracted_text_1, index)
            True_false_result_2 = True_false_func(box_dict, starting_position, ending_position, extracted_text_2, index)

            if original_response != response:
                print(f'New response: {response}')

            print(f'True_false_result from response: {True_false_result_1}')
            print(f'True_false_result from original_response: {True_false_result_2}')
            print(f'extracted_text from response: {extracted_text_1}')
            print(f'extracted_text from original_response: {extracted_text_2}')
            with open(save_code_dir + f"/True_false_result_1.txt", "w") as f:
                f.write(str(True_false_result_1))
            with open(save_code_dir + f"/True_false_result_2.txt", "w") as f:
                f.write(str(True_false_result_2))
            with open(save_code_dir + f"/extracted_answer_1.txt", "w") as f:
                f.write(extracted_text_1)
            with open(save_code_dir + f"/extracted_answer_2.txt", "w") as f:
                f.write(extracted_text_2)
            with open(save_code_dir + f"/output_response.txt", "w") as f:
                f.write(output_1)
            with open(save_code_dir + f"/output_original_response.txt", "w") as f:
                f.write(output_2)

            if True_false_result_1 == False and True_false_result_2 == False:
                print('False')
                with open(save_code_dir + f"/success_failure.txt", "w") as f:
                    f.write('False')
            elif True_false_result_1 == True or True_false_result_2 == True:
                print('True')
                with open(save_code_dir + f"/success_failure.txt", "w") as f:
                    f.write('True')
                total_correct_num += 1

            print(f'\ntotal_sample_num: {total_sample_num}')
            print(f'total_correct_num: {total_correct_num}\n')

    with open(base_save_code_dir + f"/acc_result_log_{model_name}.txt", "w") as f:
        f.write(f"correct/all:{total_correct_num}/{total_sample_num}\n")

    run_info = f"CodeSteer, Path Plan, {CodeSteer_LLM}, {model_name}, MTD_{max_tree_depth}_CodeSteer_1\n"
    run_info_result = f'correct/all:{total_correct_num}/{total_sample_num}\n'
    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}.txt")
    log_run_info(log_file_result, run_info + run_info_result)