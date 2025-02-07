import json
import re
import pandas as pd
import os
import subprocess
import sys
from openai import OpenAI
from generation_models import message_construct_func, message_construct_llama_func, GPT_response, count_total_tokens, extract_code, extract_and_check, LLM_answer_code_checker, save_file_func, paraphrase_with_GPT4, log_run_info
import math
import json
from typing import List, Tuple, Dict
import time
import numpy as np
import ast
from prompt import *
from argparse import ArgumentParser
from symbolic_code_check import analyze_computational_approach, analyze_code_and_explain
import random
from LLaMA_Factory.src.llamafactory.chat.chat_model import run_response

def extract_equation_with_GPT4_game24(response):
    prompt = 'Your task is to extract the equation from the given answer by another LLM:\n' \
             'Note that the equation should include four numbers be in the form like ((11 * 8) + 8) / 4, ((3 * 5) - 12) * 8, ((7 - 4) * 11) - 9 = 24,' \
             '((6 / 3) * 7) + 10, (37 - (29 - 16)), ((19 + 18) - 13) = 24\n No other symbols or texts should be included. If included, remove it.' \
             'Here is the reponse, return your answer with the format <<<equation>>>, like <<<((7 - 4) * 11) - 9 = 24>>>. ' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list=[prompt + response], response_total_list=[], logprobs=False)
    return extract_equation

def validate_equation(number_list, extracted_text):
    if "=" not in extracted_text:
        extracted_text = extracted_text + " = 24"

    # Split the extracted_text into left and right parts
    equation_part_list = extracted_text.split("=")
    if len(equation_part_list) == 2:
        left_side = equation_part_list[0]
        right_side = equation_part_list[1]
    elif len(equation_part_list) > 2:
        left_side = equation_part_list[0]
        right_side = equation_part_list[-1]
    #left_side, right_side = extracted_text.split("=")

    # Evaluate the right side
    try:
        right_value = eval(right_side.strip())
        if right_value != 24:
            return False
    except:
        return False

    # Prepare the number list as a multiset
    number_multiset = sorted(number_list)

    # Extract numbers and operators from the left side using regex
    # Supports LaTeX expressions as well
    left_side_numbers = re.findall(r'\d+', left_side)
    left_side_numbers = list(map(int, left_side_numbers))
    left_side_numbers_sorted = sorted(left_side_numbers)

    if number_multiset != left_side_numbers_sorted:
        return False

    # Replace LaTeX syntax with Python syntax for evaluation
    left_side = left_side.replace(r'\times', '*').replace(r'\div', '/')
    left_side = re.sub(r'\\frac{(\d+)}{(\d+)}', r'(\1/\2)', left_side)

    # Evaluate the left side
    try:
        left_value = eval(left_side.strip())
        if left_value != 24:
            return False
    except:
        return False

    return True

def run_game24(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM):
    print('\n' + '*'*30)
    print(f'Game24, Model_name: {model_name}, CodeSteer\n')

    dataset_base_dir = dataset_input_dir + f'/game24_dataset/24'
    dataset_csv = dataset_base_dir + f'/24.csv'
    df = pd.read_csv(dataset_csv)

    question_prompt = f'Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each number should be used only once but each number has to be used in the equation. ' \
                      f'Input: 9 10 11 13, Answer: ((10-9)*(11+13)) = 24 Input: 4 10 10 11, Answer: ((4*11)-(10+10)) = 24 Input: 5 6 13 13, Answer: ((5-(13/13))*6)' \
                      f'Input: 2 6 6 7, Answer: ((6+(6*7))/2) Input: 2 6 10 18, Answer: (2-(6-(10+18)))'

    base_save_code_dir = save_input_dir + f'/result_game24_{CodeSteer_LLM}_{model_name}_MTD_{max_tree_depth}_CodeSteer_1'
    if not os.path.exists(base_save_code_dir):
        os.makedirs(base_save_code_dir)

    total_sample_num = 0
    total_correct_num = 0

    for j in range(5, 6):
        #for i in range(j, min(len(df), 1500), 10):
        for i in range(j, min(len(df), 1500), 50):
            total_sample_num += 1
            number_list = list(map(int, df['Puzzles'][i].split()))

            save_code_dir = os.path.join(base_save_code_dir, f'sample_{i}')
            if not os.path.exists(save_code_dir):
                os.makedirs(save_code_dir)

            print(f'\nSample num: {i}\nNumbers: {number_list}\n')

            question = f'{question_prompt}'
            question += f'Input: '
            for number in number_list:
                question += f'{number} '
            question += f'Answer:\nOutput final answer with the format <<<answer>>>.'

            ## Todo, consideration of generation answer token length limit

            response_list = []; CodeSteer_output_prompt_guidance_list = [];
            CodeSteer_input_prompt_list = [code_text_choice_prompt + question]; CodeSteer_input_prompt_training_list = [code_text_choice_prompt + question]

            ############ Starting first guidance ############
            #starting_prompt_choice, perplexity_score, response_text_tokens, logprobs = GPT_response("", code_text_choice_prompt + question, model_name=model_name, code_interpreter=False,
            #                             user_prompt_list=[code_text_choice_prompt + question], response_total_list=[], logprobs=True)
            messages = message_construct_llama_func([code_text_choice_prompt + question], [])
            starting_prompt_choice = run_response(messages, args_path)

            print(f'Starting prompt choice: {starting_prompt_choice}')
            user_prompt_list = [starting_prompt_choice + question]
            CodeSteer_output_prompt_guidance_list.append(starting_prompt_choice)
            response = GPT_response('', user_prompt_list[0], model_name=model_name, code_interpreter=False,
                                         user_prompt_list=user_prompt_list, response_total_list=response_list, logprobs=False)
            response_list.append(response)
            #print(f'\nResponse_0: {response}\n')

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
                #response_text, perplexity_score, response_text_tokens, logprobs = GPT_response("", '', model_name=model_name, code_interpreter=False,
                #                             user_prompt_list=CodeSteer_input_prompt_list, response_total_list=CodeSteer_output_prompt_guidance_list, logprobs=True)
                #matches = re.findall(r'<<<(.*?)>>>', response_text, re.DOTALL)
                #guidance_prompt = matches[-1] if matches else response_text

                messages = message_construct_llama_func(CodeSteer_input_prompt_training_list, CodeSteer_output_prompt_guidance_list)
                guidance_prompt = run_response(messages, args_path)

                print(f'\nGuidance prompt_{tree_depth+1}: {guidance_prompt}\n')

                CodeSteer_output_prompt_guidance_list.append(guidance_prompt)
                if '<<<Code>>>' in guidance_prompt:
                    guidance_prompt = with_COT_code_output_prompt
                elif '<<<Text>>>' in guidance_prompt:
                    guidance_prompt = text_output_prompt
                elif '<<<Return Answer>>>' in guidance_prompt or 'Return Answer' in guidance_prompt or '<<<Terminate>>>' in guidance_prompt or 'Terminate' in guidance_prompt:
                    break
                user_prompt_list.append(guidance_prompt)

                response = GPT_response('', user_prompt_list[0], model_name=model_name, code_interpreter=False,
                                             user_prompt_list=user_prompt_list, response_total_list=response_list, logprobs=False)

                response_list.append(response)
                #print(f'\nResponse_{tree_depth}: {response}\n')
            save_file_func(save_code_dir, response_list, user_prompt_list, question, CodeSteer_input_prompt_list, CodeSteer_input_prompt_training_list, CodeSteer_output_prompt_guidance_list)

            response = response_list[-1]
            print(f'\nSample num: {i}\nNumbers: {number_list}')

            code_block_list = extract_code(response)
            if len(code_block_list) > 0:
                with open(save_code_dir + f"/code_evaluate_correctness_0.py", "w") as f:
                    f.write(code_block_list[0])

            # Test the generated code
            if not os.path.exists(save_code_dir + f"/code_evaluate_correctness_0.py"):
                extracted_text, itertools_present = extract_and_check(response)
            else:
                try:
                    result = subprocess.run(
                        ["python3", "-c",
                         f"exec(open('{save_code_dir}/code_evaluate_correctness_0.py').read()); print(result)"],
                        capture_output=True,
                        text=True,
                        timeout=15
                    )
                    output = result.stdout
                except Exception as e:
                    output = ""
                extracted_text, itertools_present = extract_and_check(output)
            if extracted_text == "" or len(extracted_text) < 5 or validate_equation(number_list, extracted_text) == False:
                output = None
                iteration_num_1 = 0
                while output == None and iteration_num_1 < 3:
                    iteration_num_1 += 1
                    output = extract_equation_with_GPT4_game24(response)
                if output == None:
                    output = ''
                extracted_text, _ = extract_and_check(output)

            if validate_equation(number_list, extracted_text) == True:
                print('True')
                total_correct_num += 1
            else:
                print('False')
                print(f'extracted_text: {extracted_text}')
                if extracted_text == "":
                    print(f"response: {response}\n")

            with open(save_code_dir + f"/response_answer.txt", "w") as f:
                f.write(extracted_text)
            with open(save_code_dir + f"/success_failure.txt", "w") as f:
                f.write(str(validate_equation(number_list, extracted_text)))

            print(f'\ntotal_sample_num: {total_sample_num}')
            print(f'total_correct_num: {total_correct_num}\n')

            with open(base_save_code_dir + f"/acc_result_log_{model_name}.txt", "w") as f:
                f.write(f"correct/all:{total_correct_num}/{total_sample_num}\n")

    print(f'\ntotal_sample_num: {total_sample_num}')
    print(f'total_correct_num: {total_correct_num}\n')

    run_info = f"CodeSteer, Game24, {CodeSteer_LLM}, {model_name}, MTD_{max_tree_depth}_CodeSteer_1\n"
    run_info_result = f'correct/all:{total_correct_num}/{total_sample_num}\n'
    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}.txt")
    log_run_info(log_file_result, run_info + run_info_result)