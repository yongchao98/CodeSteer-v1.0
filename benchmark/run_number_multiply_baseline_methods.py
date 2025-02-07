import json
import re
import pandas as pd
import os
import sys
from openai import OpenAI
from generation_models import message_construct_func, message_construct_llama_func, GPT_response, count_total_tokens, extract_code, extract_and_check, LLM_answer_code_checker, save_file_func, paraphrase_with_GPT4, log_run_info
import random
import math
import json
from typing import List, Tuple, Dict
import time
import numpy as np
import ast
from prompt import *
from argparse import ArgumentParser
from symbolic_code_check import analyze_computational_approach, analyze_code_and_explain
import subprocess

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

def format_number_with_commas(number):
    return f"{number:,}"

def extract_equation_with_GPT4(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<number>>>, like <<<43243.4>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<No answer found>>>.\n' \
             'If there is equation in the answer but no final numbers, do not calculate the number by yourself.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs=False)
    return extract_equation

def validate_equation(number_list, extracted_text):
    # Check if extracted_text has both left and right side
    print(f'extracted_text: {extracted_text}')
    print(extracted_text.split("="))

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

def generate_random_integers(digit_num):
    # Calculate the range based on the number of digits
    min_value = 10 ** (digit_num - 1) - 1
    max_value = 10 ** digit_num - 1
    num = random.randint(min_value, max_value) * random.choice([-1, 1])
    while num == 0 or num == 1 or num == -1:
        num = random.randint(min_value, max_value) * random.choice([-1, 1])
    return num

def generated_num_list_func(digit_num_list):
    generated_num_list = [generate_random_integers(digit_num) for digit_num in digit_num_list]
    target_answer = generated_num_list[0]
    for generated_num in generated_num_list[1:]:
        target_answer *= generated_num
    return generated_num_list, target_answer

def read_value_list(file_path):
    with open(file_path, 'r') as f:
        value_list = f.read()
    return ast.literal_eval(value_list)

def read_answer(file_path):
    with open(file_path, 'r') as f:
        answer = f.read()
    return int(answer)

def run_number_multiply_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path):
    print('\n' + '*'*30)
    print(f'Number Multiply, Model_name: {model_name}\n')

    base_save_code_dir = save_input_dir + f'/result_number_multiply_{model_name}_CodeSteer'
    if not os.path.exists(base_save_code_dir):
        os.makedirs(base_save_code_dir)

    dataset_input_dir = './dataset_gather/number_multiply'

    total_sample_num = 0
    total_correct_num = 0

    for digit_num_list in [[1, 2, 4], [1, 3, 4], [1, 2, 2, 4]]:
            dir_digit_name = f'digit'
            for digit_num in digit_num_list:
                dir_digit_name += f'_{digit_num}'

            dataset_base_dir = os.path.join(dataset_input_dir, f'{dir_digit_name}')
            save_code_dir_dir_digit_num = os.path.join(base_save_code_dir, f'{dir_digit_name}')
            if not os.path.exists(save_code_dir_dir_digit_num):
                os.makedirs(save_code_dir_dir_digit_num)

            for i in range(0, 20):
                total_sample_num += 1

                print(f'\n{dir_digit_name}, sample num: {i}')
                dataset_base_dir_sample = os.path.join(dataset_base_dir, f'sample_{i}')
                generated_num_list = read_value_list(dataset_base_dir_sample + f"/input_value_list.txt")
                target_answer = read_answer(dataset_base_dir_sample + f"/target_answer.txt")

                save_code_dir = os.path.join(save_code_dir_dir_digit_num, f'sample_{i}')
                if not os.path.exists(save_code_dir):
                    os.makedirs(save_code_dir)

                equation_prompt = f'{generated_num_list[0]}'
                for generated_num in generated_num_list[1:]:
                    equation_prompt += f'*{generated_num}'

                question = f'What is the result of ' + equation_prompt + '?'

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

                        if result.stdout == '':
                            result = subprocess.run(
                                ["python3", "-c", f"exec(open('{save_code_dir}/code_1_0.py').read()); print(final_result)"],
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

                True_false_result_1 =  str(target_answer) in extracted_text_1 or str(format_number_with_commas(int(target_answer))) in extracted_text_1
                True_false_result_2 =  str(target_answer) in extracted_text_2 or str(format_number_with_commas(int(target_answer))) in extracted_text_2

                #True_false_result_1 =  target_answer in extracted_text_1
                #True_false_result_2 =  target_answer in extracted_text_2

                print(f'True_false_result from response: {True_false_result_1}')
                print(f'True_false_result from original_response: {True_false_result_2}')
                print(f'target_answer: {target_answer}')
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

                if True_false_result_1 == False and True_false_result_2 == False:
                    print('False')
                    with open(save_code_dir + f"/success_failure.txt", "w") as f:
                        f.write('False')
                else:
                    print('True')
                    total_correct_num += 1
                    with open(save_code_dir + f"/success_failure.txt", "w") as f:
                        f.write('True')

                print(f'\ntotal_sample_num: {total_sample_num}')
                print(f'total_correct_num: {total_correct_num}\n')

                with open(base_save_code_dir + f"/acc_result_log_{model_name}.txt", "w") as f:
                    f.write(f"correct/all:{total_correct_num}/{total_sample_num}\n")

    run_info = f"CodeSteer, Number Multiply, {baseline_method_name}, {model_name}\n"
    run_info_result = f'correct/all:{total_correct_num}/{total_sample_num}\n'
    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}.txt")
    log_run_info(log_file_result, run_info + run_info_result)