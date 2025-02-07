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
import ast
from prompt import *
from argparse import ArgumentParser
from symbolic_code_check import analyze_computational_approach, analyze_code_and_explain
import random
import string

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

def evaluate_response(word, target_letter, llm_response):
    """
    Evaluate the LLM's response for correctness.

    :param word: The test word
    :param target_letter: The letter that was counted
    :param llm_response: The response from the LLM
    :return: A tuple (is_correct, explanation)
    """
    # Extract count and positions from LLM response
    match = re.search(r"Count: (\d+), Positions: \[([\d, ]+)\]", llm_response)
    if not match:
        return False, "Response format is incorrect"

    llm_count = int(match.group(1))
    llm_positions = [int(pos) for pos in match.group(2).split(',')]

    # Calculate correct count and positions
    correct_count = word.count(target_letter)
    correct_positions = [i + 1 for i, letter in enumerate(word) if letter == target_letter]

    if llm_count != correct_count:
        return False, f"Incorrect count. Expected {correct_count}, got {llm_count}"

    if set(llm_positions) != set(correct_positions):
        return False, f"Incorrect positions. Expected {correct_positions}, got {llm_positions}"

    return True, "Correct response"

def extract_equation_with_GPT4(response):
    prompt = 'Your task is to extract the final answer from the given answer by another LLM:\n' \
             'Note that the final answer should follow strictly the format like Count: 5, Positions: [2, 4, 13, 17, 22], Count: 1, Positions: [5], ' \
             'Count: 4, Positions: [3, 11, 18, 24] \n' \
             'Here is the response, return your answer with the format <<<final answer>>>, like <<<Count: 4, Positions: [3, 11, 18, 24]>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer are answered, then directly answer <<<Count: 0, Positions: []>>>.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs=False)
    return extract_equation

def create_prompt(word, target_letter):
    prompt = f"How many '{target_letter}'s are in the word '{word}' and what are their positions? The position in the word counts from 1 not 0.\n"
    prompt += "Surround the answer with <<<content>>>. "
    prompt += f"Please respond in the format: <<<Count: X, Positions: [Y, Z, ...]>>>, such as <<<Count: 2, Positions: [1, 3]>>>.\n" \
              f"Your answer:\n"
    return prompt


def write_words_to_file(words, filename):
    """
    Write a list of words to a JSON file.

    :param words: List of words to write
    :param filename: Name of the file to write to
    """
    with open(filename, 'w') as f:
        json.dump(words, f)

def read_words_from_file(filename):
    """
    Read a list of words from a JSON file.

    :param filename: Name of the file to read from
    :return: List of words
    """
    with open(filename, 'r') as f:
        words = json.load(f)
    #print(f"Words read from {filename}")
    return words

def run_letters_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path):
    print('\n' + '*'*30)
    print(f'Letters, Model_name: {model_name}\n')
    base_save_code_dir = save_input_dir + f'/result_letters_{baseline_method_name}_{model_name}'
    if not os.path.exists(base_save_code_dir):
        os.makedirs(base_save_code_dir)

    total_sample_num = 0
    total_correct_num = 0

    dataset_input_dir = './dataset_gather/Letters'

    for min_length, max_length in [(10, 15), (15, 20), (20, 25)]:
        base_dir = dataset_input_dir + f'/Letters_dataset_min_length_{min_length}_max_length_{max_length}/'

        #for letter in string.ascii_lowercase:
        for i, letter in enumerate(string.ascii_lowercase[::6]):
            for letter_freq in range(1, 6):
                #for index in range(3):
                for index in range(1, 2):
                    total_sample_num += 1
                    base_save_code_dir_2 = os.path.join(base_save_code_dir, f"min_length_{min_length}_max_length_{max_length}/")
                    if not os.path.exists(base_save_code_dir_2):
                        os.makedirs(base_save_code_dir_2)

                    save_code_dir = os.path.join(base_save_code_dir_2, f"{letter}_{letter_freq}_{index}/")
                    if not os.path.exists(save_code_dir):
                        os.makedirs(save_code_dir)

                    saving_dir = base_dir + f"{letter}_{letter_freq}_{index}/"
                    word = read_words_from_file(saving_dir + 'test_words.json')
                    print('-------###-------###-------###-------')
                    print(
                        f"\nMin_length: {min_length}, Max_length: {max_length}, Letter: {letter}, Letter_freq: {letter_freq}, Test word: {word}")

                    prompt = create_prompt(word, letter)
                    question = prompt

                    system_message = ''
                    if baseline_method_name == '1_only_ques':
                        user_prompt_list = [question]

                    if model_name in ['o1', "o1-preview", 'o1-mini', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo',
                                      "claude-3-sonnet-20240229",
                                      "claude-3-opus-20240229", "claude-3-haiku-20240307"]:
                        response = GPT_response(system_message, user_prompt_list[0], model_name=model_name,
                                                code_interpreter=False,
                                                user_prompt_list=user_prompt_list, response_total_list=[],
                                                logprobs=False)
                    else:
                        messages = message_construct_llama_func(user_prompt_list, [])
                        response = run_response(messages, args_path)
                    response_list = []
                    response_list.append(response)

                    save_file_func_baselines(save_code_dir, response_list, user_prompt_list, question, system_message)

                    response = response_list[-1]
                    original_response = response

                    print('-------###-------###-------###-------')
                    print(
                        f"\nMin_length: {min_length}, Max_length: {max_length}, Letter: {letter}, Letter_freq: {letter_freq}, Test word: {word}")

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
                                    ["python3", "-c",
                                     f"exec(open('{save_code_dir}/code_1_0.py').read()); print(Waypoints)"],
                                    capture_output=True, text=True, timeout=15)
                            if result.stdout == '':
                                result = subprocess.run(
                                    ["python3", "-c",
                                     f"exec(open('{save_code_dir}/code_1_0.py').read()); print(waypoints)"],
                                    capture_output=True, text=True, timeout=15)
                            if result.stdout == '':
                                result = subprocess.run(
                                    ["python3", "-c",
                                     f"exec(open('{save_code_dir}/code_1_0.py').read()); print(trajectory)"],
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
                    output_2 = None;
                    iteration_num_2 = 0
                    while output_2 == None and iteration_num_2 < 3:
                        iteration_num_2 += 1
                        output_2 = extract_equation_with_GPT4(original_response)

                    extracted_position_count_str_1, _ = extract_and_check(output_1)
                    is_correct_1, explanation_1 = evaluate_response(word, letter, extracted_position_count_str_1)
                    extracted_position_count_str_2, _ = extract_and_check(output_2)
                    is_correct_2, explanation_2 = evaluate_response(word, letter, extracted_position_count_str_2)

                    print(f'Position_count from response: {extracted_position_count_str_1}')
                    print(f'Position_count from original response: {extracted_position_count_str_2}')
                    correct_count = word.count(letter)
                    correct_positions = [i + 1 for i, test_letter in enumerate(word) if test_letter == letter]
                    print(f'Correct_count: {correct_count}, Correct_positions: {correct_positions}')

                    with open(save_code_dir + f"/position_count_1.txt", "w") as f:
                        f.write(extracted_position_count_str_1)
                    with open(save_code_dir + f"/position_count_2.txt", "w") as f:
                        f.write(extracted_position_count_str_2)
                    with open(save_code_dir + f"/feedback_1.txt", "w") as f:
                        f.write(explanation_1)
                    with open(save_code_dir + f"/feedback_2.txt", "w") as f:
                        f.write(explanation_2)

                    if is_correct_1 == False and is_correct_2 == False:
                        print('False')
                        print(f'Feedback_1: {explanation_1}')
                        print(f'Feedback_2: {explanation_2}')
                        print(f'Original response: {original_response}')
                        with open(save_code_dir + f"/success_failure.txt", "w") as f:
                            f.write('False')
                    else:
                        print('True')
                        with open(save_code_dir + f"/success_failure.txt", "w") as f:
                            f.write('True')
                        total_correct_num += 1

                    print(f'\ntotal_sample_num: {total_sample_num}')
                    print(f'total_correct_num: {total_correct_num}\n')

                    with open(base_save_code_dir + f"/acc_result_log_{model_name}.txt", "w") as f:
                        f.write(f"correct/all:{total_correct_num}/{total_sample_num}\n")

    run_info = f"CodeSteer, Letters, {baseline_method_name}, {model_name}\n"
    run_info_result = f'correct/all:{total_correct_num}/{total_sample_num}\n'
    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}.txt")
    log_run_info(log_file_result, run_info + run_info_result)