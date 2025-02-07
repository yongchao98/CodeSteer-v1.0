import json
import re
import pandas as pd
import os
import subprocess
import sys
from openai import OpenAI
from generation_models import message_construct_func, message_construct_llama_func, GPT_response, count_total_tokens, \
    extract_code, extract_and_check, LLM_answer_code_checker, paraphrase_with_GPT4, log_run_info, load_conversation_data
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

### To do, add tasks to import related functions
from Logic_Game_func import ArithmeticPuzzleEvaluator, read_dataset_logical_equation, verify_solution_logical_equation, extract_equation_with_GPT4_logical_equation, extract_equation_with_GPT4_combi_calcu
from Logic_Game_func import validate_solution_eight_queens, read_dataset_eight_queens, extract_equation_with_GPT4_eight_queens

from Logic_Game_func import validate_solution_syn_decom, read_dataset_syn_decom, extract_equation_with_GPT4_syn_decom
from Logic_Game_func import validate_solution_mahjong, read_dataset_mahjong, extract_equation_with_GPT4_mahjong
from Logic_Game_func import validate_solution_stat_counting, read_dataset_stat_counting, extract_equation_with_GPT4_stat_counting
from Logic_Game_func import validate_solution_new_op, read_dataset_new_op, extract_equation_with_GPT4_new_op
from Logic_Game_func import validate_solution_light, read_dataset_light, extract_equation_with_GPT4_light
from Logic_Game_func import validate_solution_reversi, read_dataset_reversi, extract_equation_with_GPT4_reversi
from Logic_Game_func import validate_solution_matrix_trans, read_dataset_matrix_trans, extract_equation_with_GPT4_matrix_trans
from Logic_Game_func import validate_solution_2048, read_dataset_2048, extract_equation_with_GPT4_2048
from Logic_Game_func import validate_solution_pooling, read_dataset_pooling, extract_equation_with_GPT4_pooling
from Logic_Game_func import validate_solution_constrained, read_dataset_constrained, extract_equation_with_GPT4_constrained
from Logic_Game_func import validate_solution_logic_puzzle, read_dataset_logic_puzzle, extract_equation_with_GPT4_logic_puzzle

from Logic_Game_func import read_dataset_pattern_recognition, extract_equation_with_GPT4_pattern_recognition,validate_solution_pattern_recognition
from Logic_Game_func import read_dataset_string_insertion, extract_equation_with_GPT4_string_insertion, validate_solution_string_insertion
from Logic_Game_func import read_dataset_letter_logic_diagram, extract_equation_with_GPT4_letter_logic_diagram, validate_solution_letter_logic_diagram
from Logic_Game_func import read_dataset_string_synthesis, extract_equation_with_GPT4_string_synthesis, validate_solution_string_synthesis
from Logic_Game_func import read_dataset_standard_sudoku, extract_equation_with_GPT4_standard_sudoku, validate_solution_standard_sudoku
from Logic_Game_func import read_dataset_permutations_and_combinations, extract_equation_with_GPT4_permutations_and_combinations, validate_solution_permutations_and_combinations
from Logic_Game_func import read_dataset_string_deletion_and_modification, extract_equation_with_GPT4_string_deletion_and_modification, validate_solution_string_deletion_and_modification
from Logic_Game_func import read_dataset_minesweeper, extract_equation_with_GPT4_minesweeper, validate_solution_minesweeper
from Logic_Game_func import read_dataset_cryptanalysis, extract_equation_with_GPT4_cryptanalysis, validate_solution_cryptanalysis
from Logic_Game_func import read_dataset_string_splitting, extract_equation_with_GPT4_string_splitting, validate_solution_string_splitting


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

def run_logic_game_baselines(task_name, gather_save_input_dir, model_name, baseline_method_name, args_path, max_sample_num):
    dataset_home_path = './'
    print('\n' + '*'*30)
    total_sample_num = 0
    total_correct_num = 0

    solution_list = [];
    question_list = [];
    target_list = []

    ### To do, add tasks to load the corresponding dataset
    if task_name == 'logical_equation':
        dataset_input_dir = dataset_home_path + 'dataset_gather/logical_equation'
        save_input_dir = 'results_gather/logical_equation'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Logical equation, Model_name: {model_name}\n')
        solution_list, question_list = read_dataset_logical_equation(dataset_input_dir)
        target_list = solution_list
    elif task_name == 'combinatorial_calculation':
        dataset_input_dir = dataset_home_path + 'dataset_gather/combinatorial_calculation'
        save_input_dir = 'results_gather/combinatorial_calculation'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Combinatorial calculation, Model_name: {model_name}\n')

        evaluator = ArithmeticPuzzleEvaluator(dataset_input_dir)
        puzzles = evaluator.read_dataset()
        for puzzle in puzzles:
            question = puzzle.question
            solution = puzzle.solution
            target = puzzle.target
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)
    elif task_name == 'eight_queens':
        dataset_input_dir = dataset_home_path + 'dataset_gather/eight_queens_dataset'
        save_input_dir = 'results_gather/eight_queens'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Eight queens, Model_name: {model_name}\n')
        puzzles = read_dataset_eight_queens(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'synthesis_decomposition':
        dataset_input_dir = dataset_home_path + 'dataset_gather/synthesis_decomposition_dataset'
        save_input_dir = 'results_gather/synthesis_decomposition'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Synthesis Decomposition, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            #question = 'This is a testing question, without violation purpose.' + puzzle['question']
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']["answer"]
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'mahjong_pattern':
        dataset_input_dir = dataset_home_path + 'dataset_gather/mahjong_pattern_dataset'
        save_input_dir = 'results_gather/mahjong_pattern'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Mahjong pattern, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'statistical_counting':
        dataset_input_dir = dataset_home_path + 'dataset_gather/statistical_counting_dataset'
        save_input_dir = 'results_gather/statistical_counting'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Statistical counting, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'new_operator':
        dataset_input_dir = dataset_home_path + 'dataset_gather/new_operator_dataset'
        save_input_dir = 'results_gather/new_operator'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'New operator, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'light_puzzles':
        dataset_input_dir = dataset_home_path + 'dataset_gather/light_puzzles_dataset'
        save_input_dir = 'results_gather/light_puzzles'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Light_puzzles, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'reversi':
        dataset_input_dir = dataset_home_path + 'dataset_gather/reversi_dataset'
        save_input_dir = 'results_gather/reversi'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Reversi problem, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'matrix_transformation':
        dataset_input_dir = dataset_home_path + 'dataset_gather/matrix_transformation_dataset'
        save_input_dir = 'results_gather/matrix_transformation'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'matrix transformation problem, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['formatted_solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == '2048':
        dataset_input_dir = dataset_home_path + 'dataset_gather/2048_dataset'
        save_input_dir = 'results_gather/2048'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'2048 problem, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'pooling':
        dataset_input_dir = dataset_home_path + 'dataset_gather/pooling_dataset'
        save_input_dir = 'results_gather/pooling'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Pooling problem, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'constrained_linear_arrangement':
        dataset_input_dir = 'dataset_gather/constrained_linear_arrangement'
        save_input_dir = 'results_gather/constrained_linear_arrangement'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'constrained_linear_arrangement problem, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'logic_puzzle':
        dataset_input_dir = 'dataset_gather/logic_puzzle_dataset'
        save_input_dir = 'results_gather/logic_puzzle'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'logic puzzle problem, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)

    elif task_name == 'pattern_recognition':
        dataset_input_dir = dataset_home_path + 'dataset_gather/pattern_recognition'
        save_input_dir = 'results_gather/pattern_recognition'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Pattern recognition, Model_name: {model_name}\n')
        puzzles = read_dataset_pattern_recognition(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            target_row, target_col = solution_data['row'], solution_data['column']
            solution_list.append([target_row, target_col])
            question_list.append(question)
            target_list = solution_list
    elif task_name == 'string_insertion':
        dataset_input_dir = dataset_home_path + 'dataset_gather/string_insertion'
        save_input_dir = 'results_gather/string_insertion'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'String Insertion, Model_name: {model_name}\n')
        puzzles = read_dataset_string_insertion(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['modified_string']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)
    elif task_name == 'letter_logic_diagram':
        dataset_input_dir = 'dataset_gather/letter_logic_diagram'
        save_input_dir = 'results_gather/letter_logic_diagram'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Letter_logic_diagram, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_letter_logic_diagram(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)
    elif task_name == 'string_synthesis':
        dataset_input_dir = 'dataset_gather/string_synthesis'
        save_input_dir = 'results_gather/string_synthesis'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'String Synthesis, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_string_synthesis(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['final_counts']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)
    elif task_name == 'standard_sudoku':
        dataset_input_dir = 'dataset_gather/standard_sudoku'
        save_input_dir = 'results_gather/standard_sudoku'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Standard Sudoku, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_standard_sudoku(dataset_input_dir)
        question_matrix_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            question_matrix = solution_data['puzzle']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            question_matrix_list.append(question_matrix)
            target_list.append(target)
    elif task_name == 'permutations_and_combinations':
        dataset_input_dir = 'dataset_gather/permutations_and_combinations'
        save_input_dir = 'results_gather/permutations_and_combinations'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'permutations_and_combinations, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_permutations_and_combinations(dataset_input_dir)
        question_constrained_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['correct_solution']
            question_constrained_linear = solution_data['constraints_text']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            question_constrained_list.append(question_constrained_linear)
            target_list.append(target)
    elif task_name == 'string_deletion_and_modification':
        dataset_input_dir = 'dataset_gather/string_deletion_and_modification'
        save_input_dir = 'results_gather/string_deletion_and_modification'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'String Deletion And Modification, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_string_deletion_and_modification(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['final_string']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)
    elif task_name == 'minesweeper':
        dataset_input_dir = 'dataset_gather/minesweeper'
        save_input_dir = 'results_gather/minesweeper'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Minesweeper, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_minesweeper(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['mines']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)
    elif task_name == 'cryptanalysis':
        dataset_input_dir = 'dataset_gather/cryptanalysis'
        save_input_dir = 'results_gather/cryptanalysis'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Cryptanalysis, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_cryptanalysis(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['answer']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)
    elif task_name == 'string_splitting':
        dataset_input_dir = 'dataset_gather/string_splitting'
        save_input_dir = 'results_gather/string_splitting'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'String Splitting, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_string_splitting(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['expected_answer']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)


    base_save_code_dir = save_input_dir + f'/result_{task_name}_{baseline_method_name}_{model_name}'
    if not os.path.exists(base_save_code_dir):
        os.makedirs(base_save_code_dir)

    ### Remain unchanged
    for i in range(0, min(len(solution_list), max_sample_num)):
        question = question_list[i]
        solution = solution_list[i]
        target = target_list[i]
        total_sample_num += 1

        save_code_dir = os.path.join(base_save_code_dir, f"Test_sample_{i}/")
        if not os.path.exists(save_code_dir):
            os.makedirs(save_code_dir)

        print('-------###-------###-------###-------')
        print(f"\nTest_sample_{i}, total: {min(len(solution_list), max_sample_num)}/")

        code_interpreter = False
        system_message = ''
        if baseline_method_name == '1_only_ques':
            user_prompt_list = [question]
        elif baseline_method_name == 'code_interpreter':
            code_interpreter = True
            user_prompt_list = [question]
        elif baseline_method_name == 'AutoGen':
            user_prompt_list = [AutoGen_prompt + question]
        elif baseline_method_name == 'All_code_CoT':
            user_prompt_list = [with_COT_code_output_prompt + question]
        elif baseline_method_name == 'All_text':
            user_prompt_list = [text_output_prompt + question]
        elif baseline_method_name == 'Text_code_summarizer':
            base_save_code_dir_all_text = save_input_dir + f'/result_{task_name}_All_text_{model_name}'
            base_save_code_dir_all_code = save_input_dir + f'/result_{task_name}_All_code_CoT_{model_name}'
            save_code_dir_all_text = os.path.join(base_save_code_dir_all_text,
                                                  f"Test_sample_{i}/")
            save_code_dir_all_code = os.path.join(base_save_code_dir_all_code,
                                                  f"Test_sample_{i}/")

            with open(save_code_dir_all_text + f"conversation_data.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            response_all_text = data['response_list'][-1]

            with open(save_code_dir_all_code + f"conversation_data.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            response_all_code = data['response_list'][-1]

            #response_list_1, user_prompt_list_1, question_1, CodeSteer_input_prompt_list_1, CodeSteer_input_prompt_training_list_1, CodeSteer_output_prompt_guidance_list_1 = \
            #    load_conversation_data(save_code_dir_all_text + f"conversation_data.json")
            #response_all_text = response_list_1[-1]

            #response_list_2, user_prompt_list_2, question_2, CodeSteer_input_prompt_list_2, \
            #CodeSteer_input_prompt_training_list_2, CodeSteer_output_prompt_guidance_list_2 = \
            #    load_conversation_data(save_code_dir_all_code + f"conversation_data.json")
            #response_all_code = response_list_2[-1]

            input_prompt = combined_agent_prompt + '###The input question is: \n' + question + f'\n\n'
            input_prompt = input_prompt + f'\nThe response from Agent_1 is: {response_all_text}'

            if os.path.exists(save_code_dir_all_text + f"/code_1_0.py"):
                with open(save_code_dir_all_text + f"/extracted_answer_1.txt", "r") as f:
                    extracted_all_text = f.read()

                input_prompt = input_prompt + f'\nThe execution result from the Agent_1 code is: {extracted_all_text}'

            input_prompt = input_prompt + f'\n\nThe response from Agent_2 is: {response_all_code}'
            if os.path.exists(save_code_dir_all_code + f"/code_1_0.py"):
                with open(save_code_dir_all_code + f"/extracted_answer_1.txt", "r") as f:
                    extracted_all_code = f.read()

                input_prompt = input_prompt + f'\nThe execution result from the Agent_2 code is: {extracted_all_code}'
            input_prompt = input_prompt + f'\n\nNow you need to analyze the problem based on their answers and output final answer with the required format in the original question. Your analysis and answer:\n'
            user_prompt_list = [input_prompt]

        if model_name in ['o1', "o1-preview", 'o1-mini', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', "claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229",
                             "claude-3-opus-20240229", "claude-3-haiku-20240307", 'open-mixtral-8x7b', "mistral-large-latest", 'DeepSeek-R1']:
            response = GPT_response(system_message, user_prompt_list[0], model_name=model_name, code_interpreter=code_interpreter,
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
                    capture_output=True, text=True, timeout=15)
                if result.stdout == '':
                    result = subprocess.run(
                        ["python3", "-c", f"exec(open('{save_code_dir}/code_1_0.py').read()); print(Answer)"],
                        capture_output=True, text=True, timeout=15)
                if result.stdout == '':
                    result = subprocess.run(
                        ["python3", "-c", f"exec(open('{save_code_dir}/code_1_0.py').read()); print(answer)"],
                        capture_output=True, text=True, timeout=15)

                # if '<<<' in result.stdout and '>>>' in result.stdout:
                response = result.stdout
                errors = result.stderr
            except Exception as e:
                pass

        ### To do, add tasks to extract the solution from the generated response and verify the correctnessm, here we check both response and original_response
        ### Take care the names of variables to be the same as other tasks since we need to save several variables in the following process.
        if task_name == 'logical_equation':
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_logical_equation(response)
            extracted_text_1, _ = extract_and_check(output_1)
            extracted_text_1 = extracted_text_1.strip()
            try:
                solution_1 = eval(extracted_text_1)
            except:
                solution_1 = []

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_logical_equation(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            extracted_text_2 = extracted_text_2.strip()
            try:
                solution_2 = eval(extracted_text_2)
            except:
                solution_2 = []

            constraints = [line.split('. ')[1] for line in question.split('\n')
                           if line.strip() and line[0].isdigit()]

            True_false_result_1 = verify_solution_logical_equation(solution, solution_1, constraints)
            True_false_result_2 = verify_solution_logical_equation(solution, solution_2, constraints)
        elif task_name == 'combinatorial_calculation':
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_combi_calcu(response)
            try:
                match = re.search(r'<<<\[(.*?)\]>>>', output_1)
                solution_1 = match.group(1)
            except:
                solution_1 = ''
            extracted_text_1 = solution_1
            parsed_solution_1 = [elem.strip().strip('"\'') for elem in solution_1.split(',')]
            puzzle = puzzles[i]
            True_false_result_1, message_1 = evaluator.verify_solution(puzzle, parsed_solution_1)
            print('Feedback1: ', message_1)

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_combi_calcu(original_response)
            try:
                match = re.search(r'<<<\[(.*?)\]>>>', output_2)
                solution_2 = match.group(1)
            except:
                solution_2 = ''
            extracted_text_2 = solution_2
            parsed_solution_2 = [elem.strip().strip('"\'') for elem in solution_2.split(',')]
            True_false_result_2, message_2 = evaluator.verify_solution(puzzle, parsed_solution_2)
            print('Feedback2: ', message_2)
        elif task_name == 'eight_queens':
            solution_data = solution_data_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_eight_queens(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_eight_queens(extracted_text_1, solution_data)

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_eight_queens(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_eight_queens(extracted_text_2, solution_data)
        elif task_name == 'synthesis_decomposition':
            solution_data = solution_data_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_syn_decom(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_syn_decom(extracted_text_1, solution_data, solution_data['complexity'])

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_syn_decom(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_syn_decom(extracted_text_2, solution_data, solution_data['complexity'])
        elif task_name == 'mahjong_pattern':
            solution_data = solution_data_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_mahjong(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_mahjong(extracted_text_1, solution_data, solution_data['complexity'])

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_mahjong(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_mahjong(extracted_text_2, solution_data, solution_data['complexity'])
        elif task_name == 'statistical_counting':
            solution_data = solution_data_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_stat_counting(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_stat_counting(extracted_text_1, solution_data, solution_data['complexity'])

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_stat_counting(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_stat_counting(extracted_text_2, solution_data, solution_data['complexity'])
        elif task_name == 'new_operator':
            solution_data = solution_data_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_new_op(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_new_op(extracted_text_1, solution_data, solution_data['complexity'])

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_new_op(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_new_op(extracted_text_2, solution_data, solution_data['complexity'])
        elif task_name == 'light_puzzles':
            solution_data = solution_data_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_light(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_light(extracted_text_1, solution_data, solution_data['complexity'])

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_light(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_light(extracted_text_2, solution_data, solution_data['complexity'])
        elif task_name == 'reversi':
            # import pdb; pdb.set_trace()
            solution_data = solution_data_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_reversi(response.replace('\n', ''))
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_reversi(extracted_text_1, solution_data, solution_data['complexity'])

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_reversi(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_reversi(extracted_text_2, solution_data, solution_data['complexity'])
        elif task_name == 'matrix_transformation':
            # import pdb; pdb.set_trace()
            solution_data = solution_data_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_matrix_trans(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_matrix_trans(extracted_text_1, solution_data, solution_data['complexity'])

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_matrix_trans(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_matrix_trans(extracted_text_2, solution_data, solution_data['complexity'])
        elif task_name == '2048':
            # import pdb; pdb.set_trace()
            solution_data = solution_data_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_2048(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_2048(extracted_text_1, solution_data, solution_data['complexity'])

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_2048(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_2048(extracted_text_2, solution_data, solution_data['complexity'])
        elif task_name == 'pooling':
            # import pdb; pdb.set_trace()
            solution_data = solution_data_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_pooling(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_pooling(extracted_text_1, solution_data, solution_data['complexity'])

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_pooling(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_pooling(extracted_text_2, solution_data, solution_data['complexity'])
        elif task_name == 'constrained_linear_arrangement':
            # import pdb; pdb.set_trace()
            solution_data = solution_data_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_constrained(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_constrained(extracted_text_1, solution_data, solution_data['complexity'])

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_constrained(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_constrained(extracted_text_2, solution_data, solution_data['complexity'])
        elif task_name == 'logic_puzzle':
            # import pdb; pdb.set_trace()
            solution_data = solution_data_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_logic_puzzle(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_logic_puzzle(extracted_text_1, solution_data, solution_data['complexity'])

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_logic_puzzle(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_logic_puzzle(extracted_text_2, solution_data, solution_data['complexity'])

        elif task_name == 'pattern_recognition':
            solution_data = solution_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_pattern_recognition(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_pattern_recognition(extracted_text_1, solution_data)

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_pattern_recognition(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_pattern_recognition(extracted_text_2, solution_data)
        elif task_name == 'string_insertion':
            solution_data = solution_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_string_insertion(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_string_insertion(extracted_text_1, solution_data)

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_string_insertion(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_string_insertion(extracted_text_2, solution_data)
        elif task_name == 'letter_logic_diagram':
            solution_data = solution_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_letter_logic_diagram(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_letter_logic_diagram(extracted_text_1, solution_data)

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_letter_logic_diagram(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_letter_logic_diagram(extracted_text_2, solution_data)
        elif task_name == 'string_synthesis':
            solution_data = solution_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_string_synthesis(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            # print(f'response: {response}')
            True_false_result_1, message_1 = validate_solution_string_synthesis(extracted_text_1, solution_data)

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_string_synthesis(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            # print(f'original_response: {original_response}')
            True_false_result_2, message_2 = validate_solution_string_synthesis(extracted_text_2, solution_data)
        elif task_name == 'standard_sudoku':
            question_data = question_matrix_list[i]
            solution_data = solution_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_standard_sudoku(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            # print(f'response: {response}')
            True_false_result_1, message_1 = validate_solution_standard_sudoku(extracted_text_1, question_data)

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_standard_sudoku(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            # print(f'original_response: {original_response}')
            True_false_result_2, message_2 = validate_solution_standard_sudoku(extracted_text_2, question_data)
        elif task_name == 'permutations_and_combinations':
            question_data = question_constrained_list[i]
            # solution_data = solution_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_permutations_and_combinations(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            # print(f'response: {response}')
            True_false_result_1, message_1 = validate_solution_permutations_and_combinations(extracted_text_1, question_data)

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_permutations_and_combinations(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            True_false_result_2, message_2 = validate_solution_permutations_and_combinations(extracted_text_2, question_data)
        elif task_name == 'string_deletion_and_modification':
            solution_data = solution_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_string_deletion_and_modification(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_string_deletion_and_modification(extracted_text_1, solution_data)

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_string_deletion_and_modification(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            # print(f'original_response: {original_response}')
            True_false_result_2, message_2 = validate_solution_string_deletion_and_modification(extracted_text_2, solution_data)
        elif task_name == 'minesweeper':
            solution_data = solution_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_minesweeper(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_minesweeper(extracted_text_1, solution_data)

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_minesweeper(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            # print(f'original_response: {original_response}')
            True_false_result_2, message_2 = validate_solution_minesweeper(extracted_text_2, solution_data)
        elif task_name == 'cryptanalysis':
            solution_data = solution_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_cryptanalysis(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_cryptanalysis(extracted_text_1, solution_data)

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_cryptanalysis(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            # print(f'original_response: {original_response}')
            True_false_result_2, message_2 = validate_solution_cryptanalysis(extracted_text_2, solution_data)
        elif task_name == 'string_splitting':
            solution_data = solution_list[i]
            output_1 = None;
            iteration_num_1 = 0
            while output_1 == None and iteration_num_1 < 3:
                iteration_num_1 += 1
                output_1 = extract_equation_with_GPT4_string_splitting(response)
            extracted_text_1, _ = extract_and_check(output_1)
            solution_1 = extracted_text_1
            extracted_text_1 = extracted_text_1.strip()
            print(f'extracted_text_1: {extracted_text_1}')
            True_false_result_1, message_1 = validate_solution_string_splitting(extracted_text_1, solution_data)

            output_2 = None;
            iteration_num_2 = 0
            while output_2 == None and iteration_num_2 < 3:
                iteration_num_2 += 1
                output_2 = extract_equation_with_GPT4_string_splitting(original_response)
            extracted_text_2, _ = extract_and_check(output_2)
            solution_2 = extracted_text_2
            extracted_text_2 = extracted_text_2.strip()
            print(f'extracted_text_2: {extracted_text_2}')
            # print(f'original_response: {original_response}')
            True_false_result_2, message_2 = validate_solution_string_splitting(extracted_text_2, solution_data)


        ### Remain unchanged
        print(f'True_false_result from response: {True_false_result_1}')
        print(f'True_false_result from original_response: {True_false_result_2}')
        print(f'target_solution: {solution}')
        print(f'target_answer: {target}')
        print(f'extracted_text from response: {solution_1}')
        print(f'extracted_text from original_response: {solution_2}')
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
            with open(save_code_dir + f"/success_failure.txt", "w") as f:
                f.write('True')
            total_correct_num += 1

        print(f'\ntotal_sample_num: {total_sample_num}')
        print(f'total_correct_num: {total_correct_num}\n')

        with open(base_save_code_dir + f"/acc_result_log_{model_name}.txt", "w") as f:
            f.write(f"correct/all:{total_correct_num}/{total_sample_num}\n")

    run_info = f"CodeSteer, {task_name}, {baseline_method_name}, {model_name}\n"
    run_info_result = f'correct/all:{total_correct_num}/{total_sample_num}\n'
    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}.txt")
    log_run_info(log_file_result, run_info + run_info_result)