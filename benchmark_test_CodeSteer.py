import json
import re
import pandas as pd
import os
import subprocess
import sys
from openai import OpenAI
from generation_models import message_construct_func, message_construct_llama_func, GPT_response, count_total_tokens, extract_code, extract_and_check, LLM_answer_code_checker, save_file_func, paraphrase_with_GPT4
import random
import math
import json
from typing import List, Tuple, Dict
import time
import numpy as np
from prompt import *
from argparse import ArgumentParser
from symbolic_code_check import analyze_computational_approach, analyze_code_and_explain

from benchmark.run_BoxLift_CodeSteer_test import run_boxlift
from benchmark.run_game24_CodeSteer_test import run_game24
from benchmark.run_big_bench_hard_CodeSteer_test import run_big_bench_hard
from benchmark.run_gsm_CodeSteer_test import run_gsm
from benchmark.run_path_plan_continuous_CodeSteer_test import run_path_plan
from benchmark.run_letters_CodeSteer_test import run_letters
from benchmark.run_math_geometry_CodeSteer_test import run_MATH_geometry
from benchmark.run_math_counting_and_probability_CodeSteer_test import run_MATH_counting_and_probability
from benchmark.run_number_multiply_CodeSteer_test import run_number_multiply
from benchmark.run_Blocksworld_CodeSteer_test import run_blocksworld
from benchmark.run_BoxNet1_CodeSteer_test import run_boxnet1
from benchmark.run_Logic_Game_CodeSteer_test import run_logic_game

if __name__ == '__main__':
    # gpt-4o, gpt-4o-mini, gpt-3.5-turbo for OpenAi API

    def log_run_info(log_file, run_info):
        with open(log_file, 'a') as f:
            f.write(run_info + "\n")

    model_name = 'gpt-4o'  # gpt-4o, gpt-3.5-turbo, claude-3-sonnet-20240229, o1, o1-preview, gpt-4o, DeepSeek-R1
    CodeSteer_LLM = 'llama3_8B_CodeSteer'  # llama3_8B_CodeSteer, gpt-4o, gpt-3.5-turbo, claude-3-sonnet-20240229, o1, o1-preview, gpt-4o, DeepSeek-R1
    args_path = './llama3_8B_CodeSteer.yaml'
    gather_save_input_dir = 'results_gather'

    ### Test CodeSteer methods
    env_name_list1 = ['permutations_and_combinations', 'logical_equation', 'combinatorial_calculation', 'eight_queens', 'pooling', 'reversi', 'light_puzzles', 'new_operator']
    env_name_list2 = ['mahjong_pattern', 'statistical_counting', 'synthesis_decomposition', '2048', 'matrix_transformation', 'pattern_recognition', 'constrained_linear_arrangement', 'string_synthesis']
    env_name_list3 = ['logic_puzzle', 'string_insertion', 'letter_logic_diagram', 'standard_sudoku', 'string_deletion_and_modification', 'string_splitting', 'cryptanalysis']

    base_path = '../CodeSteer_Submission_Code_and_Data/results_gather'
    runtime_list = []
    for task_name in env_name_list1 + env_name_list2 + env_name_list3:
        # 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', "claude-3-sonnet-20240229"
        for task_name, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM, max_sample_num in [
            (task_name, gather_save_input_dir, model_name, 5, args_path, CodeSteer_LLM, 50)
        ]:
            start_time = time.time()
            run_logic_game(task_name, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM, max_sample_num)
            end_time = time.time()
            runtime = end_time - start_time
            runtime_list.append(runtime / max_sample_num)

            output_path = base_path + f'/Cost_runtime_gather_{CodeSteer_LLM}.txt'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'a') as f:
                f.write(f"{runtime / max_sample_num}\n")
            print(f'Mean time cost: {np.mean(runtime_list)}')
            print(f'\nDataset saved to: {output_path}')

    dataset_input_dir = 'dataset_gather'
    save_input_dir = 'results_gather/game24'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 5, args_path, CodeSteer_LLM)
    ]:
        run_game24(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM)

    dataset_input_dir = 'dataset_gather'
    save_input_dir = 'results_gather/path_plan'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 5, args_path, CodeSteer_LLM)
    ]:
        run_path_plan(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM)
        
    dataset_input_dir = 'dataset_gather/BoxLift_dataset'
    save_input_dir = 'results_gather/BoxLift'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 5, args_path, CodeSteer_LLM)
    ]:
        run_boxlift(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM)

    dataset_input_dir = 'dataset_gather/Blocksworld_dataset'
    save_input_dir = 'results_gather/blocksworld'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 5, args_path, CodeSteer_LLM)
    ]:
        run_blocksworld(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM)

    dataset_input_dir = 'dataset_gather/BoxNet1_dataset'
    save_input_dir = 'results_gather/BoxNet1'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 5, args_path, CodeSteer_LLM)
    ]:
        run_boxnet1(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM)

    dataset_input_dir = 'dataset_gather/number_multiply'
    save_input_dir = 'results_gather/number_multiply'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 5, args_path, CodeSteer_LLM)
    ]:
        run_number_multiply(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM)

    dataset_input_dir = 'dataset_gather/Letters'
    save_input_dir = 'results_gather/Letters'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 5, args_path, CodeSteer_LLM)
    ]:
        run_letters(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path,
                    CodeSteer_LLM)

    dataset_input_dir = 'dataset_gather/BIG-Bench-Hard/bbh'
    save_input_dir = 'results_gather/BIG-Bench-Hard'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 5, args_path, CodeSteer_LLM)
    ]:
        run_big_bench_hard(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM)

    dataset_input_dir = 'dataset_gather'
    save_input_dir = 'results_gather/gsm'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 5, args_path, CodeSteer_LLM)
    ]:
        run_gsm(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM)

    dataset_input_dir = 'dataset_gather/MATH/train/geometry'
    save_input_dir = 'results_gather/MATH_geometry'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 5, args_path, CodeSteer_LLM)
    ]:
        run_MATH_geometry(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM)

    dataset_input_dir = 'dataset_gather/MATH/train/counting_and_probability'
    save_input_dir = 'results_gather/MATH_c_p'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 5, args_path, CodeSteer_LLM)
    ]:
        run_MATH_counting_and_probability(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM)