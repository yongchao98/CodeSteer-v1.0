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

from benchmark.run_Logic_Game_baseline_methods import run_logic_game_baselines
from benchmark.run_game24_baseline_methods import run_game24_baselines
from benchmark.run_path_plan_continuous_baseline_methods import run_path_plan_baselines
from benchmark.run_letters_baseline_methods import run_letters_baselines
from benchmark.run_number_multiply_baseline_methods import run_number_multiply_baselines
from benchmark.run_BoxNet1_baseline_methods import run_boxnet1_baselines
from benchmark.run_BoxLift_baseline_methods import run_boxlift_baselines
from benchmark.run_Blocksworld_baseline_methods import run_blocksworld_baselines
from benchmark.run_big_bench_hard_baseline_methods import run_big_bench_hard_baselines
from benchmark.run_gsm_baseline_methods import run_gsm_baselines
from benchmark.run_math_counting_and_probability_baseline_methods import run_MATH_counting_and_probability_baselines
from benchmark.run_math_geometry_baseline_methods import run_MATH_geometry_baselines

if __name__ == '__main__':
    # gpt-4o, gpt-4o-mini, gpt-3.5-turbo for OpenAi API

    def log_run_info(log_file, run_info):
        with open(log_file, 'a') as f:
            f.write(run_info + "\n")

    model_name = 'gpt-4o'  # gpt-4o, gpt-3.5-turbo, claude-3-sonnet-20240229, o1, o1-preview, gpt-4o, DeepSeek-R1

    args_path = ''
    gather_save_input_dir = 'results_gather'
    baseline_method_name = '1_only_ques' # 1_only_ques, code_interpreter, AutoGen, All_code_CoT, All_text
    max_sample_num = 50
    ### Test baseline methods
    env_name_list1 = ['permutations_and_combinations', 'logical_equation', 'combinatorial_calculation', 'eight_queens', 'pooling', 'reversi', 'light_puzzles', 'new_operator']
    env_name_list2 = ['mahjong_pattern', 'statistical_counting', 'synthesis_decomposition', '2048', 'matrix_transformation', 'pattern_recognition', 'constrained_linear_arrangement', 'string_synthesis']
    env_name_list3 = ['logic_puzzle', 'string_insertion', 'letter_logic_diagram', 'standard_sudoku', 'string_deletion_and_modification', 'string_splitting', 'cryptanalysis']

    base_path = '../CodeSteer_Submission_Code_and_Data/results_gather'
    runtime_list = []
    for task_name in env_name_list1 + env_name_list2 + env_name_list3:
        start_time = time.time()
        run_logic_game_baselines(task_name, gather_save_input_dir, model_name, baseline_method_name, args_path, max_sample_num)
        end_time = time.time()
        runtime = end_time - start_time
        runtime_list.append(runtime/max_sample_num)

        output_path = base_path + f'/Cost_runtime_gather_{model_name}.txt'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'a') as f:
            f.write(f"{runtime/max_sample_num}\n")
        print(f'Mean time cost: {np.mean(runtime_list)}')
        print(f'\nDataset saved to: {output_path}')

    save_input_dir = 'results_gather/MATH_geometry'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    run_MATH_geometry_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)

    save_input_dir = 'results_gather/MATH_c_p'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    run_MATH_counting_and_probability_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)

    save_input_dir = 'results_gather/gsm'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    run_gsm_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)
    
    save_input_dir = 'results_gather/BIG-Bench-Hard'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    run_big_bench_hard_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)

    save_input_dir = 'results_gather/BoxNet1'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    run_boxnet1_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)

    save_input_dir = 'results_gather/blocksworld'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    run_blocksworld_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)

    save_input_dir = 'results_gather/BoxLift'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    run_boxlift_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)

    save_input_dir = 'results_gather/number_multiply'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    run_number_multiply_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)

    save_input_dir = 'results_gather/Letters'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    run_letters_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)

    save_input_dir = 'results_gather/game24'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    run_game24_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)

    save_input_dir = 'results_gather/path_plan'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    run_path_plan_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)