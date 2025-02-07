import json
import re
import pandas as pd
import os
import subprocess
import io
import sys
from openai import OpenAI
from generation_models import message_construct_func, message_construct_llama_func, GPT_response, count_total_tokens, extract_code, extract_and_check, LLM_answer_code_checker, save_file_func, paraphrase_with_GPT4, log_run_info
import copy
import tqdm
import time
import numpy as np
import ast
from prompt import *
from argparse import ArgumentParser
from symbolic_code_check import analyze_computational_approach, analyze_code_and_explain
from LLaMA_Factory.src.llamafactory.chat.chat_model import run_response
import random
import string

def is_equiv_func(question, target_answer, response):
    input_prompt_equiv_func = r'Evaluate whether the answer given by another LLM is correct. ' \
                              r'I will give you the question, the answer from another LLM, and the target answer. ' \
                              r'Then you need to extract the final answer from the tested LLM response and evaluate the correctness of the answer. ' \
                              r'In the end of your response, answer with the list of both extracted answer and Correct/Wrong judgement surrounded by <<<>>>. ' \
                              r'The examples are: <<<[No clear answer, Wrong]>>>, <<<[Yes, Wrong]>>>, <<<[No, Wrong]>>>, <<<[(C) 03/07/2017, Correct]>>>, <<<[(F) 01/24/1947, Wrong]>>>, ' \
                              r'<<<[(D) The motorcyle is the oldest, Wrong]>>>, <<<[(F) The quail is the fourth from the left, Correct]>>>, ' \
                              r'<<<[True, Correct]>>>, <<<[True, Wrong]>>>, <<<[False, Correct]>>>. ' \
                              f'\nMost of the time, the final answer is displayed at the end of the LLM response. However, if there is no clear answer, just answer <<<[No clear answer, Wrong]>>>. ' \
                              f'Do not change or summarize the answer by yourself. Just compare tested LLM answer with the target answer. Especially whether the options are matching! ' \
                              f'For instance, A and D options are not matching, which is judged as wrong! ' \
                              f'Now the question is: {question}; the tested answer is: {response}; the target answer is: {target_answer}. Your evaluation answer: '

    response = GPT_response('', input_prompt_equiv_func, model_name='gpt-4o',
                            code_interpreter=False, user_prompt_list = [input_prompt_equiv_func], response_total_list = [], logprobs = False)
    return response

def run_big_bench_hard(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM):
    print('\n' + '*'*30)
    print(f'Big_bench_hard, Model_name: {model_name}, CodeSteer\n')
    base_save_code_dir_1 = save_input_dir + f'/result_big_bench_hard_{CodeSteer_LLM}_{model_name}_MTD_{max_tree_depth}_CodeSteer_1'

    if not os.path.exists(base_save_code_dir_1):
        os.makedirs(base_save_code_dir_1)

    for env_name in ['date_understanding', 'web_of_lies',
                   'logical_deduction_seven_objects', 'navigate']:
        DATA_PATH = dataset_input_dir + f'/{env_name}.json'
        total_sample_num = 0
        total_correct_num = 0

        base_save_code_dir = base_save_code_dir_1 + f'/{env_name}'
        if not os.path.exists(base_save_code_dir):
            os.makedirs(base_save_code_dir)

        question_json_list = []
        with open(DATA_PATH, 'r') as file:
            for line in file:
                question_json_list.append(json.loads(line))

        print(f'\nEnv_name: {env_name}')
        print(f'Length: ', len(question_json_list[0]['examples']))

        for i in range(0, len(question_json_list[0]['examples']), 4):
                total_sample_num += 1
                print(f'Sample num: {i} in {env_name}, total is: {len(question_json_list[0]["examples"])}')

                save_code_dir = os.path.join(base_save_code_dir, f'sample_{i}')
                if not os.path.exists(save_code_dir):
                    os.makedirs(save_code_dir)

                data = question_json_list[0]['examples'][i]
                question = data['input'] + f'\n' + f'\nOutput final answer with the format <<<answer>>>.'
                target_answer = data['target']

                response_list = [];
                CodeSteer_output_prompt_guidance_list = [];
                CodeSteer_input_prompt_list = [code_text_choice_prompt + question];
                CodeSteer_input_prompt_training_list = [code_text_choice_prompt + question]

                ############ Starting first guidance ############
                '''
                starting_prompt_choice, perplexity_score, response_text_tokens, logprobs = GPT_response("",
                                                                                                        code_text_choice_prompt + question,
                                                                                                        model_name=model_name,
                                                                                                        code_interpreter=False,
                                                                                                        user_prompt_list=[
                                                                                                            code_text_choice_prompt + question],
                                                                                                        response_total_list=[],
                                                                                                        logprobs=True)
                '''
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
                                           user_prompt_list=user_prompt_list, response_total_list=response_list, logprobs=False)

                    response_list.append(response)
                    # print(f'\nResponse_{tree_depth}: {response}\n')
                save_file_func(save_code_dir, response_list, user_prompt_list, question, CodeSteer_input_prompt_list,
                               CodeSteer_input_prompt_training_list, CodeSteer_output_prompt_guidance_list)


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
                            ["python3", save_code_dir + f"/code_1_0.py"], capture_output=True, text=True, timeout=15)
                        # result = subprocess.run(["python3", "-c", f"exec(open('{save_code_dir}/code_1_0.py').read()); print(result)"], capture_output=True, text=True, timeout=15)
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

                output_1 = None;
                iteration_num_1 = 0
                while output_1 == None and iteration_num_1 < 3:
                    iteration_num_1 += 1
                    output_1 = is_equiv_func(question, target_answer, response)
                extracted_text_1, _ = extract_and_check(output_1)

                output_2 = None;
                iteration_num_2 = 0
                while output_2 == None and iteration_num_2 < 3:
                    iteration_num_2 += 1
                    output_2 = is_equiv_func(question, target_answer, original_response)
                extracted_text_2, _ = extract_and_check(output_2)

                True_false_result_1 = 'Correct' in extracted_text_1 or 'correct' in extracted_text_1
                True_false_result_2 = 'Correct' in extracted_text_2 or 'correct' in extracted_text_2

                if original_response != response:
                    print(f'New response: {response}')

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
                else:
                    raise ValueError('Error')

                print(f'total_sample_num: {total_sample_num}')
                print(f'total_correct_num: {total_correct_num}\n')

        with open(base_save_code_dir + f"/acc_result_log_{model_name}.txt", "w") as f:
            f.write(f"correct/all, {env_name}:{total_correct_num}/{total_sample_num}\n")

        print(f'\nEnv_name: {env_name}')
        print(f'total_sample_num: {total_sample_num}')
        print(f'total_correct_num: {total_correct_num}\n')

        run_info = f"CodeSteer, {env_name}, {CodeSteer_LLM}, {model_name}, MTD_{max_tree_depth}_CodeSteer_1\n"
        run_info_result = f'correct/all:{total_correct_num}/{total_sample_num}\n'
        log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}.txt")
        log_run_info(log_file_result, run_info + run_info_result)