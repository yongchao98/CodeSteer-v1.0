import json
import os
import random
import time
import numpy as np
import ast

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

#base_save_code_dir = f'/Users/yongchaochen/Robot_NLP/MSR_intern/code_text_decision/dataset_gather/number_multiply'
base_save_code_dir = '/home/ycchen/code_text_decision/dataset_gather/number_multiply_different_digit_num'

#for digit_num_list in [[1, 1], [2, 2], [1, 2, 2, 4], [4, 4], [3, 4], [1, 2, 4], [3, 3], [1, 3, 4], [5, 5], [6,6], [1, 1, 1, 1, 1, 1, 3]]:
#for digit_num_list in [[4, 4], [8, 8], [12, 12], [16, 16]]:
for digit_num_list in [[2, 2]]:
        dir_digit_name = f'digit'
        for digit_num in digit_num_list:
            dir_digit_name += f'_{digit_num}'

        for i in range(0, 100):
            generated_num_list, target_answer = generated_num_list_func(digit_num_list)
            equation_prompt = f'{generated_num_list[0]}'
            for generated_num in generated_num_list[1:]:
                equation_prompt += f'*{generated_num}'

            question = f'What is the result of ' + equation_prompt + '?'

            # Save the extracted code to a Python file
            save_code_dir_digit_num = os.path.join(base_save_code_dir, f'{dir_digit_name}')
            if not os.path.exists(save_code_dir_digit_num):
                os.makedirs(save_code_dir_digit_num)
            save_code_dir = os.path.join(save_code_dir_digit_num, f'sample_{i}')
            if not os.path.exists(save_code_dir):
                os.makedirs(save_code_dir)

            with open(save_code_dir + f"/target_answer.txt", "w") as f:
                f.write(str(target_answer))

            with open(save_code_dir + f"/input_prompt_1.txt", "w") as f:
                f.write(question)

            with open(save_code_dir + f"/input_value_list.txt", "w") as f:
                f.write(str(generated_num_list))

            value_list_read = read_value_list(save_code_dir + f"/input_value_list.txt")
            answer_read = read_answer(save_code_dir + f"/target_answer.txt")
            print('\n')
            print(value_list_read)
            print(generated_num_list)

            print(answer_read)
            print(target_answer)

