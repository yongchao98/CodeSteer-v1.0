from itertools import permutations, product
import csv
import os
from tqdm import trange
import time
import random

def generate_list_of_lists(num_lists, value_num, a, b):
    list_of_lists = []
    for _ in range(num_lists):
        sublist = [random.randint(a, b) for _ in range(value_num)]
        list_of_lists.append(sublist)
    return list_of_lists

def nested_loops(level, max_level, loop_range, current=[], result=[]):
    if level == max_level:
        result.append(current)
        return result
    for value in loop_range:
        nested_loops(level + 1, max_level, loop_range, current + [value], result)
    return result

def solve_int_number(numbers, target_value):
    if not numbers:
        return None

    operations = ['+', '-', '*', '/']

    def evaluate_expression(expr):
        try:
            result = eval(expr)
            return abs(result - target_value) < 1e-6  # to handle floating point precision
        except ZeroDivisionError:
            return False

    def generate_expressions(numbers):
        if len(numbers) == 1:
            yield str(numbers[0])
        else:
            for i in range(1, len(numbers)):
                for left in generate_expressions(numbers[:i]):
                    for right in generate_expressions(numbers[i:]):
                        for op in operations:
                            yield f"({left}{op}{right})"
                            yield f"({right}{op}{left})"  # Both orders to handle non-commutative operations

    start_time = time.time()
    for perm in permutations(numbers):
        for expression in generate_expressions(perm):
            current_time = time.time()
            if evaluate_expression(expression):
                return expression
            if current_time - start_time > 10:
                return None

    return None

# Example usage:

target_value = 24
value_num_list = [60, 120, 240, 480]
for value_num_index in trange(len(value_num_list)):
    value_num = value_num_list[value_num_index]
    num_lists = 1000  # Number of sublists
    a = 1  # Minimum value
    b = 20000  # Maximum value

    numbers_list = generate_list_of_lists(num_lists, value_num, a, b)

    #loop_range = range(30, 35)
    #numbers_list = nested_loops(0, value_num, loop_range)
    sorted_tuples = {tuple(sorted(sublist)) for sublist in numbers_list}

    # Convert set of tuples back to a list of lists
    numbers_list = [list(t) for t in sorted_tuples]
    Puzzles_list = []
    Solution_list = []
    for numbers_index in trange(len(numbers_list)):
        numbers = numbers_list[numbers_index]
        solution = solve_int_number(numbers, target_value)
        if solution is not None:
            Puzzles_list.append(numbers)
            Solution_list.append(solution)
            #print(f"Puzzles: {numbers} Solution: {solution}")
    print(f'length: {len(Puzzles_list)}')

    # Prepare data for CSV
    data = []
    for rank, (puzzle, solution) in enumerate(zip(Puzzles_list, Solution_list), start=1):
        puzzle_str = ' '.join(map(str, puzzle))
        data.append([rank, puzzle_str, solution])

    # Write to CSV file
    dataset_base_dir = f'/Users/yongchaochen/Robot_NLP/MSR_intern/code_text_decision/dataset_gather/data/{target_value}'
    if not os.path.exists(dataset_base_dir):
        os.makedirs(dataset_base_dir)

    with open(dataset_base_dir + f'/{target_value}_{value_num}_number.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Rank', 'Puzzles', 'Solutions'])
        writer.writerows(data)

    print(f"CSV file {target_value}_{value_num}_number.csv has been created successfully.")
