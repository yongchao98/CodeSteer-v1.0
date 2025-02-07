import random
import re
import requests
import json
import string
import os
import copy
import numpy as np
import shutil
import time

def surround_index_func(row_num, coloum_num, row_index, coloum_index):
  surround_index_list = []
  for i, j in ([row_index-1, coloum_index], [row_index+1, coloum_index], [row_index, coloum_index-1], [row_index, coloum_index+1]):
    if i>=0 and i<=row_num-1 and j>=0 and j<=coloum_num-1 and not (i == row_index and j == coloum_index):
      surround_index_list.append([i+0.5,j+0.5])
  return surround_index_list

def state_update_func(pg_row_num, pg_column_num, pg_dict):
  pg_dict_copy = copy.deepcopy(pg_dict)
  state_update_prompt = ''
  for i in range(pg_row_num):
    for j in range(pg_column_num):
      square_item_list = pg_dict_copy[str(i+0.5)+'_'+str(j+0.5)]
      square_item_only_box = [item for item in square_item_list if item[:3]=='box']
      surround_index_list = surround_index_func(pg_row_num, pg_column_num, i, j)
      state_update_prompt += f'Agent[{i+0.5}, {j+0.5}]: I am in square[{i+0.5}, {j+0.5}], I can observe {square_item_list}, I can do '
      action_list = []
      for box in square_item_only_box:
        for surround_index in surround_index_list:
          action_list.append(f'move({box}, square{surround_index})')
        if 'target'+box[3:] in square_item_list:
          action_list.append(f'move({box}, target{box[3:]})')
      state_update_prompt += f'{action_list}\n'
  return state_update_prompt

def env_create(pg_row_num = 5, pg_column_num = 5, box_num_low_bound = 2, box_num_upper_bound = 2, color_list = ['blue', 'red', 'green', 'purple', 'orange']):
  # pg_dict records the items in each square over steps, here in the initial setting, we randomly assign items into each square
  pg_dict = {}
  for i in range(pg_row_num):
    for j in range(pg_column_num):
      pg_dict[str(i+0.5)+'_'+str(j+0.5)] = []

  for color in color_list:
    box_num = random.randint(box_num_low_bound, box_num_upper_bound)
    for _ in range(box_num):
      N_box = random.randint(0, pg_row_num*pg_column_num - 1)
      a_box = N_box // pg_column_num
      b_box = N_box % pg_column_num
      N_target = random.randint(0, pg_row_num*pg_column_num - 1)
      a_target = N_target // pg_column_num
      b_target = N_target % pg_column_num
      pg_dict[str(a_box+0.5)+'_'+str(b_box+0.5)].append('box_' + color)
      pg_dict[str(a_target+0.5)+'_'+str(b_target+0.5)].append('target_' + color)
  return pg_dict

if __name__ == "__main__":
      repeat_num = 25
      #Saving_path = f'/n/vlassak_lab/Lab/simulation_data/NLP_robotics/experiment/T5/large_model/llama3/ICLR_Code/dataset_gather/BoxNet1_dataset/'
      Saving_path = f'/Users/yongchaochen/Robot_NLP/Codesteer/ICLR_Code/dataset_gather/BoxNet1_dataset_SFT/'
      if not os.path.exists(Saving_path):
        os.makedirs(Saving_path, exist_ok=True)

      #for i ,j in [(1,2), (2,2), (2,4), (4,4), (4,8)]:
      for i, j in [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4)]:
        print(f'Creating dataset for pg_row_num = {i}, pg_column_num = {j}, repeat_num = {repeat_num}')
        if not os.path.exists(Saving_path+f'/env_pg_state_{i}_{j}'):
          os.makedirs(Saving_path+f'/env_pg_state_{i}_{j}', exist_ok=True)
        else:
          shutil.rmtree(Saving_path+f'/env_pg_state_{i}_{j}')
          os.makedirs(Saving_path+f'/env_pg_state_{i}_{j}', exist_ok=True)

        for iteration_num in range(repeat_num):
          # Define the total row and column numbers of the whole playground, and the item number of each colored target and box
          pg_row_num = i; pg_column_num = j; box_num_low_bound = 1; box_num_upper_bound = 1
          # Define the used colors
          #color_list = ['blue', 'red', 'green', 'purple', 'orange']
          color_list = ['blue', 'red', 'green']
          pg_dict = env_create(pg_row_num, pg_column_num, box_num_low_bound, box_num_upper_bound, color_list)
          os.makedirs(Saving_path+f'/env_pg_state_{i}_{j}/pg_state{iteration_num}', exist_ok=True)
          with open(Saving_path+f'/env_pg_state_{i}_{j}/pg_state{iteration_num}/pg_state{iteration_num}.json', 'w') as f:
            json.dump(pg_dict, f)