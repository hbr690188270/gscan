import json 

# with open('./target_lengths_run_3/target_lengths_eos_distribution.json', 'r', encoding = 'utf-8') as f:
#     filtered_item_list = json.load(fp = f)

# with open("./target_lengths_run_3/target_lengths_filtered_res.txt",'w', encoding = 'utf-8') as f:
#     for item_dict in filtered_item_list:
#         input_x = item_dict['input_x']

import json
import matplotlib.pyplot as plt
import pickle
import numpy as np
with open('./target_lengths_run_3/target_lengths_eos_distribution.json', 'r', encoding = 'utf-8') as f:
    all_item_list = json.load(fp = f)

length_to_distribution = {}
for item_dict in all_item_list:
    eos_dis = item_dict['eos_distribution']
    second_dis = item_dict['second_distribution']
    target = item_dict['target']
    target_len = len(target)
    # print(eos_dis)
    # print(second_dis)
    # pause = input("???")
    if target_len not in length_to_distribution:
        length_to_distribution[target_len] = [(eos_dis, second_dis)]
    else:
        length_to_distribution[target_len].append((eos_dis, second_dis))

print(length_to_distribution.keys())
with open("eos_distribution.pkl",'wb') as f:
    pickle.dump(length_to_distribution, f)

