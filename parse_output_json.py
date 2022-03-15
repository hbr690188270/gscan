import json

file_dir = './target_lengths_run_3/target_lengths_target_lengths_predict_run_3.json'
with open(file_dir, 'r', encoding = 'utf-8') as f:
    item_list = json.load(f)

filtered_item_list = []

with open("./target_lengths_run_3/target_lengths_filtered_res.txt",'w', encoding = 'utf-8') as f:
    for item_dict in item_list:
        target = item_dict['target']
        output_y = item_dict['prediction']
        f.write(' '.join(target) + '\n')
        f.write(' '.join(output_y) + '\n')
        f.write('\n')

