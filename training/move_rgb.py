import torch
import os
import shutil

abs_path = os.path.abspath("../")
input_dataset_path = f"{abs_path}/test_dataset/1_80"
temp_path = f"{abs_path}/test_dataset/1_80/input"
os.makedirs(f'{input_dataset_path}/input_rgb', exist_ok=True)
for root, dirs, files in os.walk(temp_path):
    for file in files:
        file_info = file.rstrip(".png").split("_")
        file_type = file_info[1]
        if file_type == 'rgb':
            shutil.move(f'{root}/{file}',f'{input_dataset_path}/input_rgb')