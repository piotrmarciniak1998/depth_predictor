import torch
import os


abs_path = os.path.abspath("../")
temp_path = f"{abs_path}/dataset/1_20"
for root, dirs, files in os.walk(temp_path):
    for file in files:
        file_info = file.rstrip(".png").split("_")
        file_type = file_info[1]
        if file_type == 'rgb':
            os.remove(f'{root}/{file}')