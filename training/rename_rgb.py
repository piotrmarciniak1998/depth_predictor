

import torch
import os
import shutil

abs_path = os.path.abspath("../")
temp_path = f"{abs_path}/test_dataset/1_80/input_rgb"
for root, dirs, files in os.walk(temp_path):
    for i, file in enumerate(files):
        file_path = os.path.join(root, file)
        file_info = file.rstrip(".png").split("_")
        file_index = int(file_info[0])
        file_type = file_info[1]
        file_occlusion = int(file_info[2])
        new_file_name = f"{file_index}_depth_{file_occlusion}.png"
        os.rename(f"{root}/{file}", f"{root}/{new_file_name}")