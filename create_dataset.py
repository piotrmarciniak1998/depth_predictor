import shutil
import os
import zipfile


USE_RGB = False
OCCLUSION_BOUNDS = (1, 20)

abs_path = os.path.abspath("./")
input_dataset_path = f"{abs_path}/input_dataset"
dataset_path = f"{abs_path}/dataset"
temp_path = f"{abs_path}/temp"

try:
    shutil.rmtree(f"{abs_path}/dataset")
except FileNotFoundError:
    pass

os.makedirs(f"{dataset_path}/input", exist_ok=True)
os.makedirs(f"{dataset_path}/ground_truth", exist_ok=True)
os.makedirs(f"{temp_path}", exist_ok=True)

files = [f for f in os.listdir(input_dataset_path) if os.path.isfile(os.path.join(input_dataset_path, f))]

for file in files:
    with zipfile.ZipFile(f"{input_dataset_path}/{file}", 'r') as zip_ref:
        zip_ref.extractall(f"{temp_path}")

last_index = 0
global_index = 0
for root, dirs, files in os.walk(temp_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_info = file.rstrip(".png").split("_")
        file_index = int(file_info[0])
        file_type = file_info[1]
        file_content = file_info[2]
        file_occlusion = int(file_info[3])

        if not OCCLUSION_BOUNDS[0] <= file_occlusion <= OCCLUSION_BOUNDS[1]:
            continue
        elif not os.path.isfile(file_path):
            continue
        elif file_type == "rgb" and not USE_RGB:
            continue

        new_file_name = f"{global_index}_{file_type}_{file_occlusion}.png"

        if file_content == "u":
            new_file_path = f"{dataset_path}/input/"
        elif file_content == "o":
            new_file_path = f"{dataset_path}/ground_truth/"
        else:
            new_file_path = None

        shutil.move(file_path, new_file_path)
        os.rename(f"{new_file_path}/{file}", f"{new_file_path}/{new_file_name}")

        if last_index != file_index:
            global_index += 1
            last_index = file_index

try:
    shutil.rmtree(f"{abs_path}/temp")
except FileNotFoundError:
    pass
