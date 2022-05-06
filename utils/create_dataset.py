import shutil
import cv2
import os
import zipfile
import numpy as np


CREATE_TEMP = True
DELETE_TEMP = True
CREATE_RGB = False
CREATE_CIRCLE = True
CIRCLE_SIZE = 125
OCCLUSION_BOUNDS = [(1, 20), (1, 40), (1, 60), (1, 80), (1, 99)]

abs_path = os.path.abspath("./")
input_dataset_path = f"{abs_path}/input_dataset"
temp_path = f"{abs_path}/temp"

if CREATE_TEMP:
    try:
        shutil.rmtree(temp_path)
    except FileNotFoundError:
        pass

    os.makedirs(temp_path, exist_ok=True)

    files = [f for f in os.listdir(input_dataset_path) if os.path.isfile(os.path.join(input_dataset_path, f))]

    for i, file in enumerate(files):
        with zipfile.ZipFile(f"{input_dataset_path}/{file}", 'r') as zip_ref:
            print(f"[{i + 1}/{len(files)}] unzipping {file}.")
            zip_ref.extractall(temp_path)

for occlusion_bound in OCCLUSION_BOUNDS:
    print(f"Started with occlusion bound set as {occlusion_bound}.")
    dataset_path = f"{abs_path}/dataset/{occlusion_bound[0]}_{occlusion_bound[1]}"

    try:
        shutil.rmtree(dataset_path)
    except FileNotFoundError:
        pass

    os.makedirs(f"{dataset_path}/input", exist_ok=True)
    os.makedirs(f"{dataset_path}/ground_truth", exist_ok=True)

    if CREATE_CIRCLE:
        try:
            shutil.rmtree(f"{dataset_path}_circle")
        except FileNotFoundError:
            pass

        os.makedirs(f"{dataset_path}_circle/input", exist_ok=True)
        os.makedirs(f"{dataset_path}_circle/ground_truth", exist_ok=True)

    last_index = 0
    global_index = 0
    all_files = 1

    for root, dirs, files in os.walk(temp_path):
        for i, file in enumerate(files):
            file_path = os.path.join(root, file)
            file_info = file.rstrip(".png").split("_")
            file_index = int(file_info[0])
            file_type = file_info[1]
            file_content = file_info[2]
            file_occlusion = int(file_info[3])
            file_camera_x = int(file_info[4])
            file_camera_y = int(file_info[5])

            print(f"\t[{i + 1}/{len(files)}] \t{file_path}")

            if not occlusion_bound[0] <= file_occlusion <= occlusion_bound[1]:
                continue
            elif not os.path.isfile(file_path):
                continue
            elif file_type == "rgb" and not CREATE_RGB:
                continue

            if last_index != file_index:
                global_index += 1
                last_index = file_index

            new_file_name = f"{global_index}_{file_type}_{file_occlusion}.png"

            if file_content == "o":
                new_file_path = f"{dataset_path}/input/"

            elif file_content == "u":
                new_file_path = f"{dataset_path}/ground_truth/"

            else:
                new_file_path = None

            shutil.copy(file_path, new_file_path)
            os.rename(f"{new_file_path}/{file}", f"{new_file_path}/{new_file_name}")

            if CREATE_CIRCLE:
                if file_content == "o":
                    new_file_path = f"{dataset_path}_circle/input/"
                    shutil.copy(file_path, new_file_path)
                    os.rename(f"{new_file_path}/{file}", f"{new_file_path}/{new_file_name}")

                elif file_content == "u":
                    new_file_path = f"{dataset_path}_circle/ground_truth/"
                    img_u = cv2.imread(file_path)
                    img_o = cv2.imread(file_path.replace("_u_", "_o_"))
                    mask = np.zeros(shape=img_o.shape, dtype=img_o.dtype)
                    cv2.circle(img=mask,
                               center=(file_camera_x, file_camera_y),
                               radius=CIRCLE_SIZE,
                               color=(255, 255, 255),
                               thickness=-1)
                    img_1 = cv2.bitwise_and(img_u, mask)
                    mask_inv = cv2.bitwise_not(mask)
                    img_2 = cv2.bitwise_and(img_o, mask_inv)
                    img = cv2.bitwise_or(img_1, img_2)
                    cv2.imwrite(f"{new_file_path}/{new_file_name}", img)

        all_files += len(files)

    print(f"Created {global_index} pairs of images. Total number of pairs is {int(round(all_files / 2, 0))}.")

if DELETE_TEMP:
    print("Deleting temporary directory.")
    try:
        shutil.rmtree(f"{abs_path}/temp")
    except FileNotFoundError:
        pass

print("Done.")
