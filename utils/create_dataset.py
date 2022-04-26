import shutil
import os
import zipfile


USE_RGB = False
DELETE_TEMP = True
# OCCLUSION_BOUNDS = [(1, 20), (1, 40), (1, 60), (1, 80), (1, 99)]
OCCLUSION_BOUNDS = [(1, 20)]

for occlusion_bound in OCCLUSION_BOUNDS:
    abs_path = os.path.abspath("./")
    input_dataset_path = f"{abs_path}/input_dataset"
    dataset_path = f"{abs_path}/dataset/{occlusion_bound[0]}_{occlusion_bound[1]}"
    temp_path = f"{abs_path}/temp"

    print(f"Started with occlusion bound set as {occlusion_bound}.")

    try:
        shutil.rmtree(dataset_path)
    except FileNotFoundError:
        pass

    if DELETE_TEMP:
        try:
            shutil.rmtree(f"{abs_path}/temp")
        except FileNotFoundError:
            pass

    os.makedirs(f"{dataset_path}/input", exist_ok=True)
    os.makedirs(f"{dataset_path}/ground_truth", exist_ok=True)
    os.makedirs(f"{temp_path}", exist_ok=True)

    files = [f for f in os.listdir(input_dataset_path) if os.path.isfile(os.path.join(input_dataset_path, f))]

    for i, file in enumerate(files):
        with zipfile.ZipFile(f"{input_dataset_path}/{file}", 'r') as zip_ref:
            print(f"[{i + 1}/{len(files)}] unzipping {file}.")
            zip_ref.extractall(f"{temp_path}")

    last_index = 0
    global_index = 0
    all_files = 1
    for root, dirs, files in os.walk(temp_path):
        if USE_RGB:
            print(f"{global_index} pairs ready. \t"
                  f"{int(round(len(files) / 2, 0))} pairs in current directory to check. \t"
                  f"{int(round(100 * global_index * 4 / all_files, 0))}% of files are used.")
        else:
            print(f"{global_index} pairs ready. \t"
                  f"{int(round(len(files) / 2, 0))} pairs in current directory to check. \t"
                  f"{int(round(100 * global_index * 2 / all_files, 0))}% of files are used.")

        for file in files:
            file_path = os.path.join(root, file)
            file_info = file.rstrip(".png").split("_")
            file_index = int(file_info[0])
            file_type = file_info[1]
            file_content = file_info[2]
            file_occlusion = int(file_info[3])

            if not occlusion_bound[0] <= file_occlusion <= occlusion_bound[1]:
                continue
            elif not os.path.isfile(file_path):
                continue
            elif file_type == "rgb" and not USE_RGB:
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

            shutil.move(file_path, new_file_path)
            os.rename(f"{new_file_path}/{file}", f"{new_file_path}/{new_file_name}")

        all_files += len(files)

    if DELETE_TEMP:
        print("Deleting temporary directory.")
        try:
            shutil.rmtree(f"{abs_path}/temp")
        except FileNotFoundError:
            pass

    print(f"Created {global_index} pairs of images. Total number of pairs is {int(round(all_files / 2, 0))}.")
print("Done.")
