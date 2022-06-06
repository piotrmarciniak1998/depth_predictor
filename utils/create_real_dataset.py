import shutil
import cv2
import os
import zipfile
import numpy as np
from matplotlib import pyplot as plt


CREATE_TEMP = True
DELETE_TEMP = True
CREATE_RGB = True
CREATE_CIRCLE = True
CIRCLE_SIZE = 125
OCCLUSION_BOUNDS = [(1, 20), (1, 40), (1, 60), (1, 80), (1, 99)]
OUTPUT_SIZE = (480, 640)
# OUTPUT_SIZE = (240, 320)


def find_nearest(start, end, place, axis=0):
    if start < end:
        for i in range(start, end):
            if axis == 0:
                if file_img[i][place] != 0:
                    row_index = i
                    col_index = place
                    return row_index, col_index
            if axis == 1:
                if file_img[place][i] != 0:
                    row_index = place
                    col_index = i
                    return row_index, col_index
    else:
        for i in range(start, end, -1):
            if axis == 0:
                if file_img[i][place] != 0:
                    row_index = i
                    col_index = place
                    return row_index, col_index
            if axis == 1:
                if file_img[place][i] != 0:
                    row_index = place
                    col_index = i
                    return row_index, col_index
    return None, None


abs_path = os.path.abspath("../")
input_dataset_path = f"{abs_path}/input_real_dataset"
dataset_path = f"{abs_path}/real_dataset"
dataset_rgb_path = f"{dataset_path}/rgb"
dataset_depth_path = f"{dataset_path}/depth"

os.makedirs(f"{dataset_rgb_path}", exist_ok=True)
os.makedirs(f"{dataset_depth_path}", exist_ok=True)

output_index = 0
for root, dirs, files in os.walk(input_dataset_path):
    for i, file in enumerate(files):
        file_path = os.path.join(root, file)
        file_info = file.rstrip(".png").split("_")
        file_index = int(file_info[0])
        file_type = file_info[1]
        if file_type == "depth":
            file_img = cv2.imread(file_path, cv2.CV_16UC1)
            # print(f"{file}\t{file_index}\t{file_type}\t{file_img.shape}")
            width_start = int((file_img.shape[1] - OUTPUT_SIZE[1]) / 2)
            width_end = int(width_start + OUTPUT_SIZE[1])
            height_start = int((file_img.shape[0] - OUTPUT_SIZE[0]) / 2)
            height_end = int(height_start + OUTPUT_SIZE[0])

            file_img = file_img[height_start:height_end, width_start:width_end]

            #print(np.average(file_img, axis=0))
            rows = file_img.shape[0]
            cols = file_img.shape[1]
            for col in range(cols):
                for row in range(rows):
                    if file_img[row][col] == 0:
                        # file_img[row][col] = sum(file_img[row][:]) / rows
                        data_row = []
                        data_col = []
                        data = find_nearest(row, rows, col, 0)
                        data_row.append(data[0])
                        data_col.append(data[1])
                        data = find_nearest(row, 0, col, 0)
                        data_row.append(data[0])
                        data_col.append(data[1])
                        data = find_nearest(col, cols, row, 1)
                        data_row.append(data[0])
                        data_col.append(data[1])
                        data = find_nearest(col, 0, row, 1)
                        data_row.append(data[0])
                        data_col.append(data[1])

                        best_row = 10000
                        best_col = 10000
                        row_diff = abs(row - 10000)
                        col_diff = abs(col - 10000)
                        best_abs_diff = row_diff + col_diff

                        for index in range(1, len(data_row)):
                            if data_row[index] is None:
                                continue
                            if data_col[index] is None:
                                continue
                            row_diff = abs(row - data_row[index])
                            col_diff = abs(col - data_col[index])
                            abs_diff = row_diff + col_diff
                            # print(abs_diff, best_abs_diff)

                            if abs_diff < best_abs_diff:
                                best_row = data_row[index]
                                best_col = data_col[index]
                                best_abs_diff = abs_diff

                        file_img[row][col] = file_img[best_row][best_col]

            # file_img_2 = np.where(file_img == 0, np.average(file_img, axis=1).astype(np.uint16), file_img)
            # file_img_2 = np.where(file_img == 0, 2**15, file_img)
            brightest = file_img.max()
            # ratio = 255 / brightest
            ratio = 0.015
            file_img = file_img * ratio

            path = f"{dataset_depth_path}/{output_index}_depth.png"
            output_index += 1
            print(path)
            cv2.imwrite(path, file_img)
