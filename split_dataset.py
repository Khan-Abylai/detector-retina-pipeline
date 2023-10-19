import os
from glob import glob
import shutil
import random

data_folder = '/home/yeleussinova/data_SSD/kz_data/train'
# intxt = open(os.path.join(data_folder, "label.txt"), "r")
# all_lines = [line.strip() for line in intxt]
outPath = '/home/yeleussinova/data_SSD/kz_data/valid'
# # print(all_lines)
# images = os.listdir(data_folder + "/images")
# valid = int(len(images)*0.1)
# print("train: ", len(images)-valid)
# print("valid: ", valid)
# random.shuffle(images)
# random_images = images[:valid]
# valid_ann = []

# for img in random_images:
#     try:
#         shutil.move(data_folder + "/images/" + img, outPath + "/images")
#         img_n = "# " + img
#         if img_n in all_lines:
#             print(img_n)
#             ind = all_lines.index(img_n)
#             valid_ann.append(img_n)
#             valid_ann.append(all_lines[ind+1])
#             all_lines.pop(ind)
#             all_lines.pop(ind+1)
#     except Exception as e:
#         print(img, e)
#
# with open(os.path.join(data_folder, "new_label.txt"), 'w') as fp:
#     for item in all_lines:
#         # write each item on a new line
#         fp.write("%s\n" % item)
#     print('train done')
#
#
# with open(os.path.join(outPath, "label.txt"),  'w') as fp:
#     for item in valid_ann:
#         # write each item on a new line
#         fp.write("%s\n" % item)
#     print('vaild done')

intxt = open(os.path.join(outPath, "label.txt"), "r")
all_lines = [line.strip() for line in intxt]
for idx, item in enumerate(all_lines):
    name = item.split(" ")[-1:]

    if str(name[0]).endswith(".jpeg"):
        with open(os.path.join(outPath, str(name[0]).replace(".jpeg", ".txt")), 'w') as f:
            f.write(all_lines[idx+1])
            f.close

