# dataset preparing for retinaface model format
import os
from glob import glob
# import shutil
import cv2

def sng_prepare(filename, out_path):
    outTxt = open(os.path.join(out_path, "labels.txt"), "a")
    file_in = open(filename, "r")
    images = file_in.readlines()[3539078:]
    fp = 0.0
    th = 0.8
    for idx, img_path in enumerate(images):
        print(idx, img_path.strip())
        img_path = img_path.strip()
        img = cv2.imread(img_path)
        width = img.shape[1]
        height = img.shape[0]
        txt_path = img_path.replace(".jpeg", ".pb").replace(".jpg", ".pb")
        f = open(txt_path)
        all_lines = f.readlines()
        line = [float(x) for x in all_lines]
        xc = float(line[0]) * width
        yc = float(line[1]) * height

        # w = float(line[2]) * width
        # h = float(line[3]) * height

        ltx = float(line[4]) * width
        lty = float(line[5]) * height

        lbx = float(line[6]) * width
        lby = float(line[7]) * height

        rtx = float(line[8]) * width
        rty = float(line[9]) * height

        rbx = float(line[10]) * width
        rby = float(line[11]) * height

        w = rbx - ltx
        h = rby - lty

        # draw
        # cv2.circle(img, (int(xc), int(yc)), 3, (0, 0, 255), -1)
        # cv2.circle(img, (int(ltx), int(lty)), 3, (255, 0, 0), -1)
        # cv2.circle(img, (int(lbx), int(lby)), 3, (0, 255, 0), -1)
        # cv2.circle(img, (int(rtx), int(rty)), 3, (0, 255, 255), -1)
        # cv2.circle(img, (int(rbx), int(rby)), 3, (255, 255, 0), -1)
        # cv2.rectangle(img, (int(ltx), int(lty)), (int(ltx + w), int(lty + h)), (0, 125, 125), 2)
        # cv2.imwrite(os.path.join(out_path, os.path.basename(img_path)), img)

        # write to file
        outTxt.writelines('#' + " " + img_path + "\n")
        outTxt.writelines(str(ltx) + " " + str(lty) + " " + str(w) + " " + str(h) + " "

                          + str(ltx) + " " + str(lty) + " " + str(fp) + " "   # 1
                          + str(rtx) + " " + str(rty) + " " + str(fp) + " "   # 2
                          + str(xc) + " " + str(yc) + " " + str(fp) + " "     # 3
                          + str(lbx) + " " + str(lby) + " " + str(fp) + " "   # 4
                          + str(rbx) + " " + str(rby) + " " + str(fp) + " " + str(th) + "\n") # 5
    outTxt.close()

# data_folder = ['KZ/merey/result2022']
# folders = []
# for folder in data_folder:
#     folders.append(glob("/data_sda1/" + folder + "/*/*/*"))
# folders = sum(folders, [])
# print("Folders:", len(folders))
#
# outPath = '/uae_data/retina_label'
#
# outTxt = open(os.path.join(outPath, "labels.txt"), "a")
#
# fp = 0.0
# th = 0.8
# count = 0
# for folder in folders:
#     for idx, item in enumerate([x for x in os.listdir(folder) if x.endswith(".pb")]):
#         # try:
#         count += 1
#         img_path = os.path.join(folder, item.replace(".pb", ".jpeg"))
#         print(idx, item, img_path)
#         img = cv2.imread(img_path)
#         width = img.shape[1]
#         height = img.shape[0]
#         f = open(os.path.join(folder, item))
#         all_lines = f.readlines()
#         line = []
#         for l in all_lines:
#             line.append(float(l.strip().split()[0]))
#             # line.append(float(l.strip().split()[1]))
#
#         # print(line)
#         xc = float(line[0]) * width
#         yc = float(line[1]) * height
#
#         # w = float(line[2]) * width
#         # h = float(line[3]) * height
#
#         ltx = float(line[4]) * width
#         lty = float(line[5]) * height
#
#         lbx = float(line[6]) * width
#         lby = float(line[7]) * height
#
#         rtx = float(line[8]) * width
#         rty = float(line[9]) * height
#
#         rbx = float(line[10]) * width
#         rby = float(line[11]) * height
#
#         w = rbx - ltx
#         h = rby - lty
#
#         # draw
#         # cv2.circle(img, (int(xc), int(yc)), 3, (0, 0, 255), -1)
#         # cv2.circle(img, (int(ltx), int(lty)), 3, (255, 0, 0), -1)
#         # cv2.circle(img, (int(lbx), int(lby)), 3, (0, 255, 0), -1)
#         # cv2.circle(img, (int(rtx), int(rty)), 3, (0, 255, 255), -1)
#         # cv2.circle(img, (int(rbx), int(rby)), 3, (255, 255, 0), -1)
#         # cv2.rectangle(img, (int(ltx), int(lty)), (int(ltx + w), int(lty + h)), (0, 125, 125), 2)
#         # cv2.imwrite(os.path.join(outPath, 'test', item.replace(".pb", ".jpg")), img)
#
#         # write file
#         imgName = item.replace(".pb", ".jpeg")
#         outTxt.writelines('#' + " " + img_path + "\n")
#         outTxt.writelines(str(ltx) + " " + str(lty) + " " + str(w) + " " + str(h) + " "
#
#                          + str(ltx) + " " + str(lty) + " " + str(fp) + " "   # 1
#                          + str(rtx) + " " + str(rty) + " " + str(fp) + " "   # 2
#                          + str(xc) + " " + str(yc) + " " + str(fp) + " "     # 3
#                          + str(lbx) + " " + str(lby) + " " + str(fp) + " "   # 4
#                          + str(rbx) + " " + str(rby) + " " + str(fp) + " " + str(th) + "\n") # 5
#         # shutil.copy(img_path, os.path.join(outPath, "images"))
#         # except Exception as e:
#         #     print(item, e)
#
# print("files: ", count)

# def pairwise(iterable):
#     "s -> (s0, s1), (s2, s3), (s4, s5), ..."
#     a = iter(iterable)
#     return zip(a, a)
#
# with open(os.path.join(outPath, "labels_new.txt"), 'r') as infile:
#     lines = []
#     i = 0
#     for x, y in pairwise(infile):
#         f_name = x.rstrip().split()[1]
#         if os.path.exists(f_name):
#             lines.append(x)
#             lines.append(y)
#             if len(lines) % 1000 == 0:
#                 i += 1
#                 with open(os.path.join(outPath, "test" + str(i) + '_new' + ".txt"), 'w') as f:
#                     for l in lines:
#                         f.writelines(l)
#                 lines = []
#                 break
#         else:
#             print("file does not exist", f_name )

# def uae_prepare(filename, out_path):
#     outTxt = open(os.path.join(out_path, "labels.txt"), "w")
#     file_in =  open(filename, "r")
#     images = file_in.readlines()
#     fp = 0.0
#     th = 0.8
#     for img_path in images:
#         img_path = str("/") + img_path.strip()
#         print(img_path)
#         img = cv2.imread(img_path)
#         width = img.shape[1]
#         height = img.shape[0]
#         txt_path = img_path.replace(".jpg", ".pb")
#         f = open(txt_path)
#         all_lines = f.readlines()[0].split()
#
#         line = [float(x) for x in all_lines]
#         print(line)
#
#         xc = float(line[0]) * width
#         yc = float(line[1]) * height
#
#         # w = float(line[2]) * width
#         # h = float(line[3]) * height
#
#         ltx = float(line[4]) * width
#         lty = float(line[5]) * height
#
#         lbx = float(line[6]) * width
#         lby = float(line[7]) * height
#
#         rtx = float(line[8]) * width
#         rty = float(line[9]) * height
#
#         rbx = float(line[10]) * width
#         rby = float(line[11]) * height
#
#         w = rbx - ltx
#         h = rby - lty
#
#         # draw
#         # cv2.circle(img, (int(xc), int(yc)), 3, (0, 0, 255), -1)
#         # cv2.circle(img, (int(ltx), int(lty)), 3, (255, 0, 0), -1)
#         # cv2.circle(img, (int(lbx), int(lby)), 3, (0, 255, 0), -1)
#         # cv2.circle(img, (int(rtx), int(rty)), 3, (0, 255, 255), -1)
#         # cv2.circle(img, (int(rbx), int(rby)), 3, (255, 255, 0), -1)
#         # cv2.rectangle(img, (int(ltx), int(lty)), (int(ltx + w), int(lty + h)), (0, 125, 125), 2)
#         # cv2.imwrite(os.path.join(out_path, os.path.basename(img_path)), img)
#
#         # write to file
#         outTxt.writelines('#' + " " + img_path + "\n")
#         outTxt.writelines(str(ltx) + " " + str(lty) + " " + str(w) + " " + str(h) + " "
#
#                           + str(ltx) + " " + str(lty) + " " + str(fp) + " "   # 1
#                           + str(rtx) + " " + str(rty) + " " + str(fp) + " "   # 2
#                           + str(xc) + " " + str(yc) + " " + str(fp) + " "     # 3
#                           + str(lbx) + " " + str(lby) + " " + str(fp) + " "   # 4
#                           + str(rbx) + " " + str(rby) + " " + str(fp) + " " + str(th) + "\n") # 5
#     outTxt.close()

if __name__ == "__main__":
    file_path = "/mnt_sda1/sng_eu/txt_detector/filenames_all.txt"
    out_path = "/mnt_sda1/sng_eu/retina_label"
    sng_prepare(file_path, out_path)