import os

PATH = "/home/only/workspace/wang/deeplearning/segment/BiseNetV2/datasets/export_data_list/image"
txt_path = "/home/only/workspace/wang/deeplearning/segment/BiseNetV2/datasets/export_data_list"
img_list = os.listdir(PATH)

with open(os.path.join(txt_path, "img_list.txt"), "w") as f:
    for img_name in img_list:
        full_path = os.path.join(PATH, img_name)
        f.write(full_path + "\n")