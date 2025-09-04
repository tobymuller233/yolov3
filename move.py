import os

imagelist = os.listdir("./datasets/SCUT_HEAD_A_B_three/train/images")
targetlist = os.listdir("./datasets/SCUT_HEAD_A_B_three/train/labels")

for image in imagelist:
    name = image.split(".")[0]
    labelname = name + ".txt"
    if not labelname in targetlist:
        os.system(f"rm ./datasets/SCUT_HEAD_A_B_three/train/images/{image}")
    