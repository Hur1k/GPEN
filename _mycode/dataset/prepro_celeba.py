# pre-process of Celeba
# 注意：请在dataset文件夹内运行

from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

# 可自定义参数 ####################################################
data_path = "./img_align_celeba"
#################################################################

parser = argparse.ArgumentParser(description='gen_celeba_img_npy')
parser.add_argument('--img-size', type=int, default=64, metavar='')
args = parser.parse_args()
resize_len = args.img_size

labels = []
images = []
counter = 0
total_images = 0

f = open("output_img_celeba.txt","w") # output_img_celeba.txt文件记录了本脚本制作npy所用到的图像

with open("output_feature_celeba.txt") as ori:
    lines = ori.readlines()
    for line in lines:
        line = line.split("\t")
        image_path = data_path +"/" + line[1].strip("\n")
        
        f.write(f"{total_images}\t{image_path}\n")
        
        raw_image = Image.open(image_path)
        raw_image_crop = raw_image.crop((35,70,35+108,70+108)) #根据官方给出的坐标进行裁剪
        # raw_image_crop.save("test.png") 查看图像
        raw_image_resized = raw_image_crop.resize((resize_len, resize_len),Image.ANTIALIAS)
        raw_image_resized_blackwhite = raw_image_resized.convert('L') #转换为灰度图
        image_array = np.array(raw_image_resized_blackwhite)
        images.append(image_array)
        
        total_images+=1

        # if total_images>=num:
        #     break

images = np.array(images)
np.save(f'img_celeba.npy',images)

print("="*20)
print("npy制作完成.")
print(f"图像尺寸为{resize_len}")
print(f"文件保存至 ./")
print(f"Total images: {total_images}")
print("output_img_celeba.txt文件记录了本脚本制作npy所用到的图像")
print("="*20)