'''
generate dataset of feature vector
注意：请在dataset文件夹内运行,以确保输出文件在dataset文件夹内
通过正则化匹配每张图片的 "[" "]" 来确定特征向量位置
再通过正则化匹配数字来将str转换为int
内置 assert len(feature) == 512 以确保数据读取无误
'''

##########################################################################
import numpy as np
import time
import os
import re
feature_celeba_path = "./feature_baidu/feature_celeba"
feature_facescrub_path = "./feature_baidu/feature_facescrub"
##########################################################################


def gen_celeba_feature(feature_path, n=-1):
    total_data = []
    names=[]
    op = open("output_feature_celeba.txt","w")

    # 数据预处理：txt -> list
    print("="*20)
    print("数据预处理中。。。")
    tag0 = int(time.time())
    for file in os.listdir(feature_path):
        print(f"正在处理文件{file}")
        file_path = feature_path+"/"+file
        with open(file_path, "r") as f:
            data = f.read()
            ext_idx = [r.span() for r in re.finditer(r'.jpg', data)]
            left_idx = [r.span() for r in re.finditer(r'\[', data)]
            right_idx = [r.span() for r in re.finditer(r'\]', data)]
            for i in range(len(left_idx)):
                # 处理特征向量
                feature = data[left_idx[i][0]:right_idx[i][1]]
                feature = re.findall('[0-9]+', feature)
                feature = list(map(lambda x: int(x, 10), feature))
                assert len(feature) == 512  # 确保数据读取无误
                total_data.append(feature)
                # 处理名字
                if i==0:
                    name = data[0:ext_idx[i][1]]
                    names.append(name)
                else:
                    name = data[right_idx[i-1][1]+1:ext_idx[i][1]]
                    names.append(name)
                # 记录特征向量和名字
                op.write(f"{len(total_data)-1}\t{name}\n")
                    
                if len(total_data)==n:
                    print(f"Total images: {len(total_data)}")
                    return total_data
    print(f"Total images: {len(total_data)}")
    return total_data


def gen_facescrub_feature(feature_path, n=-1):
    total_data = []
    names=[]
    op = open("output_feature_facescrub.txt","w")
    # 数据预处理：txt -> list
    print("="*20)
    print("数据预处理中。。。")

    for file in os.listdir(feature_path):
        print(f"正在处理文件{file}")
        file_path = feature_path+"/"+file
        with open(file_path, "r") as f:
            data = f.read()
            ext_idx = [r.span() for r in re.finditer(r'.jpg', data)]
            left_idx = [r.span() for r in re.finditer(r'\[', data)]
            right_idx = [r.span() for r in re.finditer(r'\]', data)]
            for i in range(len(left_idx)):
                # 处理特征向量
                feature = data[left_idx[i][0]:right_idx[i][1]]
                feature = re.findall('[0-9]+', feature)
                feature = list(map(lambda x: int(x, 10), feature))
                assert len(feature) == 512  # 确保数据读取无误
                total_data.append(feature)
                # 处理名字
                if i==0:
                    name = data[0:ext_idx[i][1]]
                    names.append(name)
                else:
                    name = data[right_idx[i-1][1]+1:ext_idx[i][1]]
                    names.append(name)
                # 记录特征向量和名字
                op.write(f"{len(total_data)-1}\t{name}\n")
                
                if len(total_data)==n:
                    print(f"Total images: {len(total_data)}")
                    return total_data
    print(f"Total images: {len(total_data)}")
    return total_data


def main():
    n = int(input("输入制作数据集大小 (-1则制作所有)："))
    choice = int(
        input("输入1生成feature_celeba.npy\n输入2生成feature_facescrub.npy\n输入3生成两者\n"))
    if choice == 1:
        tag0 = int(time.time())
        np.save(f'feature_celeba.npy', np.array(gen_celeba_feature(feature_celeba_path, n)))
    elif choice == 2:
        tag0 = int(time.time())
        np.save(f'feature_facescrub.npy', np.array(gen_facescrub_feature(feature_facescrub_path, n)))
    elif choice == 3:
        tag0 = int(time.time())
        np.save(f'feature_celeba.npy', np.array(gen_celeba_feature(feature_celeba_path, n)))
        np.save(f'feature_facescrub.npy', np.array(gen_facescrub_feature(feature_facescrub_path, n)))
    else:
        raise("ERROR")
    
    tag1 = int(time.time())
    cost_time = tag1-tag0

    print("数据预处理完成。")
    print(f"Cost time: {cost_time}")
    print("="*20)


main()
