from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import copy


class FaceScrub(Dataset):  # ANCHOR 添加了参数seed
    def __init__(self, root, transform=None, target_transform=None, seed=666):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        # ANCHOR - 获取输入
        features = []
        feature_path = os.path.join(self.root, 'feature_facescrub.npy')
        features = np.load(feature_path,allow_pickle=True)
        # ori_feature = copy.deepcopy(data)
        v_min = features.min(axis=0)
        v_max = features.max(axis=0)
        features = (features - v_min) / (v_max - v_min)
        
        # 获取文件名
        labels = []
        with open("./dataset/output_feature_facescrub.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                name = line.split('\t')[-1].split('/')[0]
                hashname = line.split('\t')[-1].split('/')[-1][:-5]
                labels.append(name+'-'+hashname)
                pass
        
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature,label = self.features[index], self.labels[index]
        # img = Image.fromarray(img)

        # if self.transform is not None:
        #     img = self.transform(img)

        if self.target_transform is not None:
            feature = self.target_transform(feature)

        return feature, label


class CelebA(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        features = []
        feature_path = os.path.join(self.root, 'feature_celeba.npy')
        features = np.load(feature_path, allow_pickle=True)
        
        v_min = features.min(axis=0)
        v_max = features.max(axis=0)
        features = (features - v_min) / (v_max - v_min)

        # 获取文件名
        labels = []
        with open("./dataset/output_feature_celeba.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                name = line.split('\t')[-1][:-5]
                labels.append(name)
                pass


        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature, label = self.features[index], self.labels[index]
        # img = Image.fromarray(img)

        # if self.transform is not None:
        #     img = self.transform(img)

        if self.target_transform is not None:
            feature = self.target_transform(feature)

        return feature, label
