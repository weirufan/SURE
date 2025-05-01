# -*- codding: utf-8 -*-
'''
@Author : Yuren
@Dare   : 2022/1/8-10:16 下午
'''

import os
import numpy as np
from PIL import Image


from torch.utils.data import DataLoader, Dataset
from typing import Optional,Callable

import torchvision
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']

# images dataset


class ImagesDataset(Dataset):
    def __init__(
            self,
            root,
            train,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform=target_transform

        self.train = train

        self.paths, self.targets = self._load_data()
        print(f'{len(self.paths)} images found')

    def __len__(self):
        return len(self.paths)

    def _load_data(self):
        image_file = f"{'train' if self.train else 'test'}-images"
        image_path = os.path.join(self.root,image_file)
        name = os.listdir(image_path)
        name.sort(key=lambda x: int(x[:-4]))
        paths = []
        for path in name:
            paths.append(os.path.join(image_path,path))

        label_file = f"{'train' if self.train else 'test'}-labels.txt"
        label_path = os.path.join(self.root,label_file)
        targets = []
        f = open(label_path)
        for line in f.readlines():
            targets.append(int(line))

        f.close()
        targets = np.array(targets)
        return paths, targets

    def __getitem__(self, index):
        path = self.paths[index]

        img, target = Image.open(path), int(self.targets[index])
        img = Image.fromarray(np.uint8(img))
        trans = torchvision.transforms.CenterCrop(64)
        img = trans(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img , target



class ImagesDataset_NSL(Dataset):
    def __init__(
            self,
            root,
            train,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform=target_transform

        self.train = train

        self.paths, self.targets = self._load_data()
        print(f'{len(self.paths)} images found')
        # 尾号为4 6的作为测试集
    def __len__(self):
        return len(self.paths)

    def _load_data(self):
        image_path = self.root
        name = os.listdir(image_path)
        name.sort(key=lambda x: int(x[:-4]))
        t_paths = []
        if self.train:
            for path in name:
                if path[-5] !='4' and path[-5] !='6':
                    t_paths.append(path)
                    # paths.append(os.path.join(image_path,path))
        else:
            for path in name:
                if path[-5] =='4' or path[-5] =='6':
                    t_paths.append(path)
                    # paths.append(os.path.join(image_path,path))

        targets = []
        paths = []
        for line in t_paths:
            label = (int(line[:-4])-1)//1000
            targets.append(int(label))
            paths.append(os.path.join(image_path, line))

        targets = np.array(targets)
        return paths, targets

    def __getitem__(self, index):
        path = self.paths[index]

        img, target = Image.open(path), int(self.targets[index])
        img = Image.fromarray(np.uint8(img))
        # trans = torchvision.transforms.CenterCrop(64)
        trans = torchvision.transforms.Resize(256)
        img = trans(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img , target
