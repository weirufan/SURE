import torch
import os
import numpy as np
from PIL import Image


from torch.utils.data import DataLoader, Dataset
from typing import Optional,Callable

import torchvision
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']

# images dataset

import random

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
        '''
        image_path = self.root
        name = os.listdir(image_path)
        name.sort(key=lambda x: int(x[:-4]))

        
        key = [1,2,3,4,5,6,7,8,9,10]
        path_ind = []
        q_ind = 0
        all_label = []
        # f = open('../../DATASET/DATAD_label.txt')
        f = open('../../DATASET/DATAD/024_label.txt')
        for line in f.readlines():
            pnd = int(line)
            all_label.append(pnd-1)
            if pnd in key:
                path_ind.append(q_ind)
            q_ind+=1
        f.close()
        print("q_ind len : {}".format(len(path_ind)))
        test_split = int(0.1*len(path_ind))
        random.shuffle(path_ind)
        targets = []
        t_paths = []
        if self.train:
            path_ind = path_ind[:len(path_ind)-test_split]
            for ind in path_ind:
                t_paths.append(name[ind])
                targets.append(all_label[ind])
                    
        else:
            path_ind = path_ind[len(path_ind)-test_split:]
            for ind in path_ind:
                t_paths.append(name[ind])
                targets.append(all_label[ind])

        paths = []
        for line in t_paths:
            paths.append(os.path.join(image_path, line))
        targets = np.array(targets)


        '''
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
        # trans = torchvision.transforms.CenterCrop(64)
        # img = trans(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img , target

def greyscale_make_transforms(input_sz= 64):
  tf_list_train = [
                torchvision.transforms.CenterCrop(input_sz),
                   torchvision.transforms.RandomRotation(degrees=30),
                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                    torchvision.transforms.RandomVerticalFlip(p=0.5),
                    torchvision.transforms.ToTensor(),
                   torchvision.transforms.Normalize(0.5,0.5)]
  tf_list_test =  [torchvision.transforms.CenterCrop(input_sz),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(0.5, 0.5)]

  tf_train = torchvision.transforms.Compose(tf_list_train)
  tf_test = torchvision.transforms.Compose(tf_list_test)

  return tf_train,tf_test


class ImagesDataset2(Dataset):
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
        image_path = self.root
        name = os.listdir(image_path)
        name.sort(key=lambda x: int(x[:-4]))


        key = [1,2,3,4,5,6,7,8,9,10]
        path_ind = []
        q_ind = 0
        all_label = []
        f = open('../../DATASET/DATAD_label.txt')
        # f = open('../../DATASET/DATAD/45_label.txt')
        for line in f.readlines():
            pnd = int(line)
            all_label.append(pnd-1)
            if pnd in key:
                path_ind.append(q_ind)
            q_ind+=1
        f.close()
        print("q_ind len : {}".format(len(path_ind)))
      
        targets = []
        t_paths = []
        for ind in path_ind:
            t_paths.append(name[ind])
            targets.append(all_label[ind])
                    
        paths = []
        for line in t_paths:
            paths.append(os.path.join(image_path, line))
        targets = np.array(targets)

        targets = np.array(targets)
        return paths, targets

    def __getitem__(self, index):
        path = self.paths[index]

        img, target = Image.open(path), int(self.targets[index])
        img = Image.fromarray(np.uint8(img))
        # trans = torchvision.transforms.CenterCrop(64)
        # img = trans(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img , target




if __name__ == '__main__':
    dataset_class = ImagesDataset
    dataset_root ='../../DATASET/mnist_zno_02345679'
    tf1,tf2 = greyscale_make_transforms(64)

    img_curr = dataset_class(
        root = dataset_root,
        transform=tf1,
        train = False
    )
    dataloader = torch.utils.data.DataLoader(img_curr,batch_size=16,shuffle=False)
    for data in dataloader:
        x,y = data
        print(x.shape)
        print(y)
        print('=='*20)



