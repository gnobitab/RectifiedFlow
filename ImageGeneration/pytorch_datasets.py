import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import os
import numpy as np
import lmdb
import cv2
import random


class celeba_hq_dataset(Dataset):
    """CelebA HQ dataset."""

    def __init__(self, data_dir, batchsize, transform=None):
        self.root_dir = data_dir
        self.num_imgs = len(os.listdir(self.root_dir))
        self.transform = transform
        self.batchsize = batchsize

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        np.random.seed()
        idx = np.random.randint(0, self.num_imgs)
        img_name = os.path.join(self.root_dir, '%d.jpg'%(idx))
        image = io.imread(img_name)
        image = image * 1.0 / 255.0
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        return image



class afhq_dataset(Dataset):
    """AFHQ dataset."""

    def __init__(self, data_dir, batchsize, category='cat', transform=None):
        self.root_dir = os.path.join(data_dir, category)
        self.files = os.listdir(self.root_dir)
        self.num_imgs = len(self.files)
        self.transform = transform
        self.batchsize = batchsize

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        np.random.seed()
        while True:
            try:
                idx = np.random.randint(0, self.num_imgs)
                img_name = os.path.join(self.root_dir, '%s'%(self.files[idx]))
                image = io.imread(img_name)
                break
            except:
                continue
        image = image * 1.0 / 255.0
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        return image


