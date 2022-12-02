import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import os
import numpy as np
import lmdb
import cv2
import random

class reflow_dataset(Dataset):
    """Reflow dataset: z0 and data."""

    def __init__(self, batchsize, category='train', transform=None):
        self.root_dir = '/scratch/cluster/xcliu/ODE_Diffusion/assets/rematch_data_split'
        if category == 'train':
            self.folder = [i for i in range(16)]
        else:
            self.folder = [16]
        self.num_imgs = 512000
        self.transform = transform
        self.batchsize = batchsize

    def __len__(self):
        return 30000

    def __getitem__(self, idx):
        np.random.seed()
        while True:
            try:
                folder_idx = random.choice(self.folder)
                idx = np.random.randint(0, self.num_imgs)
                z0_name = os.path.join(self.root_dir, folder_idx, 'rematch_fake_z0_ckpt_fid_257_%d.npy'%idx)
                img_name = os.path.join(self.root_dir, folder_idx, 'rematch_fake_data_ckpt_fid_257_%d.npy'%idx)
                import time
                start =time.time()
                image = np.load(img_name)
                z0 = np.load(z0_name)
                end = time.time()
                print(end - start)
                break
            except:
                continue

        image = torch.from_numpy(image)
        z0 = torch.from_numpy(z0)

        return z0, image



class celeba_hq_dataset(Dataset):
    """CelebA HQ dataset."""

    def __init__(self, batchsize, transform=None):
        self.root_dir = '/scratch/cluster/xcliu/tf_datasets/CelebAMask-HQ/CelebA-HQ-img/'
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

    def __init__(self, batchsize, category='cat', transform=None):
        self.root_dir = os.path.join('/scratch/cluster/xcliu/tf_datasets/afhq/data/train/', category)
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


