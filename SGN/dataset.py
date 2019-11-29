import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import utils

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1: self.h2, self.w1: self.w2, :]
        else:
            return img[self.h1: self.h2, self.w1: self.w2]

class DenoisingDataset(Dataset):
    def __init__(self, opt):                                   		    # root: list ; transform: torch transform
        self.baseroot_A = opt.baseroot_A
        self.baseroot_B = opt.baseroot_B
        self.imglist_A = utils.get_jpgs(opt.baseroot_A)
        self.imglist_B = utils.get_jpgs(opt.baseroot_B)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        imgpath_A = os.path.join(self.baseroot_A, self.imglist_A)       # path of one image
        img_A = Image.open(imgpath_A).convert('RGB')                    # read one image
        img_A = self.transform(img_A)
        imgpath_B = os.path.join(self.baseroot_B, self.imglist_B)       # path of one image
        img_B = Image.open(imgpath_B).convert('RGB')                    # read one image
        img_B = self.transform(img_B)
        return img_A, img_B
    
    def __len__(self):
        return len(self.imglist_A)
