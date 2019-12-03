import os
import random
import numpy as np
import cv2
import math
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
        self.opt = opt
        self.imglist = utils.get_files(opt.baseroot)

    def __getitem__(self, index):
        ## read an image
        img = cv2.imread(self.imglist[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ## data augmentation
        # random scale
        if self.opt.geometry_aug:
            H_in = img[0].shape[0]
            W_in = img[0].shape[1]
            sc = np.random.uniform(self.opt.scale_min, self.opt.scale_max)
            H_out = int(math.floor(H_in * sc))
            W_out = int(math.floor(W_in * sc))
            # scaled size should be greater than opts.crop_size
            if H_out < W_out:
                if H_out < self.opt.crop_size:
                    H_out = self.opt.crop_size
                    W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
            else: # W_out < H_out
                if W_out < self.opt.crop_size:
                    W_out = self.opt.crop_size
                    H_out = int(math.floor(H_in * float(W_out) / float(W_in)))
            img = cv2.resize(img, (W_out, H_out))
        # random crop
        cropper = RandomCrop(img.shape[:2], (self.opt.crop_size, self.opt.crop_size))
        img = cropper(img)
        # random rotate and horizontal flip
        # according to paper, these two data augmentation methods are recommended
        if self.opt.angle_aug:
            rotate = random.randint(0, 3)
            if rotate != 0:
                img = np.rot90(img, rotate)
            if np.random.random() >= 0.5:
                img = cv2.flip(img, flipCode = 0)
        
        # add noise
        img = img.astype(np.float32) # RGB image in range [0, 255]
        noise = np.random.normal(self.opt.mu, self.opt.sigma, img.shape).astype(np.float32)
        noisy_img = img + noise

        # normalization
        img = (img - 128) / 128
        img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
        noisy_img = (noisy_img - 128) / 128
        noisy_img = torch.from_numpy(noisy_img.transpose(2, 0, 1)).contiguous()

        return noisy_img, img
    
    def __len__(self):
        return len(self.imglist)

class FullResDenoisingDataset(Dataset):
    def __init__(self, opt):                                   		    # root: list ; transform: torch transform
        self.opt = opt
        self.imglist = utils.get_files(opt.baseroot)

    def __getitem__(self, index):
        ## read an image
        img = cv2.imread(self.imglist[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ## re-arrange the data for fitting network
        H_in = img[0].shape[0]
        W_in = img[0].shape[1]
        H_out = int(math.floor(H_in / 8)) * 8
        W_out = int(math.floor(W_in / 8)) * 8
        img = cv2.resize(img, (W_out, H_out))
        
        # add noise
        img = img.astype(np.float32) # RGB image in range [0, 255]
        noise = np.random.normal(self.opt.mu, self.opt.sigma, img.shape).astype(np.float32)
        noisy_img = img + noise

        # normalization
        img = (img - 128) / 128
        img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
        noisy_img = (noisy_img - 128) / 128
        noisy_img = torch.from_numpy(noisy_img.transpose(2, 0, 1)).contiguous()

        return noisy_img, img
    
    def __len__(self):
        return len(self.imglist)
