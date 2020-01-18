import argparse
import random
import math
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

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

def img_processor(opt):
    ## read an image
    img = cv2.imread(opt.baseroot)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ## data augmentation
    # random scale
    if opt.geometry_aug:
        H_in = img[0].shape[0]
        W_in = img[0].shape[1]
        sc = np.random.uniform(opt.scale_min, opt.scale_max)
        H_out = int(math.floor(H_in * sc))
        W_out = int(math.floor(W_in * sc))
        # scaled size should be greater than opts.crop_size
        if H_out < W_out:
            if H_out < opt.crop_size:
                H_out = opt.crop_size
                W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
        else: # W_out < H_out
            if W_out < opt.crop_size:
                W_out = opt.crop_size
                H_out = int(math.floor(H_in * float(W_out) / float(W_in)))
        img = cv2.resize(img, (W_out, H_out))
    # random crop
    cropper = RandomCrop(img.shape[:2], (opt.crop_size, opt.crop_size))
    img = cropper(img)
    # random rotate and horizontal flip
    # according to paper, these two data augmentation methods are recommended
    if opt.angle_aug:
        rotate = random.randint(0, 3)
        if rotate != 0:
            img = np.rot90(img, rotate)
        if random.random() >= 0.5:
            img = cv2.flip(img, flipCode = 0)
    
    # add noise
    img = img.astype(np.float32) # RGB image in range [0, 255]
    noise = np.random.normal(opt.mu, opt.sigma, img.shape).astype(np.float32)
    noisy_img = img + noise

    # normalization
    img = (img - 128) / 128
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    noisy_img = (noisy_img - 128) / 128
    noisy_img = torch.from_numpy(noisy_img.transpose(2, 0, 1)).unsqueeze(0).contiguous()

    return noisy_img, img

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = 'input channels for generator')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output channels for generator')
    parser.add_argument('--start_channels', type = int, default = 32, help = 'start channels for generator')
    parser.add_argument('--m_block', type = int, default = 2, help = 'the additional blocks used in mainstream')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = "/mnt/lustre/zhaoyuzhi/dataset/COCO2014/COCO2014_val_256/COCO_val2014_000000264615.jpg", help = 'the testing folder')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    parser.add_argument('--geometry_aug', type = bool, default = False, help = 'geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type = bool, default = False, help = 'geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type = float, default = 1, help = 'min scaling factor')
    parser.add_argument('--scale_max', type = float, default = 1, help = 'max scaling factor')
    parser.add_argument('--mu', type = float, default = 0, help = 'min scaling factor')
    parser.add_argument('--sigma', type = float, default = 30, help = 'max scaling factor')
    # Other parameters
    parser.add_argument('--pre_train', type = bool, default = False, help = 'test phase')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'test batch size, always 1')
    parser.add_argument('--load_name', type = str, default = 'SGN_iter1000000_bs32_mu0_sigma30.pth', help = 'test model name')
    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------
    model = utils.create_generator(opt).cuda()

    # To Tensor
    noisy_img, img = img_processor(opt)
    noisy_img = noisy_img.cuda()
    img = img.cuda()

    # Generator output
    with torch.no_grad():
        recon_img = model(noisy_img)

    # convert to visible image format
    img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = (img + 1) * 128
    img = img.astype(np.uint8)
    noisy_img = noisy_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    noisy_img = (noisy_img + 1) * 128
    noisy_img = noisy_img.astype(np.uint8)
    recon_img = recon_img.data.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    recon_img = (recon_img + 1) * 128
    recon_img = recon_img.astype(np.uint8)

    # show
    show_img = np.concatenate((img, noisy_img, recon_img), axis = 1)
    r, g, b = cv2.split(show_img)
    show_img = cv2.merge([b, g, r])
    #cv2.imshow('comparison.jpg', show_img)
    #cv2.waitKey(1000)
    cv2.imwrite('result.jpg', show_img)
