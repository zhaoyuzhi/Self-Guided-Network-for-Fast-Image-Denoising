import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = "/home/alien/Documents/zhaoyuzhi/denoising", help = 'the testing folder')
    parser.add_argument('--geometry_aug', type = bool, default = False, help = 'geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type = bool, default = False, help = 'geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type = float, default = 1, help = 'min scaling factor')
    parser.add_argument('--scale_max', type = float, default = 1, help = 'max scaling factor')
    parser.add_argument('--mu', type = float, default = 0, help = 'min scaling factor')
    parser.add_argument('--sigma', type = float, default = 30, help = 'max scaling factor')
    # Other parameters
    parser.add_argument('--batch_size', type = int, default = 1, help = 'test batch size, always 1')
    parser.add_argument('--load_name', type = str, default = 'SGN_epoch1000_batchsize64.pth', help = 'test model name')
    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------

    # Define the dataset
    testset = dataset.FullResDenoisingDataset(opt)
    print('The overall number of images equals to %d' % len(testset))

    # Define the dataloader
    dataloader = DataLoader(testset, batch_size = opt.batch_size, pin_memory = True)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------

    model = torch.load(opt.load_name)

    for batch_idx, (noisy_img, img) in enumerate(dataloader):

        # To Tensor
        noisy_img = noisy_img.cuda()
        img = img.cuda()

        # Generator output
        recon_img = model(noisy_img)

        # convert to visible image format
        h = img.shape[2]
        w = img.shape[3]
        img = img.cpu().numpy().reshape(3, h, w).transpose(1, 2, 0)
        img = (img + 1) * 128
        img = img.astype(np.uint8)
        recon_img = recon_img.detach().cpu().numpy().reshape(3, h, w).transpose(1, 2, 0)
        recon_img = (recon_img + 1) * 128
        recon_img = recon_img.astype(np.uint8)

        # show
        show_img = np.concatenate((img, recon_img), axis = 1)
        r, g, b = cv2.split(show_img)
        show_img = cv2.merge([b, g, r])
        cv2.imshow('comparison.jpg', show_img)
        cv2.imwrite('result_%d.jpg' % batch_idx, show_img)
