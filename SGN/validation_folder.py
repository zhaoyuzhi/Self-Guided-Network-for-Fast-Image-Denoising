import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset
import utils

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
    parser.add_argument('--baseroot', type = str, default = "/home/alien/Documents/LINTingyu/denoising", help = 'the testing folder')
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
    #       Initialize testing dataset
    # ----------------------------------------

    # Define the dataset
    testset = dataset.DenoisingDataset(opt)
    print('The overall number of images equals to %d' % len(testset))

    # Define the dataloader
    dataloader = DataLoader(testset, batch_size = opt.batch_size, pin_memory = True)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------
    model = utils.create_generator(opt).cuda()

    for batch_idx, (noisy_img, img) in enumerate(dataloader):

        # To Tensor
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
        cv2.imshow('comparison.jpg', show_img)
        cv2.waitKey(1000)
        cv2.imwrite('result_%d.jpg' % (batch_idx), show_img)
