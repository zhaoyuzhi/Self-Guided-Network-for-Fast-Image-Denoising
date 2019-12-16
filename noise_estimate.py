import argparse
import os
import random
import numpy as np
import cv2
import torch

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def estimate(pred, target, pixel_max_cnt = 255):
    mse = np.multiply(target - pred, target - pred)
    rmse_avg = (np.mean(mse)) ** 0.5
    psnr = 20 * np.log10(pixel_max_cnt / rmse_avg)
    l2 = np.mean(mse)
    l1 = np.mean(np.abs(target - pred))
    return psnr, l2, l1

if __name__ == "__main__":

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseroot', type = str, default = 'C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256', help = 'root')
    parser.add_argument('--num', type = int, default = 1000, help = 'number of sample images')
    parser.add_argument('--range', type = str, default = '01', help = 'normalization range, e.g. 01 represents [0, 1]')
    parser.add_argument('--gray', type = bool, default = True, help = 'False for color images, True for grayscale')
    parser.add_argument('--noise_scale', type = float, default = 0.01, help = 'Gaussian noise standard deviation')
    opt = parser.parse_args()

    # random sample all the images
    imglist = get_files(opt.baseroot)
    imglist = random.sample(imglist, opt.num)

    # add noise
    if opt.range == '01':
        overall_psnr = 0
        overall_l2 = 0
        overall_l1 = 0
        for (i, imgname) in enumerate(imglist):
            print('Now it is the %d-th image' % (i))
            img = cv2.imread(imgname).astype(np.float32)
            if opt.gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img / 255.0
            noise = np.random.normal(loc = 0.0, scale = opt.noise_scale, size = img.shape)
            img_noise = img + noise
            current_psnr, l2, l1 = estimate(img_noise, img, 1)
            print('PSNR:', current_psnr)
            print('L2 Loss:', l2)
            print('L1 Loss:', l1)
            overall_psnr += current_psnr
        overall_psnr = overall_psnr / len(imglist)
        overall_l2 = overall_l2 / len(imglist)
        overall_l1 = overall_l1 / len(imglist)
        print('Overall PSNR:', overall_psnr)
        print('Overall L2:', overall_l2)
        print('Overall L1:', overall_l1)

    if opt.range == '-11':
        overall_psnr = 0
        overall_l2 = 0
        overall_l1 = 0
        for (i, imgname) in enumerate(imglist):
            print('Now it is the %d-th image' % (i))
            img = cv2.imread(imgname).astype(np.float32)
            if opt.gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img - 128.0) / 128.0
            noise = np.random.normal(loc = 0.0, scale = opt.noise_scale, size = img.shape)
            img_noise = img + noise
            current_psnr, l2, l1 = estimate(img_noise, img, 1)
            print('PSNR:', current_psnr)
            print('L2 Loss:', l2)
            print('L1 Loss:', l1)
            overall_psnr += current_psnr
        overall_psnr = overall_psnr / len(imglist)
        overall_l2 = overall_l2 / len(imglist)
        overall_l1 = overall_l1 / len(imglist)
        print('Overall PSNR:', overall_psnr)
        print('Overall L2:', overall_l2)
        print('Overall L1:', overall_l1)

    if opt.range == '0255':
        overall_psnr = 0
        overall_l2 = 0
        overall_l1 = 0
        for (i, imgname) in enumerate(imglist):
            print('Now it is the %d-th image' % (i))
            img = cv2.imread(imgname).astype(np.float32)
            if opt.gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            noise = np.random.normal(loc = 0.0, scale = opt.noise_scale, size = img.shape)
            img_noise = img + noise
            current_psnr, l2, l1 = estimate(img_noise, img, 1)
            print('PSNR:', current_psnr)
            print('L2 Loss:', l2)
            print('L1 Loss:', l1)
            overall_psnr += current_psnr
        overall_psnr = overall_psnr / len(imglist)
        overall_l2 = overall_l2 / len(imglist)
        overall_l1 = overall_l1 / len(imglist)
        print('Overall PSNR:', overall_psnr)
        print('Overall L2:', overall_l2)
        print('Overall L1:', overall_l1)
