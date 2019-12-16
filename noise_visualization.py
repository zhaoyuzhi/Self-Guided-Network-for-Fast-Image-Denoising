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
    parser.add_argument('--gray', type = bool, default = False, help = 'False for color images, True for grayscale')
    parser.add_argument('--noise_scale', type = float, default = 0.05, help = 'Gaussian noise standard deviation')
    opt = parser.parse_args()

    # random sample all the images
    imglist = get_files(opt.baseroot)
    imglist = random.sample(imglist, opt.num)
    ran = random.randint(0, opt.num - 1)

    # add noise
    if opt.range == '01':
        img = cv2.imread(imglist[ran]).astype(np.float32)
        if opt.gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0
        noise = np.random.normal(loc = 0.0, scale = opt.noise_scale, size = img.shape)
        img_noise = img + noise
        current_psnr, l2, l1 = estimate(img_noise, img, 1)
        print('PSNR:', current_psnr)
        print('L2 Loss:', l2)
        print('L1 Loss:', l1)
        img_noise = img_noise * 255.0
        img_noise = np.clip(img_noise, 0, 255).astype(np.uint8)
        cv2.imshow('noisy image', img_noise)
        cv2.waitKey(0)

    if opt.range == '-11':
        img = cv2.imread(imglist[ran]).astype(np.float32)
        if opt.gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img - 128.0) / 128.0
        noise = np.random.normal(loc = 0.0, scale = opt.noise_scale, size = img.shape)
        img_noise = img + noise
        current_psnr, l2, l1 = estimate(img_noise, img, 1)
        print('PSNR:', current_psnr)
        print('L2 Loss:', l2)
        print('L1 Loss:', l1)
        img_noise = (img_noise + 1) * 128.0
        img_noise = np.clip(img_noise, 0, 255).astype(np.uint8)
        cv2.imshow('noisy image', img_noise)
        cv2.waitKey(0)

    if opt.range == '0255':
        img = cv2.imread(imglist[ran]).astype(np.float32)
        if opt.gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        noise = np.random.normal(loc = 0.0, scale = opt.noise_scale, size = img.shape)
        img_noise = img + noise
        current_psnr, l2, l1 = estimate(img_noise, img, 1)
        print('PSNR:', current_psnr)
        print('L2 Loss:', l2)
        print('L1 Loss:', l1)
        img_noise = np.clip(img_noise, 0, 255).astype(np.uint8)
        cv2.imshow('noisy image', img_noise)
        cv2.waitKey(0)
    
    #cv2.imwrite(str(opt.noise_scale) + '.png', img_noise)
