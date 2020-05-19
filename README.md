# Self-Guided-Network-for-Fast-Image-Denoising

The PyTorch implementation of SGN, and the estimation PSNR of given noise range

## Training

I trained this SGN on Python 3.6 and PyTorch 1.0 environment. The training strategy is the same as paper. You may use following script to train it on your own data (noted that you need to modify dataset path):
```bash
cd SGN
python train.py   or   sh zyz.sh
```

## Testing

I trained it using ILSVRC2012 validation set on 4 NVIDIA TITAN Xp GPUs and tested it on 1 TITAN Xp GPU. The details are shown in code `train.py`. This demo is from SGN on ILSVRC2012 validation set (mu = 0, sigma = 30, batchsize = 32, 1000000 iterations).

left: clean image  (selected from COCO2014 validation set, COCO_val2014_000000264615.jpg)

middle: additive Gaussian noise + clean image

right: denoised image using trained SGN

<img src="./result.jpg" width="1000"/>

You can download pre-trained models (also on ILSVRC2012 validation set, different mu and sigma) via this [link](https://portland-my.sharepoint.com/:f:/g/personal/yzzhao2-c_ad_cityu_edu_hk/EnJkUyTlR1ZPghHh6_Z0GDQBcHpH60LUUtYvKihrILkeRA?e=W10vvF).

You may use following script to test it on your own data (noted that you need to modify dataset path):
```bash
cd SGN
python validation.py   or   python validation_folder.py
```

## Noise Estimate

zero mean Gaussian noise

| standard deviation in [0, 1] | 0.1 | 0.075 | 0.05 | 0.04 | 0.03 | 0.02 | 0.01 | 0.00075 | 0.0005 | 0.0001 |
| :----- | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| standard deviation in [0, 255] | 25.5 | 19.125 | 12.75 | 10.2 | 7.65 | 5.1 | 2.55 | 1.9125 | 1.275 | 0.0255 |
| average PSNR | 20.00 | 22.46 | 26.00 | 27.95 | 30.47 | 33.98 | 40.00 | 42.49 | 46.00 | 60.00 |

## Noise Examples

There are some examples, corresponding to specific noise standard deviation.

### Standard Deviation = 0.3; Standard Deviation = 0.2; Standard Deviation = 0.18; Standard Deviation = 0.15

<img src="./noisy_img_examples/0.3.png" width="200"/><img src="./noisy_img_examples/0.2.png" width="200"/><img src="./noisy_img_examples/0.18.png" width="200"/><img src="./noisy_img_examples/0.15.png" width="200"/>

### Standard Deviation = 0.12; Standard Deviation = 0.1; Standard Deviation = 0.075; Standard Deviation = 0.05

<img src="./noisy_img_examples/0.12.png" width="200"/><img src="./noisy_img_examples/0.1.png" width="200"/><img src="./noisy_img_examples/0.075.png" width="200"/><img src="./noisy_img_examples/0.05.png" width="200"/>

### Standard Deviation = 0.04; Standard Deviation = 0.03; Standard Deviation = 0.02; Standard Deviation = 0.01

<img src="./noisy_img_examples/0.04.png" width="200"/><img src="./noisy_img_examples/0.03.png" width="200"/><img src="./noisy_img_examples/0.02.png" width="200"/><img src="./noisy_img_examples/0.01.png" width="200"/>

### Standard Deviation = 0.0075; Standard Deviation = 0.005; Standard Deviation = 0.001

<img src="./noisy_img_examples/0.0075.png" width="200"/><img src="./noisy_img_examples/0.005.png" width="200"/><img src="./noisy_img_examples/0.001.png" width="200"/>

## Our new work that improves this architecture and achieves 1st Place in NTIRE 2020 Challenge

https://github.com/zhaoyuzhi/Hierarchical-Regression-Network-for-Spectral-Reconstruction-from-RGB-Images
