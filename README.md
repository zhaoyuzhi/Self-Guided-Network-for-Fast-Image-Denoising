# Self-Guided-Network-for-Fast-Image-Denoising

The PyTorch implementation of SGN, and the estimation PSNR of given noise range

## 1 Noise Estimate

zero mean Gaussian noise

| standard deviation in [0, 1] | 0.1 | 0.075 | 0.05 | 0.04 | 0.03 | 0.02 | 0.01 | 0.00075 | 0.0005 | 0.0001 |
| :----- | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| standard deviation in [0, 255] | 25.5 | 19.125 | 12.75 | 10.2 | 7.65 | 5.1 | 2.55 | 1.9125 | 1.275 | 0.0255 |
| average PSNR | 20.00 | 22.46 | 26.00 | 27.95 | 30.47 | 33.98 | 40.00 | 42.49 | 46.00 | 60.00 |
