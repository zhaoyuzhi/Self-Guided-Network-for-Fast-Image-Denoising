import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from skimage import color

import network

# ----------------------------------------
#                 Testing
# ----------------------------------------

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def test(rgb, colornet):
    out_rgb = colornet(rgb)
    out_rgb = out_rgb.cpu().detach().numpy().reshape([1, 256, 256])
    out_rgb = out_rgb.transpose(1, 2, 0)
    out_rgb = (out_rgb * 0.5 + 0.5) * 255
    out_rgb = out_rgb.astype(np.uint8)
    return out_rgb
    
def getImage(root):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    img = Image.open(root).convert('L')
    #img.show()
    #img = img.crop((256, 0, 512, 256))
    rgb = img.resize((256, 256), Image.ANTIALIAS)
    rgb = transform(rgb)
    rgb = rgb.reshape([1, 1, 256, 256]).cuda()
    return rgb

def comparison(root, colornet):
    # Read raw image
    img = Image.open(root).convert('RGB')
    real = img.crop((0, 0, 256, 256))
    real = real.resize((256, 256), Image.ANTIALIAS)
    real = np.array(real)
    # Forward propagation
    torchimg = getImage(root)
    out_rgb = test(torchimg, colornet)
    # Show
    out_rgb = np.concatenate((out_rgb, real), axis = 1)
    img_rgb = Image.fromarray(out_rgb)
    img_rgb.show()
    return img_rgb

def colorization(root, colornet):
    # Forward propagation
    torchimg = getImage(root)
    out_rgb = test(torchimg, colornet)
    # Show
    out_rgb = out_rgb.reshape([256, 256])
    img_rgb = Image.fromarray(out_rgb, mode = 'L' )
    img_rgb.show()
    return img_rgb

def generation(baseroot, saveroot, imglist, colornet):
    for i in range(len(imglist)):
		# Read raw image
        readname = baseroot + imglist[i]
        print(readname)
        # Forward propagation
        torchimg = getImage(readname)
        out_rgb = test(torchimg, colornet)
        # Save
        img_rgb = Image.fromarray(out_rgb)
        savename = saveroot + imglist[i]
        img_rgb.save(savename)
    print('Done!')

if __name__ == "__main__":

    # Define the basic variables
    #root = '/home/alien/Documents/data/ILSVRC2012_val_256/ILSVRC2012_val_00000002.JPEG'
    #colornet = torch.load('G_AB_LSGAN_epoch100_bs1.pth').cuda()
    root = '/home/alien/Documents/LINTingyu/new2old/ILSVRC2012_val_00000002.JPEG'
    colornet = torch.load('G_BA_LSGAN_epoch100_bs1.pth').cuda()
    
    '''
    # Define generation variables
    txtname = 'ILSVRC2012_val_name.txt'
    imglist = text_readlines(txtname)
    baseroot = 'D:\\datasets\\ILSVRC2012\\ILSVRC2012_val_256\\'
    saveroot = 'D:\\datasets\\ILSVRC2012\\ILSVRC2012_val_256_colorization\\'
    '''

    # Choose a task:
    choice = 'colorization'
    save = True

    # comparison: Compare the colorization output and ground truth
    # colorization: Show the colorization as original size
    # generation: Generate colorization results given a folder
    if choice == 'comparison':
        img_rgb = comparison(root, colornet)
        if save:
            imgname = root.split('/')[-1]
            img_rgb.save('./' + imgname)
    if choice == 'colorization':
        img_rgb = colorization(root, colornet)
        if save:
            imgname = root.split('/')[-1]
            img_rgb.save('./' + imgname)
    if choice == 'generation':
        generation(baseroot, saveroot, imglist, colornet)
    

