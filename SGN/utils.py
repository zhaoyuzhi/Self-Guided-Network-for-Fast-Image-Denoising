import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import os

import network

# There are many functions:
# ----------------------------------------
# 1. text_readlines:
# In: a str nominating the a txt
# Parameters: None
# Out: list
# ----------------------------------------
# 2. create_generator:
# In: opt, init_type, init_gain
# Parameters: init type and gain, we highly recommend that Gaussian init with standard deviation of 0.02
# Out: colorizationnet
# ----------------------------------------
# 3. create_discriminator:
# In: opt, init_type, init_gain
# Parameters: init type and gain, we highly recommend that Gaussian init with standard deviation of 0.02
# Out: discriminator_coarse_color, discriminator_coarse_sal, discriminator_fine_color, discriminator_fine_sal
# ----------------------------------------
# 4. create_perceptualnet:
# In: None
# Parameters: None
# Out: perceptualnet
# ----------------------------------------
# 5. load_dict
# In: process_net (the net needs update), pretrained_net (the net has pre-trained dict)
# Out: process_net (updated)
# ----------------------------------------
# 6. savetxt
# In: list
# Out: txt
# ----------------------------------------
# 7. get_files
# In: path
# Out: txt
# ----------------------------------------
# 8. get_jpgs
# In: path
# Out: txt
# ----------------------------------------
# 9. text_save
# In: list
# Out: txt
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

def create_generator(opt):
    if opt.pre_train:
        # Initialize the network
        generator = network.SGN(opt)
        # Init the network
        network.weights_init(SGN, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Generator is created!')
    else:
        # Initialize the network
        generator = network.SGN(opt)
        # Load a pre-trained network
        pretrained_net = torch.load(opt.load_name + '.pth')
        load_dict(generator, pretrained_net)
        print('Generator is loaded!')
    return generator
    
def create_discriminator(opt):
    # Initialize the network
    discriminator = network.CyclePatchDiscriminator(opt)
    # Init the network
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Discriminators is created!')
    return discriminator
    
def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net.state_dict()
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            ret.append(os.path.join(root,filespath)) 
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = [] 
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            ret.append(filespath) 
    return ret

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0) # Decay must start before the training session ends!
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

'''
a = torch.randn(1, 3, 4, 4)
b = torch.randn(1, 3, 4, 4)
c = (a, b)
d = repackage_hidden(c)
print(d)
'''
'''
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight = 1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class GradLoss(nn.Module):
    def __init__(self, GradLoss_weight = 1):
        super(GradLoss, self).__init__()
        self.GradLoss_weight = GradLoss_weight
        self.MSEloss = nn.MSELoss()

    def forward(self, x, y):
        h_x = x.size()[2]
        w_x = x.size()[3]

        x_h_grad = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]
        x_w_grad = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
        y_h_grad = y[:, :, 1:, :] - y[:, :, :h_x - 1, :]
        y_w_grad = y[:, :, :, 1:] - y[:, :, :, :w_x - 1]
        
        h_loss = self.MSEloss(x_h_grad, y_h_grad)
        w_loss = self.MSEloss(x_w_grad, y_w_grad)
        
        return self.GradLoss_weight * (h_loss + w_loss)
'''
