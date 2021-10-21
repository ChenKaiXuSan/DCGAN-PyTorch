# %% 
import os
import numpy as np 
import torch
import torch.autograd as autograd
import torch.nn as nn
from torchvision.utils import save_image

import shutil

# %%
def del_folder(path, version):
    '''
    delete the folder which path/version

    Args:
        path (str): path
        version (str): version
    '''    
    if os.path.exists(os.path.join(path, version)):
        shutil.rmtree(os.path.join(path, version))
    
def make_folder(path, version):
    '''
    make folder which path/version

    Args:
        path (str): path
        version (str): version
    '''    
    if not os.path.exists(os.path.join(path, version)):
        os.makedirs(os.path.join(path, version))

def tensor2var(x, grad=False):
    '''
    put tensor to gpu, and set grad to false

    Args:
        x (tensor): input tensor
        grad (bool, optional):  Defaults to False.

    Returns:
        tensor: tensor in gpu and set grad to false 
    '''    
    if torch.cuda.is_available():
        x = x.cuda()
        x.requires_grad_(grad)
    return x

def var2tensor(x):
    '''
    put date to cpu

    Args:
        x (tensor): input tensor 

    Returns:
        tensor: put data to cpu
    '''    
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def str2bool(v):
    return v.lower() in ('true')

def to_Tensor(x, *arg):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.LongTensor
    return Tensor(x, *arg)

# custom weights initialization called on netG and netD
def weights_init(m):
    '''
    custom weights initializaiont called on G and D, from the paper DCGAN

    Args:
        m (tensor): the network parameters
    '''    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_sample_one_image(G, sample_path, real_images, epoch, z_dim, n_classes, number_real=0, number_fake=0):
    if epoch % 1000 == 0:
        
        make_folder(sample_path, str(epoch) + '/real_images')
        make_folder(sample_path, str(epoch) + '/fake_images')
        real_images_path = os.path.join(sample_path, str(epoch), 'real_images')
        fake_images_path = os.path.join(sample_path, str(epoch), 'fake_images')

        if len(os.listdir(real_images_path)) <= 1000:
            # save real image
            for i in range(real_images.size(0)):
                one_real_image = real_images[i]
                save_image(
                    one_real_image.data, 
                    os.path.join(real_images_path, '{}_real.png'.format(number_real)),
                    normalize=True
                )
                number_real += 1

            # save fake image 
            # for generate sample
                

            fixed_z = tensor2var(torch.randn(real_images.size(0), z_dim))
            # labels = np.array([num for _ in range(n_classes) for num in range(n_classes)])
            # labels = np.random.randint(0, n_classes, real_images.size(0))
            with torch.no_grad():
                # labels = to_LongTensor(labels)
                gen_image = G(fixed_z)

                for i in range(gen_image.size(0)):
                    one_fake_image = gen_image[i]
                    save_image(
                        one_fake_image.data,
                        os.path.join(fake_images_path, '{}_fake.png'.format(number_fake)),
                        normalize = True,
                        nrow=n_classes
                    )
                    number_fake += 1

    return number_real, number_fake

def save_sample(path, images, epoch):
    '''
    save the tensor sample to nrow=10 image

    Args:
        path (str): saved path
        images (tensor): images want to save
        epoch (int): now epoch int, for the save image name
    '''    
    save_image(images.data[:100], os.path.join(path, '{}.png'.format(epoch)), normalize=True, nrow=10)