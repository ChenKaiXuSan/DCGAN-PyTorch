# DCGAN-PyTorch

## Overview
This repository contains an Pytorch implementation of Deep Convolution GAN and Linear GAN structure.
With full coments and my code style.

## About DCGAN
If you're new to DCGAN, here's an abstract straight from the paper[1]:

In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsuper- vised learning. We introduce a class ofCNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolu- tional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.

## Dataset 
- MNIST
`python3 main.py --dataset mnist --channels 1`
- FashionMNIST
`python3 main.py --dataset fashion --channels 1`
- Cifar10
`python3 main.py --dataset cifar10 --channels 3`

## Implement
``` python
usage: main.py [-h] [--model {gan,dcgan}] [--img_size IMG_SIZE]
               [--channels CHANNELS] [--g_num G_NUM] [--z_dim Z_DIM]
               [--g_conv_dim G_CONV_DIM] [--d_conv_dim D_CONV_DIM]
               [--version VERSION] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
               [--num_workers NUM_WORKERS] [--g_lr G_LR] [--d_lr D_LR]
               [--beta1 BETA1] [--beta2 BETA2]
               [--pretrained_model PRETRAINED_MODEL] [--train TRAIN]
               [--parallel PARALLEL] [--dataset {mnist,cifar10,fashion}]
               [--use_tensorboard USE_TENSORBOARD] [--dataroot DATAROOT]
               [--log_path LOG_PATH] [--model_save_path MODEL_SAVE_PATH]
               [--sample_path SAMPLE_PATH] [--log_step LOG_STEP]
               [--sample_step SAMPLE_STEP] [--model_save_step MODEL_SAVE_STEP]

optional arguments:
  -h, --help            show this help message and exit
  --model {gan,dcgan}
  --img_size IMG_SIZE
  --channels CHANNELS   number of image channels
  --g_num G_NUM         train the generator every 5 steps
  --z_dim Z_DIM         noise dim
  --g_conv_dim G_CONV_DIM
  --d_conv_dim D_CONV_DIM
  --version VERSION     the version of the path, for implement
  --epochs EPOCHS       numer of epochs of training
  --batch_size BATCH_SIZE
                        batch size for the dataloader
  --num_workers NUM_WORKERS
  --g_lr G_LR           use TTUR lr rate for Adam
  --d_lr D_LR           use TTUR lr rate for Adam
  --beta1 BETA1
  --beta2 BETA2
  --pretrained_model PRETRAINED_MODEL
  --train TRAIN
  --parallel PARALLEL
  --dataset {mnist,cifar10,fashion}
  --use_tensorboard USE_TENSORBOARD
                        use tensorboard to record the loss
  --dataroot DATAROOT   dataset path
  --log_path LOG_PATH   the output log path
  --model_save_path MODEL_SAVE_PATH
                        model save path
  --sample_path SAMPLE_PATH
                        the generated sample saved path
  --log_step LOG_STEP   every default{10} epoch save to the log
  --sample_step SAMPLE_STEP
                        every default{100} epoch save the generated images and
                        real images
  --model_save_step MODEL_SAVE_STEP
```

## Usage
- MNSIT
`python3 main.py --dataset mnist --channels 1 --version [version] --batch_size [] >logs/[log_path]`
- FashionMNIST
`python3 main.py --dataset fashion --channels 1 --version [version] --batch_size [] >logs/[log_path]`
- Cifar10
`python3 main.py --dataset cifar10 --channels 3 -version [version] --batch_size [] >logs/[log_path]`

## Tensorboard 
The G loss and D loss record by the tensorboard in the folder, /logs.
``` python
tensorboard --logdir logs/
```

## FID
FID is a measure of similarity between two datasets of images. It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks. FID is calculated by computing the Fréchet distance between two Gaussians fitted to feature representations of the Inception network.

For the FID, I use the pytorch implement of this repository. [FID score for PyTorch](https://github.com/mseitzer/pytorch-fid)

For the 10k epochs training on different dataset, compare with about 10000 samples, I get the FID: 

| dataset | DCGAN |
| ---- | ---- |
| MNIST | 42.45529067975039(9500epoch) |
| FASHION-MNIST | 86.17794805798678(7800epoch) | 
| CIFAR10 | 80.03665139141242(9200epoch) |

> :warning: I dont konw if the FID is right or not, because I cant get the lowwer score like the paper or the other people get it. 
## Network structure
``` python
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Generator                                --                        --
├─Sequential: 1-1                        [64, 512, 4, 4]           --
│    └─ConvTranspose2d: 2-1              [64, 512, 4, 4]           819,200
│    └─BatchNorm2d: 2-2                  [64, 512, 4, 4]           1,024
│    └─ReLU: 2-3                         [64, 512, 4, 4]           --
├─Sequential: 1-2                        [64, 256, 8, 8]           --
│    └─ConvTranspose2d: 2-4              [64, 256, 8, 8]           2,097,152
│    └─BatchNorm2d: 2-5                  [64, 256, 8, 8]           512
│    └─ReLU: 2-6                         [64, 256, 8, 8]           --
├─Sequential: 1-3                        [64, 128, 16, 16]         --
│    └─ConvTranspose2d: 2-7              [64, 128, 16, 16]         524,288
│    └─BatchNorm2d: 2-8                  [64, 128, 16, 16]         256
│    └─ReLU: 2-9                         [64, 128, 16, 16]         --
├─Sequential: 1-4                        [64, 64, 32, 32]          --
│    └─ConvTranspose2d: 2-10             [64, 64, 32, 32]          131,072
│    └─BatchNorm2d: 2-11                 [64, 64, 32, 32]          128
│    └─ReLU: 2-12                        [64, 64, 32, 32]          --
├─Sequential: 1-5                        [64, 1, 64, 64]           --
│    └─ConvTranspose2d: 2-13             [64, 1, 64, 64]           1,024
│    └─Tanh: 2-14                        [64, 1, 64, 64]           --
==========================================================================================
Total params: 3,574,656
Trainable params: 3,574,656
Non-trainable params: 0
Total mult-adds (G): 26.88
==========================================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 127.93
Params size (MB): 14.30
Estimated Total Size (MB): 142.25
==========================================================================================
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Discriminator                            --                        --
├─Sequential: 1-1                        [64, 64, 32, 32]          --
│    └─Conv2d: 2-1                       [64, 64, 32, 32]          3,072
│    └─LeakyReLU: 2-2                    [64, 64, 32, 32]          --
├─Sequential: 1-2                        [64, 128, 16, 16]         --
│    └─Conv2d: 2-3                       [64, 128, 16, 16]         131,072
│    └─BatchNorm2d: 2-4                  [64, 128, 16, 16]         256
│    └─LeakyReLU: 2-5                    [64, 128, 16, 16]         --
├─Sequential: 1-3                        [64, 256, 8, 8]           --
│    └─Conv2d: 2-6                       [64, 256, 8, 8]           524,544
│    └─BatchNorm2d: 2-7                  [64, 256, 8, 8]           512
│    └─LeakyReLU: 2-8                    [64, 256, 8, 8]           --
├─Sequential: 1-4                        [64, 512, 4, 4]           --
│    └─Conv2d: 2-9                       [64, 512, 4, 4]           2,097,664
│    └─BatchNorm2d: 2-10                 [64, 512, 4, 4]           1,024
│    └─LeakyReLU: 2-11                   [64, 512, 4, 4]           --
├─Sequential: 1-5                        [64, 1, 1, 1]             --
│    └─Conv2d: 2-12                      [64, 1, 1, 1]             8,192
==========================================================================================
Total params: 2,766,336
Trainable params: 2,766,336
Non-trainable params: 0
Total mult-adds (G): 6.65
==========================================================================================
Input size (MB): 3.15
Forward/backward pass size (MB): 92.28
Params size (MB): 11.07
Estimated Total Size (MB): 106.49
==========================================================================================

```
## Result
- MNIST  
![9900_MNSIT](img/9900_MNIST.png)
- CIFAR10  
![9900_cifar10](img/9900_cifar10.png)
- Fashion-MNIST
![9900_fashion](img/9900_fashion.png)
## Reference
1. [DCGAN](https://arxiv.org/abs/1511.06434)
2. [GAN](https://arxiv.org/abs/1406.2661)
