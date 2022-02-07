# %%
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch
import torchvision.transforms as transform
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets

# %%
def getdDataset(opt):
    normalize = transform.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    trans = transform.Compose(
        [transform.Resize(opt.img_size), transform.ToTensor(), normalize])

    if opt.dataset == 'mnist':
        dst = datasets.MNIST(
            # 相对路径，以调用的文件位置为准
            root=opt.dataroot,
            train=True,
            download=True,
            transform=transform.Compose(
                [transform.Resize(opt.img_size), transform.ToTensor(
                ), transform.Normalize([0.5], [0.5])]
            )
        )
    elif opt.dataset == 'fashion':
        dst = datasets.FashionMNIST(
            root=opt.dataroot,
            train=True,
            download=True,
            # split='mnist',
            transform=transform.Compose(
                [transform.Resize(opt.img_size), transform.ToTensor(
                ), transform.Normalize([0.5], [0.5])]
            )
        )
    elif opt.dataset == 'cifar10':
        dst = datasets.CIFAR10(
            root=opt.dataroot,
            train=True,
            download=True,
            transform=transform.Compose(
                [transform.Resize(opt.img_size),
                 transform.ToTensor(), normalize]
            )
        )
    elif opt.dataset == 'imagenet':
        dst = datasets.ImageNet(
            root=opt.dataroot,
            split='train',
            transform=trans,
            # loader=
        )

    dataloader = DataLoader(
        dst,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    return dataloader


# %%

if __name__ == "__main__":
    class opt:
        dataroot = '/GANs/data/imagenet_2012/'
        dataset = 'imagenet'
        img_size = 32
        batch_size = 10

    dataloader = getdDataset(opt)
    for i, (imgs, labels) in enumerate(dataloader):
        print(i, imgs.shape, labels.shape)
        print(labels)

        img = imgs[0]
        img = img.numpy()
        img = make_grid(imgs, normalize=True).numpy()
        img = np.transpose(img, (1, 2, 0))

        plt.imshow(img)
        plt.show()
        plt.close()
        break
# %%

# Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
