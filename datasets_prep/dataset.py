import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import CIFAR10, STL10
from .lsun import LSUN
from .stackmnist_data import StackedMNIST, _data_transforms_stacked_mnist
from .lmdb_datasets import LMDBDataset

def create_dataset(args):
    if args.dataset == 'cifar10':
        dataset = CIFAR10(args.datadir, train=True, transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
    elif args.dataset == 'stl10':
        dataset = STL10(args.datadir, split="unlabeled", transform=transforms.Compose([
                        transforms.Resize(64),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
    elif args.dataset == 'stackmnist':
        train_transform, valid_transform = _data_transforms_stacked_mnist()
        dataset = StackedMNIST(root=args.datadir, train=True, download=False, transform=train_transform)

    elif args.dataset == 'tiny_imagenet_200':
        train_transform = transforms.Compose([
                        transforms.Resize(64),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        dataset = datasets.ImageFolder(os.path.join(args.datadir, 'train'), transform=train_transform)
        
    elif args.dataset == 'lsun':
        
        train_transform = transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.CenterCrop(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    ])

        train_data = LSUN(root=args.datadir, classes=['church_outdoor_train'], transform=train_transform)
        subset = list(range(0, 120000))
        dataset = torch.utils.data.Subset(train_data, subset)
      
    
    elif args.dataset == 'celeba_256':
        train_transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        dataset = LMDBDataset(root=args.datadir, name='celeba', train=True, transform=train_transform)

    elif args.dataset == 'celeba_512':
        from torchtoolbox.data import ImageLMDB
        train_transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        dataset = ImageLMDB(db_path=args.datadir, db_name='celeba_512', transform=train_transform, backend="pil")

    elif args.dataset == 'celeba_1024':
        from torchtoolbox.data import ImageLMDB
        train_transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        dataset = ImageLMDB(db_path=args.datadir, db_name='celeba_1024', transform=train_transform, backend="pil")

    elif args.dataset == 'ffhq_256':
        train_transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        dataset = LMDBDataset(root=args.datadir, name='ffhq', train=True, transform=train_transform)

    return dataset
