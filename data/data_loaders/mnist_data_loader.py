from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from Utils.utils import *
import matplotlib.pyplot as plt
import numpy as np
import random
import time 

class MnistDataLoader:
    def __init__(self):
        self.train_transforms = transforms.Compose([
            transforms.RandomRotation((-7.0, 7.0), fill=(1,)),                                   
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        self.train = datasets.MNIST('./data', train=True, download=True, transform=self.train_transforms)
        self.test = datasets.MNIST('./data', train=False, download=True, transform=self.test_transforms)
			
        SEED = 1

    	# CUDA?
        cuda = torch.cuda.is_available()
        #logger.info("CUDA Available?", cuda)
        # logger.info(f"CUDA Available? {cuda}")

        # For reproducibility
        torch.manual_seed(SEED)

        if cuda:
            torch.cuda.manual_seed(SEED)

        # dataloader arguments - something you'll fetch these from cmdprmt
        self.dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

        # train dataloader
        self.train_loader = torch.utils.data.DataLoader(self.train, **self.dataloader_args)

        # test dataloader
        self.test_loader = torch.utils.data.DataLoader(self.test, **self.dataloader_args)

