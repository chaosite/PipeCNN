#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import os
import sys
import math

import numpy as np

import cv2

data = dset.CIFAR10(root='cifar', train=True, download=True, transform=transforms.ToTensor()).train_data
data = data.astype(np.float32)
img = data.mean(axis=0)
# cv2.imwrite("/home/ANT.AMAZON.COM/ofririps/workspace/PipeCNN-DL/model/mean_data.png", data.mean(axis=0))

cv2.imwrite("/home/ANT.AMAZON.COM/ofririps/workspace/PipeCNN-DL/model/mean_data.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

