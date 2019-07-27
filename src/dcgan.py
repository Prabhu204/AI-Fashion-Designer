# -*- coding: utf-8 -*-
"""
@author: Prabhu <prabhu.appalapuri@gmail.com>
"""

import torch
import torch.nn as nn
import torchvision.transforms as transform
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
from src.generator import Gen
from src.discriminator import Disc

torch.manual_seed(546)

image_size = 64  # using image size as 64x64 with 3 channel i.e 3x64x64
dataset = dset.ImageFolder(root='data/celeba', transform=transform.Compose([transform.Resize(image_size),
                                                                            transform.CenterCrop(image_size),
                                                                            transform.ToTensor(),
                                                                            transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))

datasetLoader = DataLoader(dataset=dataset, batch_size= 128, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
sample  = iter(datasetLoader).__next__()
# print(vutils.make_grid(sample[0].to(device)[:64], padding= 2, normalize= True).cpu().size())
# print(sample)
plt.figure(figsize=(8,8))
plt.axis('off')
plt.title("Train set images")
plt.imshow(np.transpose(vutils.make_grid(sample[0].to(device)[:64], padding= 2, normalize= True).cpu(),(1,2,0)))

# initialize weights for Generator and Discriminator networks
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') !=-1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


model_G = Gen(num_GenFeatureMaps= 64, num_OutputChannels= 3, latent_vector= 100).to(device)
model_D = Disc(num_Channels=3, num_DisFeaturesMaps= 64, latent_vectors= 100).to(device)

model_G.apply(weights_init)
model_D.apply(weights_init)