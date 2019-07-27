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


# latent space_vectors --> size of the input (here considered as 100)
# num_Outputchannels --> orignal image channels (i.e 3 for RGB, 1 for Gray scale)
class Gen(nn.Module):
    def __init__(self, device, num_GenFeatureMaps, latent_vector, num_OutputChannels):
        super(Gen, self).__init__()
        self.device = device
        self.num_GenFeatureMaps  =num_GenFeatureMaps
        self.latent_vector = latent_vector
        self.num_OutputChannels = num_OutputChannels
        self.ConvG1= nn.Sequential(nn.ConvTranspose2d(latent_vector,num_GenFeatureMaps*8, kernel_size=4,stride=1, padding=0,bias=False),
                                   nn.BatchNorm2d(num_GenFeatureMaps*8),
                                   nn.ReLU(True))
        self.ConvG2 =nn.Sequential(nn.ConvTranspose2d(num_GenFeatureMaps*8, num_GenFeatureMaps*4,kernel_size= 4,stride=2,padding=1,bias=False),
                                   nn.BatchNorm2d(num_GenFeatureMaps*4),
                                   nn.ReLU(True))
        self.ConvG3 = nn.Sequential(nn.ConvTranspose2d(num_GenFeatureMaps*4,num_GenFeatureMaps*2, kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(num_GenFeatureMaps*2),
                                    nn.ReLU(True))
        self.ConvG4 = nn.Sequential(nn.ConvTranspose2d(num_GenFeatureMaps*2, num_GenFeatureMaps, kernel_size=4, stride=2, padding= 1, bias=False),
                                    nn.BatchNorm2d(num_GenFeatureMaps),
                                    nn.ReLU(True))
        self.ConvG4 = nn.Sequential(nn.ConvTranspose2d(num_GenFeatureMaps, num_OutputChannels, kernel_size=4, stride=2, padding=1, bias= False),
                                    nn.Tanh())

    def forward(self, x):
        output = self.ConvG1(x)
        output = self.ConvG2(output)
        output = self.ConvG3(output)
        output = self.ConvG4(output)
        return output


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model  = Gen(device, num_GenFeatureMaps= 64, num_OutputChannels= 3, latent_vector= 100)
print(model)