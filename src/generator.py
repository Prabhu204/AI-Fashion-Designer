# -*- coding: utf-8 -*-
"""
@author: Prabhu <prabhu.appalapuri@gmail.com>
"""

import torch
import torch.nn as nn

# latent space_vectors --> size of the input (here considered as 100)
# num_Outputchannels --> orignal image channels (i.e 3 for RGB, 1 for Gray scale)
# Generator
class Gen(nn.Module):
    def __init__(self, num_GenFeatureMaps, vector_size, num_OutputChannels):
        super(Gen, self).__init__()
        self.num_GenFeatureMaps  =num_GenFeatureMaps
        self.vector_size = vector_size
        self.num_OutputChannels = num_OutputChannels
        self.ConvG1= nn.Sequential(nn.ConvTranspose2d(vector_size,num_GenFeatureMaps*8, kernel_size=4,stride=1, padding=0,bias=False),
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
        self.ConvG5 = nn.Sequential(nn.ConvTranspose2d(num_GenFeatureMaps, num_OutputChannels, kernel_size=4, stride=2, padding=1, bias= False),
                                    nn.Tanh())    # tanh function returns input data range [-1, 1]

    def forward(self, x):
        x = x.view(-1,self.vector_size,1,1)
        output = self.ConvG1(x)
        output = self.ConvG2(output)
        output = self.ConvG3(output)
        output = self.ConvG4(output)
        output = self.ConvG5(output)
        return output
