# -*- coding: utf-8 -*-
"""
@author: Prabhu <prabhu.appalapuri@gmail.com>
"""

from torch import nn

#  Discriminator
# input_channel will be 3 for RGB, 1 for Gray scale)
class Disc(nn.Module):
    def __init__(self, num_Channels, num_DisFeaturesMaps, vector_size):
        super(Disc,self).__init__()
        self.num_DisFeaturesMaps = num_DisFeaturesMaps
        self.num_Channels = num_Channels
        self.vector_size = vector_size
        self.ConvD1 = nn.Sequential(nn.Conv2d(num_Channels, num_DisFeaturesMaps, kernel_size=4,stride=2, padding=1, bias= False),
                                    nn.LeakyReLU(0.02, inplace=True))
        self.ConvD2 = nn.Sequential(nn.Conv2d(num_DisFeaturesMaps, num_DisFeaturesMaps*2, kernel_size=4, stride=2, padding=1,bias=False),
                                    nn.BatchNorm2d(num_DisFeaturesMaps*2),
                                    nn.LeakyReLU(0.02, inplace= True))
        self.ConvD3 = nn.Sequential(nn.Conv2d(num_DisFeaturesMaps*2, num_DisFeaturesMaps*4, kernel_size=4, stride=2,padding=1,bias= False),
                                    nn.BatchNorm2d(num_DisFeaturesMaps*4),
                                    nn.LeakyReLU(0.02, inplace= True))
        self.ConvD4 = nn.Sequential(nn.Conv2d(num_DisFeaturesMaps * 4, num_DisFeaturesMaps * 8, kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(num_DisFeaturesMaps * 8),
                                    nn.LeakyReLU(0.02, inplace=True))
        self.ConvD5 = nn.Sequential(nn.Conv2d(num_DisFeaturesMaps * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
                                    nn.Sigmoid())

    def forward(self, x):
        output = self.ConvD1(x)
        output = self.ConvD2(output)
        output = self.ConvD3(output)
        output = self.ConvD4(output)
        output = self.ConvD5(output)
        return output