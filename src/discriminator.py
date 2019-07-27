# -*- coding: utf-8 -*-
"""
@author: Prabhu <prabhu.appalapuri@gmail.com>
"""

from torch import nn

#  Discriminator
class Dis(nn.Module):
    def __init__(self, num_DisFeaturesMaps):
        self.num_DisFeaturesMaps = num_DisFeaturesMaps
