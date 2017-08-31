# Conv2D(in_channel, o_channel, kernel_size, stride, mode)
# [int, 3D FloatTensor] Conv2D.forward(input_image)

import numpy as np
import torch
import math
from PIL import Image
import torchvision.transforms as transforms

class Conv2D():
    def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
        self.in_c = in_channel
        self.o_c = o_channel
        self.k_size = kernel_size
        self.stride = stride
        self.K1 = torch.IntTensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        self.K2 = torch.transpose(self.K1, 0, 1)
        self.K3 = torch.IntTensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self.K4 = torch.IntTensor(([[-1] * self.k_size]) * (int)((self.k_size - 1)/2) \
                               + ([[0] * self.k_size]) \
                               + ([[-1] * self.k_size]) * (int)((self.k_size - 1)/2))
        self.K5 = torch.transpose(self.K4, 0, 1)
        if mode == "rand":
            self.kernel = torch.randn(self.o_c, self.k_size, self.k_size)
        else:
            if(self.o_c == 1):
                self.kernel = self.K1.unsqueeze(0)
            elif(self.o_c == 2):
                self.kernel = torch.cat((self.K4.unsqueeze(0), self.K5.unsqueeze(0)), 0)
            else:
                self.kernel = torch.cat((self.K1.unsqueeze(0), self.K2.unsqueeze(0), \
                                         self.K3.unsqueeze(0)), 0)
        
    def forward(self, input_image):
        i_c = input_image.size(0)
        i_h = input_image.size(1)
        i_w = input_image.size(2)
        
        o_h = math.ceil((i_h - self.k_size + 1)/self.stride)
        o_w = math.ceil((i_w - self.k_size + 1)/self.stride)
        
        no_ops = 0
        output_FT = torch.randn(self.o_c, o_h, o_w)
        for i in range(self.o_c):
           for j in range(o_h):
               for k in range(o_w):
                   h_start = j * self.stride
                   w_start = k * self.stride
                   no_ops += 1
                   
                   for h in range(self.k_size):
                       for w in range(self.k_size):
                           for c in range(i_c):
                               output_FT[i][j][k] += input_image[c][h_start + h][w_start + w] * self.kernel[i][h][w] 
           output_FT[i] = (output_FT[i] - output_FT[i].min()) / (output_FT[i].max() - output_FT[i].min())                    
        return no_ops, output_FT 
