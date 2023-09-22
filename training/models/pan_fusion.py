import torch
import torch.nn as nn
from .helpers import get_edge,get_highpass
import numpy as np
from torch.nn.functional import interpolate

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size = 3,stride = 1, padding = 1, withBN = True):
        super(ResBlock, self).__init__()
        self.block = []
        self.block.append(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding))
        if withBN:
            self.block.append(nn.BatchNorm2d(out_channels))
        self.block.append(nn.ReLU(inplace = True))
        self.block.append(nn.Conv2d(out_channels,out_channels,kernel_size,stride,padding))
        if withBN:
            self.block.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*self.block)

    def forward(self,x):
        return self.block(x) + x

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6,32,3,1,1),
            nn.LeakyReLU(0.2,True),
            #32x256x256
            nn.Conv2d(32,32,4,2,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(32,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True),
            #64x128x128
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            #128x64x64
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            #256x32x32
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            #512x16x16
            nn.Conv2d(512,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*8*8,100),
            nn.LeakyReLU(0.2,True),
            nn.Linear(100,1)
        )

    def forward(self,X):
        out = self.features(X)
        out = torch.flatten(out,1)
        out = self.classifier(out)
        return out


class PANF_Generator(nn.Module):
    def __init__(self,withBN = True,high_pass = False, res_layer = 3, Norm = 'BN',activation = 'LReLU'):
        super(PANF_Generator,self).__init__()
        self.high_pass = high_pass
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "PReLU":
            self.activation = nn.PReLU()
        elif activation == "LReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        # Low Resolution Multi Spectral Feature Extractor
        self.FE_LRMS = []
        self.FE_LRMS.append(nn.Conv2d(6,32,7,1,3)) #LANDSAT-MS => 32x86x86(30m)
        if withBN:
            self.FE_LRMS.append(nn.BatchNorm2d(32))
        self.FE_LRMS.append(self.activation)
        self.FE_LRMS.append(nn.Conv2d(32,64,3,1,1)) #LANDSAT-MS => 64x86x86(30m)
        if withBN:
            self.FE_LRMS.append(nn.BatchNorm2d(64))
        self.FE_LRMS.append(self.activation)
        self.FE_LRMS.append(nn.Conv2d(64,128,3,1,1)) #LANDSAT-MS => 128x86x86(30m)
        if withBN:
            self.FE_LRMS.append(nn.BatchNorm2d(128))
        self.FE_LRMS = nn.Sequential(*self.FE_LRMS)

        # Panchromatic Feature Extractor
        self.FE_pan = []
        self.FE_pan.append(nn.Conv2d(1,32,7,1,3)) #LANDSAT-PAN => 32x172x172
        if withBN:
            self.FE_pan.append(nn.BatchNorm2d(32))
        self.FE_pan.append(self.activation)
        self.FE_pan.append(nn.Conv2d(32,64,3,2,1)) #LANDSAT-PAN => 64x86x86
        if withBN:
            self.FE_pan.append(nn.BatchNorm2d(64))
        self.FE_pan.append(self.activation)
        self.FE_pan.append(nn.Conv2d(64,128,3,1,1)) #LANDSAT-PAN => 128x86x86
        if withBN:
            self.FE_pan.append(nn.BatchNorm2d(128))
        self.FE_pan = nn.Sequential(*self.FE_pan)

        #RESNET CONCAT
        self.res=[]
        for _ in range(res_layer):
            self.res.append(self.activation)
            self.res.append(ResBlock(256,256,3,1,1,withBN)) #128x86x86
        self.res.append(self.activation)

        self.res.append(nn.Upsample(mode = 'bilinear',scale_factor = 2))
        self.res.append(nn.Conv2d(256,256,1,1,0))
        self.res.append(nn.Conv2d(256,128,3,1,1)) #128x172x172
        if withBN:
            self.res.append(nn.BatchNorm2d(128))
        self.res.append(self.activation)

        self.res.append(nn.Upsample(mode = 'bilinear',scale_factor = 1.5))
        self.res.append(nn.Conv2d(128,128,1,1,0))
        self.res.append(nn.Conv2d(128,64,3,1,1)) #64x258x258
        if withBN:
            self.res.append(nn.BatchNorm2d(64))
        self.res.append(self.activation)
        self.res.append(nn.Conv2d(64,6,7,1,3))  #6x258x258
        self.res = nn.Sequential(*self.res)

    def forward(self,pan,lr):
        if self.high_pass:
            # lr_edge = get_edge(lr)
            pan_edge = get_highpass(pan)
            lr_feat = self.FE_LRMS(lr)
            pan_feat = self.FE_pan(pan_edge)
            res = self.res(torch.cat([lr_feat,pan_feat]),dim = 1)
        else:
            lr_feat = self.FE_LRMS(lr)
            pan_feat = self.FE_pan(pan)
            res = self.res(torch.cat((lr_feat,pan_feat),dim = 1))
        return res


if __name__ == '__main__':
    # GEN = PANF_Generator(withBN = True,res_layer = 3,activation = 'ReLU')
    # pan = torch.rand(size = [1,1,172,172])
    # l8_ms = torch.rand(size = [1,6,86,86])
    # l8_u = torch.rand(size = [1,6,258,258])
    # op = GEN(pan,l8_ms,l8_u)
    din = torch.rand(size = [1,6,256,256])
    d = discriminator()
    dlog = d(din)
    print(dlog)
