import torch
import torch.nn as nn

from .hpb import HPB, Config
from .et import EfficientTransformer
from torchsummary import summary


class BackBoneBlock(nn.Module):
    def __init__(self, num, fm, **args):
        super().__init__()
        self.arr = nn.ModuleList([])
        for _ in range(num):
            self.arr.append(fm(**args))

    def forward(self, x):
        for block in self.arr:
            x = block(x)
        return x


class ESRT(nn.Module):
    def __init__(self, in_channels = 7, out_channels = 6, hiddenDim=32, mlpDim=128, scaleFactor=2, num_heads = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv3 = nn.Conv2d(self.in_channels, hiddenDim,
                               kernel_size=3, padding=1)
        
        lamRes = torch.nn.Parameter(torch.ones(1))
        lamX = torch.nn.Parameter(torch.ones(1))
        self.adaptiveWeight = (lamRes, lamX)

        self.path1 = nn.Sequential(
            BackBoneBlock(self.in_channels, HPB, inChannel=hiddenDim,
                          outChannel=hiddenDim, reScale=self.adaptiveWeight),
            BackBoneBlock(1, EfficientTransformer,
                          mlpDim=mlpDim, inChannels=hiddenDim, heads = num_heads),
            nn.Conv2d(hiddenDim, hiddenDim, kernel_size=3, padding=1),
            nn.PixelShuffle(scaleFactor),
            nn.Conv2d(hiddenDim // (scaleFactor**2),
                      self.out_channels, kernel_size=3, padding=1),
        )

        self.path2 = nn.Sequential(
            nn.PixelShuffle(scaleFactor),
            nn.Conv2d(hiddenDim // (scaleFactor**2),
                      self.out_channels, kernel_size=3, padding=1),
        )

    def forward(self, MS, PAN = None):
        if PAN is None:
            x = MS
        else:
            x = torch.cat([MS,PAN], dim=1)
        x = self.conv3(x)
        x1, x2 = self.path1(x), self.path2(x)
        return x1 + x2


if __name__ == '__main__':
    # x = torch.tensor([float(i+1)
    #                  for i in range(2*7*128*128)]).reshape((2, 7, 128, 128)).cuda()

    model = ESRT(mlpDim=128, scaleFactor=1,hiddenDim=16, num_heads = 4).cuda()
    summary(model, input_size=(7,128,128))
    # y = model(x)
    # print(y.shape)
