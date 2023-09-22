import torch
import torch.nn.functional as F
import torch.nn as nn


class UNetConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1))
        block.append(nn.PReLU())

        block.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1))
        block.append(nn.PReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)

        return out


class UNetUpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upmode,up_factor = 2):
        super(UNetUpConvBlock, self).__init__()

        if upmode == 'upsample':
            self.Upsize = nn.Sequential(
                                        nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False),
                                        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
                                      )
        elif upmode == 'upconv':
            self.Upsize = nn.ConvTranspose2d(in_channel,out_channel,kernel_size=2,stride = 2)

        self.conv = UNetConvBlock(in_channel, out_channel)

    def forward(self, x, residue):
        x = self.Upsize(x)
        x = F.interpolate(x, size=residue.shape[2:], mode='bilinear')
        out = torch.cat([x, residue], dim=1)
        out = self.conv(out)
        return out


class SRUNet(nn.Module):
    def __init__(self,in_channels = 6,out_channels = 6,depth = 3,growth_factor = 6,
                 interp_mode = 'bilinear',SR_model = False, up_mode = 'upsample'):
        super(SRUNet,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.growth_factor = growth_factor
        self.interp_mode = interp_mode
        prev_channels = self.in_channels
        self.encoding_module = nn.ModuleList()
        self.upmode = up_mode

        for i in range(self.depth):
            self.encoding_module.append(UNetConvBlock(in_channel=prev_channels,out_channel=2**(self.growth_factor + i)))
            prev_channels = 2**(self.growth_factor+i)

        self.decoding_module = nn.ModuleList()

        for i in reversed(range(self.depth-1)):
            self.decoding_module.append(UNetUpConvBlock(prev_channels,2**(self.growth_factor+i),upmode = 'upconv'))
            prev_channels = 2**(self.growth_factor+i)

        if SR_model:
            self.final = nn.ModuleList()
            final = []
            final.append(nn.Upsample(scale_factor=3, mode='bilinear', align_corners=False))
            final.append(nn.Conv2d(prev_channels, prev_channels, kernel_size=1, stride=1, padding=0))
            final.append(nn.Conv2d(prev_channels, prev_channels,kernel_size = 3, padding = 0))
            final.append(nn.PReLU())
            final.append(nn.Conv2d(prev_channels,out_channels,kernel_size = 1))
            self.final = nn.Sequential(*final)
        else:
            self.final = nn.Conv2d(prev_channels,out_channels,1,1,0)

    def forward(self,MS,PAN = None):
        if PAN == None:
            x = MS
        else:
            x = torch.cat([MS,PAN],dim = 1)
        blocks = []
        for i,down in enumerate(self.encoding_module):
            x = down(x)
            if i != len(self.encoding_module)-1:
                blocks.append(x)
                x = F.avg_pool2d(x,2)

        for i,up in enumerate(self.decoding_module):
            x = up(x,blocks[-i-1])

        x = self.final(x)
        return x



if __name__ == '__main__':
    model = SRUNet(in_channels=7,out_channels=6,depth=3,SR_model=False,up_mode='upconv')
    print(model)
    dimensions = 1, 6, 256, 256
    ms = torch.randn(dimensions)
    dimensions = 1, 1, 256, 256
    pan = torch.randn(dimensions)
    x = torch.cat([ms,pan],dim = 1)
    x = model(x)
    print(x.shape)
