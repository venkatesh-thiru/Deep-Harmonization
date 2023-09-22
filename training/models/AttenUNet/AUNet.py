import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqeezeExcite(nn.Module):
    def __init__(self, channel, reduction_ratio = 16):
        super(SqeezeExcite,self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channel, channel//reduction_ratio, bias = False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction_ratio,channel,bias = False),
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c,_,_ = x.size()
        out = self.GAP(x).view(b,c)
        out = self.mlp(out).view(b,c,1,1)
        return x * out.expand_as(x)


class ECA(nn.Module):
    # https://wandb.ai/diganta/ECANet-sweep/reports/Efficient-Channel-Attention--VmlldzozNzgwOTE
    def __init__(self,channels, b = 1, gamma = 2):
        super(ECA, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(1, 1, kernel_size=self.adaptive_kernel(),padding = (self.adaptive_kernel()-1)//2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        attn = self.GAP(x)
        attn = self.conv(attn.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        attn = self.sigmoid(attn)
        return x * attn.expand_as(x)


    def adaptive_kernel(self):
        k = int(abs(math.log2(self.channels)/self.gamma) + self.b)
        ksize = k if k%2 else k+1
        return ksize



class UNetConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, ca_layer):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1))
        block.append(nn.PReLU())

        block.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1))
        block.append(nn.PReLU())

        if ca_layer:
            block.append(ECA(out_channel))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, dimensions):
        super(AttentionGate, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, dimensions, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(dimensions)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, dimensions, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(dimensions)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(dimensions, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.PReLU()

    def forward(self, g, x):
        g1 = self.W_gate(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class UNetUpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upmode, ca_layer, up_factor = 2, att_mode = "standard"):
        super(UNetUpConvBlock, self).__init__()
        self.att_mode = att_mode
        self.ca_layer = ca_layer
        if upmode == 'upsample':
            self.Upsize = nn.Sequential(
                                        nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False),
                                        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
                                      )
        elif upmode == 'upconv':
            self.Upsize = nn.ConvTranspose2d(in_channel,out_channel,kernel_size=2,stride = 2)
        elif upmode == 'shuffle':
            self.Upsize = nn.Sequential(
                                        nn.Conv2d(in_channel,out_channel*4,kernel_size=3,stride=1,padding=1),
                                        nn.PReLU(),
                                        nn.PixelShuffle(2),
                                        nn.Conv2d(out_channel,out_channel,kernel_size=3,stride = 1,padding=1)
                                        )


        # self.conv = UNetConvBlock(in_channel, out_channel)
        if self.att_mode == 'standard':
            self.attention_gate = AttentionGate(out_channel, out_channel, out_channel)
            self.conv = UNetConvBlock(in_channel, out_channel, ca_layer=self.ca_layer)
        elif self.att_mode == 'modified':
            self.attention_gate = AttentionGate(out_channel, out_channel, out_channel )
            self.conv = UNetConvBlock(3*out_channel, out_channel, ca_layer = self.ca_layer)
        elif self.att_mode == 'None':
            self.conv = UNetConvBlock(in_channel, out_channel, ca_layer=self.ca_layer)



    def forward(self, x, residue):
        x = self.Upsize(x)
        x = F.interpolate(x, size=residue.shape[2:], mode='bilinear')
        if self.att_mode == "standard":
            attn = self.attention_gate(g = x, x=residue)
            out = torch.cat([x, attn],dim = 1)
            out = self.conv(out)
        elif self.att_mode == 'modified':
            attn = self.attention_gate(g = x, x = residue)
            out = torch.cat([x,residue,attn],dim = 1)
            out = self.conv(out)
        elif self.att_mode == 'None':
            out = torch.cat([x,residue], dim = 1)
            out = self.conv(out)
        return out





class AUNet(nn.Module):
    def __init__(self,in_channels = 6,out_channels = 6,depth = 3,growth_factor = 6,
                 interp_mode = 'bilinear', up_mode = 'upconv',spatial_attention = "standard", ca_layer = True):
        super(AUNet,self).__init__()

        if not spatial_attention in ['None', 'modified', 'standard']:
            raise AssertionError("spatial_attention options : \'None\'- no spatial attention, \'standard\'-spatial attention as in attention unet paper, \'modified\'-modified attention unet")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.growth_factor = growth_factor
        self.interp_mode = interp_mode
        prev_channels = self.in_channels
        self.up_mode = up_mode
        self.att_mode = spatial_attention
        self.ca_layer = ca_layer

        self.encoding_module = nn.ModuleList()
        for i in range(self.depth):
            self.encoding_module.append(UNetConvBlock(in_channel=prev_channels,out_channel=2**(self.growth_factor + i), ca_layer=self.ca_layer))
            prev_channels = 2**(self.growth_factor+i)

        self.decoding_module = nn.ModuleList()
        for i in reversed(range(self.depth-1)):
            self.decoding_module.append(UNetUpConvBlock(prev_channels,2**(self.growth_factor+i),upmode = self.up_mode, att_mode = self.att_mode, ca_layer = self.ca_layer))
            prev_channels = 2**(self.growth_factor+i)

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
    x = torch.rand([9,7,256,256]).cuda()
    model = AUNet(in_channels=7, out_channels=6, depth=5, spatial_attention="modified", growth_factor=6,
                  interp_mode='bilinear', up_mode='upconv', ca_layer=True).cuda()
    x = model(x)
    # print(model)


    activation = {}
    for layer in model:
        print(layer)

