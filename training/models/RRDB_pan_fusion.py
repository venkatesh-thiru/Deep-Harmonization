import torch
import torch.nn as nn
import torch.nn.functional as F
from models.helpers import get_highpass

class make_dense(nn.Module):
    def __init__(self,nChannels,GrowthRate,kernel_size=3):
        super(make_dense,self).__init__()
        self.conv = nn.Conv2d(nChannels,GrowthRate,kernel_size=kernel_size,padding=(kernel_size-1)//2,bias=True)
        # self.norm = nn.BatchNorm2d(nChannels)
    def forward(self,x):
        # out = self.norm(x)
        out = F.relu(self.conv(x))
        out = torch.cat([x,out],dim=1)
        return out

class RDB(nn.Module):
    def __init__(self,inChannels,outChannels,nDenseLayer,GrowthRate,KernelSize = 3):
        super(RDB,self).__init__()
        nChannels_ = inChannels
        modules = []
        for i in range (nDenseLayer):
            modules.append(make_dense(nChannels_,GrowthRate,kernel_size=KernelSize))
            nChannels_ += GrowthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_,outChannels,kernel_size=1,padding=0,bias = False)
    def forward(self,x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        # out = out + x
        return out

class RRDB_pan_guided(nn.Module):
    def __init__(self,nChannels,nDenseLayers,nInitFeat,GrowthRate,featureFusion=True,
                 kernel_config = [3,3,3,3],pan_highpass = False,pan_loss = False):
        super(RRDB_pan_guided,self).__init__()
        nChannels_ = nChannels
        nDenseLayers_ = nDenseLayers
        nInitFeat_ = nInitFeat
        GrowthRate_ = GrowthRate
        self.featureFusion = featureFusion
        self.pan_highpass = pan_highpass
        self.pan_loss = pan_loss

        #MS BRANCH
        self.C1 = nn.Conv2d(nChannels_, nInitFeat_, kernel_size=kernel_config[0], padding=(kernel_config[0] - 1) // 2,
                            bias=True)
        self.RDB1 = RDB(nInitFeat_, nInitFeat_, nDenseLayers_, GrowthRate_, kernel_config[1])
        self.RDB2 = RDB(nInitFeat_ * 2, nInitFeat_, nDenseLayers_, GrowthRate_, kernel_config[2])
        self.RDB3 = RDB(nInitFeat_ * 3, nInitFeat_, nDenseLayers_, GrowthRate_, kernel_config[3])

        #PAN_BRANCH
        self.PANC1 = nn.Conv2d(1 , nInitFeat_, kernel_size=kernel_config[0], padding=(kernel_config[0] - 1) // 2,
                            bias=True)
        self.PRDB1 = RDB(nInitFeat_, nInitFeat_, nDenseLayers_, GrowthRate_, kernel_config[1])
        self.PRDB2 = RDB(nInitFeat_* 2, nInitFeat_, nDenseLayers_, GrowthRate_, kernel_config[2])
        self.PRDB3 = RDB(nInitFeat_* 3, nInitFeat_, nDenseLayers_, GrowthRate_, kernel_config[3])

        #FEATURES FUSION LAYER
        self.FF_1X1 = nn.Conv2d(nInitFeat_ * 8, nChannels_, kernel_size=1, padding=0, bias=True)
        self.PANRES = nn.Conv2d(nInitFeat_ * 4, 1, kernel_size=1,padding = 0,bias = True)


    def forward(self,MS,PAN):
        #MS FORWARD
        MS_FIRST = self.C1(MS)
        MS_FIRST = F.relu(MS_FIRST)
        MS_R_1 = self.RDB1(MS_FIRST)

        MS_FF0 = torch.cat([MS_FIRST,MS_R_1],dim = 1)
        MS_R2 = self.RDB2(MS_FF0)

        MS_FF1 = torch.cat([MS_FIRST,MS_R_1,MS_R2],dim = 1)
        MS_R3 = self.RDB3(MS_FF1)

        MS_FF2 = torch.cat([MS_FIRST,MS_R_1,MS_R2,MS_R3],dim = 1)

        #PAN FORWARD
        PAN_FIRST = self.PANC1(PAN)
        PAN_FIRST = F.relu(PAN_FIRST)
        PAN_R_1 = self.PRDB1(PAN_FIRST)

        PAN_FF0 = torch.cat([PAN_FIRST, PAN_R_1], dim=1)
        PAN_R2 = self.PRDB2(PAN_FF0)

        PAN_FF1 = torch.cat([PAN_FIRST, PAN_R_1, PAN_R2], dim=1)
        PAN_R3 = self.PRDB3(PAN_FF1)

        PAN_FF2 = torch.cat([PAN_FIRST, PAN_R_1, PAN_R2, PAN_R3], dim=1)

        GFF = torch.cat([MS_FF2,PAN_FF2],dim = 1)
        MS_RES = self.FF_1X1(GFF)


        if self.pan_loss:
            if self.training:
                PAN_RES = self.PANRES(PAN_FF2)
                if self.pan_highpass:
                    PAN_RES = get_highpass(PAN_RES)
                return MS_RES,PAN_RES
            else:
                return MS_RES
        else:
            return MS_RES



if __name__ == '__main__':
    MS = torch.randn([10,6,256,256]).cuda()
    PAN = torch.randn([10,1,256,256]).cuda()
    model = RRDB_pan_guided(nChannels=6, nDenseLayers=6, nInitFeat=6, GrowthRate=12, featureFusion=True,
                 kernel_config=[3, 3, 3, 3],pan_highpass=True,pan_loss=True).cuda()
    model.train()
    X,P = model(MS,PAN)
    print(X[:,2,...].shape,P.shape)