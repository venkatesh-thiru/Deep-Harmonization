import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "GPU-8ab9a0c8-909c-3f13-97e6-7376d6d4a029"
import torch
import torch.nn.functional as F
from Dataset import s2l8hDataset
from skimage.metrics import normalized_root_mse,peak_signal_noise_ratio,structural_similarity
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import json
from models.AttenUNet.AUNet import AUNet
import segmentation_models_pytorch as smp
import torch.nn as nn
import torchmetrics
import math


def gaussian_nll_loss(input, target, variance, full = True, eps = 1e-6, distance = "mae" ):
    variance = variance.clone()
    with torch.no_grad():
        variance.clamp_(min = eps)
    if distance == "mae":
        loss = 0.5 * (torch.log(variance) + torch.abs(input - target)/variance)
    elif distance == "mse":
        loss = 0.5 * (torch.log(variance) + (input - target)**2/variance)
    else:
        raise ValueError("expected [mae or mse] for distance")
    if full:
        loss += 0.5 * math.log(2 * math.pi)

    return loss.mean()


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
    def __init__(self, in_channel, out_channel, ca_layer, infer_mode):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1))
        block.append(nn.PReLU())

        block.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1))
        block.append(nn.PReLU())

        if ca_layer:
            block.append(ECA(out_channel))
        if not infer_mode is None:
            block.append(nn.Dropout(p = 0.1))
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
    def __init__(self, in_channel, out_channel, upmode, ca_layer, up_factor = 2, att_mode = "standard", infer_mode = None):
        super(UNetUpConvBlock, self).__init__()
        self.att_mode = att_mode
        self.ca_layer = ca_layer
        self.infer_mode = infer_mode
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
            self.conv = UNetConvBlock(in_channel, out_channel, ca_layer=self.ca_layer, infer_mode = self.infer_mode)
        elif self.att_mode == 'modified':
            self.attention_gate = AttentionGate(out_channel, out_channel, out_channel )
            self.conv = UNetConvBlock(3*out_channel, out_channel, ca_layer = self.ca_layer, infer_mode = self.infer_mode)
        elif self.att_mode == 'None':
            self.conv = UNetConvBlock(in_channel, out_channel, ca_layer=self.ca_layer, infer_mode = self.infer_mode)



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
                 interp_mode = 'bilinear', up_mode = 'upconv',spatial_attention = "standard", ca_layer = True,
                infer_mode = None):
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
        self.infer_mode = infer_mode
        self.encoding_module = nn.ModuleList()
        for i in range(self.depth):
            self.encoding_module.append(UNetConvBlock(in_channel=prev_channels,out_channel=2**(self.growth_factor + i), ca_layer=self.ca_layer, infer_mode = self.infer_mode))
            prev_channels = 2**(self.growth_factor+i)

        self.decoding_module = nn.ModuleList()
        for i in reversed(range(self.depth-1)):
            self.decoding_module.append(UNetUpConvBlock(prev_channels,2**(self.growth_factor+i),upmode = self.up_mode, att_mode = self.att_mode, ca_layer = self.ca_layer,infer_mode = self.infer_mode))
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
    
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

def make_mc_dropout_prediction(model, batch, M, pre_interpolated = True):
    s2_batch, l8_batch, l8_pan = batch['s2_img'].cuda(), batch['l8_img'].cuda(), batch['l8_pan'].cuda()
    if pre_interpolated:
        l8_batch = torch.nn.functional.interpolate(l8_batch, size=s2_batch.shape[2:], mode='bicubic')
        l8_pan = torch.nn.functional.interpolate(l8_pan, size=s2_batch.shape[2:], mode='bicubic')
    logits = [model(l8_batch, l8_pan).detach() for i in tqdm(range(M), disable=True)]
    logits = [torch.nn.functional.interpolate(logit, size = s2_batch.shape[2:], mode="nearest") for logit in logits]
    logits = torch.cat(logits, dim = 0)
    variance, prediction = torch.var_mean(logits, dim = 0, keepdim = True)
    return s2_batch, l8_batch, logits, prediction, variance

def generate_uncertainty_results(model, test_loader, model_name):
    uncertainty_metrics = {}
    M = 500
    for i, batch in enumerate(tqdm(test_loader)):
        # SC = torchmetrics.SpearmanCorrCoef().cuda()
        PC = torchmetrics.PearsonCorrCoef().cuda()
        gnll = torch.nn.GaussianNLLLoss(full = True)
        uncertainty_metrics_per_image = {}
        patch_name = os.path.split(batch['patch_path'][0])[-1][:-3]
        s2_batch, l8_batch, logits, prediction, variance = make_mc_dropout_prediction(model, batch, M, pre_interpolated=True)
        abs_diff = torch.abs(s2_batch-prediction) #L1 Loss

        uncertainty_metrics_per_image['SSIM'] =  structural_similarity(s2_batch.detach().squeeze().cpu().numpy(), prediction.detach().squeeze().cpu().numpy(),channel_axis=0,data_range=1).astype(float)
        # uncertainty_metrics_per_image['SC'] = SC(variance.detach().view(-1), abs_diff.detach().view(-1)).item()
        uncertainty_metrics_per_image['PC'] = PC(variance.detach().view(-1), abs_diff.detach().view(-1)).item()
        uncertainty_metrics_per_image['NRMSE'] = normalized_root_mse(s2_batch.detach().squeeze().cpu().numpy(), prediction.detach().squeeze().cpu().numpy()).astype(float)
        uncertainty_metrics_per_image['GNLL'] = gnll(prediction.detach().cpu(), s2_batch.detach().cpu(), variance.detach().cpu()).item()
        
        uncertainty_metrics[patch_name] = uncertainty_metrics_per_image

    with open(f"/data_fast/venkatesh/s2l8h/uncertainty_metrics/{model_name}_20_dropout_500_samples.json","w") as outfile:
        json.dump(uncertainty_metrics, outfile, indent = 4)


def load_model_weights(parameters, weight_path):
    model = AUNet(**parameters)
    enable_dropout(model)
    weights = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(weights['model_state_dict'])
    return model.cuda()


model_zoo = {
       "5_upsample":{"model_parameters":{"in_channels":7, "out_channels":6, "depth":5, 
                                    "spatial_attention":"None", "growth_factor":6,
                                    "interp_mode":"bicubic", "up_mode":"upsample",
                                    "ca_layer":False, "infer_mode":"dropout"},
                     "model_weight":"/data_fast/venkatesh/s2l8h/training/model_checkpoints/AttUNets/5 depth/None/AUNet - standard No SA and No CA Scaler 5 depth pixel upsample.pth.pth"},
       "5_shuffle":{"model_parameters":{"in_channels":7, "out_channels":6, "depth":5, 
                                    "spatial_attention":"None", "growth_factor":6,
                                    "interp_mode":"bicubic", "up_mode":"shuffle",
                                    "ca_layer":False, "infer_mode":"dropout"},
                     "model_weight":"/data_fast/venkatesh/s2l8h/training/model_checkpoints/AttUNets/5 depth/None/AUNet - standard No SA and No CA Scaler 5 depth pixel shuffle.pth.pth"},
       "6_upsample":{"model_parameters":{"in_channels":7, "out_channels":6, "depth":6, 
                                    "spatial_attention":"None", "growth_factor":6,
                                    "interp_mode":"bicubic", "up_mode":"upsample",
                                    "ca_layer":False, "infer_mode":"dropout"},
                     "model_weight":"/data_fast/venkatesh/s2l8h/training/model_checkpoints/AttUNets/6 depth/AUNet - No CA No SA 6 depth pixel upsample.pth"},
       "6_shuffle":{"model_parameters":{"in_channels":7, "out_channels":6, "depth":6, 
                                    "spatial_attention":"None", "growth_factor":6,
                                    "interp_mode":"bicubic", "up_mode":"shuffle",
                                    "ca_layer":False, "infer_mode":"dropout"},
                     "model_weight":"/data_fast/venkatesh/s2l8h/training/model_checkpoints/AttUNets/6 depth/AUNet - No CA No SA 6 depth pixel shuffle.pth"},
}
# models = [load_model_weights(model_zoo[key]['model_parameters'], model_zoo[key]['model_weight']) for key in model_zoo.keys()]


test_dataset = s2l8hDataset(
       csv_path = "/data_fast/venkatesh/s2l8h/train_test_validation_patch_extended_final.csv",
       patch_dir = "/data_fast/venkatesh/s2l8h/DATA_L2_cloudless_patches",
       type = "Test",
       BOA = True,
       s2_array_size = 256,
       l8_array_size = 86,
       band_pass = False
)
test_loader = DataLoader(test_dataset,batch_size = 1, num_workers=8)

model_name = "6_shuffle"
model = load_model_weights(model_zoo[model_name]['model_parameters'], model_zoo[model_name]['model_weight'])

# generate_uncertainty_results(model, test_loader, model_name)

