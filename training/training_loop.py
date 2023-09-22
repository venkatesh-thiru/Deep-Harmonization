import os
import wandb
import random
from tqdm import tqdm
import statistics
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.cuda.amp import autocast,GradScaler
from models.UNet import SRUNet
from models.RRDB_pan_fusion import RRDB_pan_guided
# from models.ESRT.ESRT import ESRT
from models.ESRT2.models import ESRT
from models.helpers import get_highpass
from Dataset import s2l8hDataset
from loss.mgl import mixed_gradient_error
from inference_plot import make_fig
from config import get_cfg_defaults
from pytorch_msssim import SSIM
import segmentation_models_pytorch as smp
from models.AttenUNet.AUNet import AUNet
from models.AttenUNet.AUNet_Gaussian import AUNet_Gaussian
import math


torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


# from acquisition_preprocess.utils import *


def gaussian_nll_loss(input, target, variance, full = True, eps = 1e-6, distance = "mae" ):
    variance = variance.clone()

    with torch.no_grad():
        variance.clamp_(min = eps)
            # print(variance.mean())
    if distance == "mae":
        loss = 0.5 * (torch.log(variance) + torch.abs(input - target)/variance)
    elif distance == "mse":
        loss = 0.5 * (torch.log(variance) + (input - target)**2/variance)
    else:
        raise ValueError("expected [mae or mse] for distance")
    if full:
        loss += 0.5 * math.log(2 * math.pi)
    return loss.mean()



def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

def calculate_loss(s2_batch,logit):
    return loss_function(s2_batch, logit)

def write_plot(image_dict, epoch):
    fig = make_fig(image_dict,epoch)
    wandb.log({f"EPOCH {epoch}" : fig})
    print("figure added")

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def test_network(epoch):
    sample = random.choice(validation_dataset)
    model.eval()
    l8_batch,s2_batch,l8_pan = torch.from_numpy(np.expand_dims(sample['l8_img'],axis = 0)).cuda(),sample['s2_img'],torch.from_numpy(np.expand_dims(sample['l8_pan'],axis = 0)).cuda()
    # l8_u = F.interpolate(l8_img,size = [258,258],mode = 'bilinear')
    if pre_interpolate:
        l8_in = F.interpolate(l8_batch, size=s2_batch.shape[1:], mode='bicubic')
        l8_pan_in= F.interpolate(l8_pan, size=s2_batch.shape[1:], mode='bicubic')
        if pan_highpass:
            l8_pan_in = get_highpass(l8_pan_in)
    else:
        l8_in = l8_batch
        l8_pan_in = l8_pan
    if model_kind == "SMP_UNet_mit":
        prediction = model(torch.cat([l8_in,l8_pan_in], dim=1))
    elif model_kind == "AUNet_Gaussian":
        prediction, variance = model(l8_in, l8_pan_in)
        variance = variance.detach().squeeze().cpu().numpy()
    else:        
        prediction = model(l8_in,l8_pan_in if cfg.fuse_pan else None)
    l8_img,l8_pan,prediction = l8_batch.detach().squeeze().cpu().numpy(),l8_pan.detach().squeeze().cpu().numpy(),prediction.detach().squeeze().cpu().numpy()
    img_dict = {"l8":l8_img,"l8_pan":l8_pan,"s2":s2_batch,"pred":prediction}
    write_plot(img_dict,epoch)

def validation_loop():
    print("validating...........")
    overall_validation_loss = []
    model.eval()
    for batch in validation_loader:
        s2_batch,l8_batch,l8_pan = batch['s2_img'].cuda(),batch['l8_img'].cuda(),batch['l8_pan'].cuda()
        # l8_u = F.interpolate(l8_batch,size = [258,258],mode = 'bilinear')
        if pre_interpolate:
            l8_batch = F.interpolate(l8_batch,size = s2_batch.shape[2:],mode = 'bicubic')
            l8_pan = F.interpolate(l8_pan,size = s2_batch.shape[2:],mode = 'bicubic')
            if pan_highpass:
                l8_pan = get_highpass(l8_pan)
        if model_kind == "SMP_UNet_mit":
            logit = model(torch.cat([l8_batch,l8_pan], dim=1))
        elif model_kind == "AUNet_Gaussian":
            mean, variance = model(l8_batch, l8_pan)
            mean = F.interpolate(mean, size = s2_batch.shape[2:], mode = "nearest")
            variance = F.interpolate(variance, size = s2_batch.shape[2:], mode = "nearest")
            loss = gaussian_nll_loss(mean, s2_batch, variance, full = False, distance = "mse")
        else:        
            logit = model(l8_batch,l8_pan if cfg.fuse_pan else None)
            logit = F.interpolate(logit, size=s2_batch.shape[2:], mode="nearest")
            loss = calculate_loss(s2_batch, logit)
        overall_validation_loss.append(loss.item())
    model.train()
    validation_loss = statistics.mean(overall_validation_loss)
    return validation_loss

def train():
    model.train()
    overall_training_loss = []
    for i, batch in enumerate(tqdm(train_loader)):
        s2_batch, l8_batch, l8_pan = batch['s2_img'].cuda(), batch['l8_img'].cuda(), batch['l8_pan'].cuda()
        opt.zero_grad()
        if pre_interpolate:
            l8_batch = F.interpolate(l8_batch,size = s2_batch.shape[2:],mode = 'bicubic')
            l8_pan = F.interpolate(l8_pan,size = s2_batch.shape[2:],mode = 'bicubic')
            if pan_highpass:
                l8_pan = get_highpass(l8_pan)
        if model_kind == "SMP_UNet_mit":
            logit = model(torch.cat([l8_batch,l8_pan], dim=1))
        elif model_kind == "AUNet_Gaussian":
            mean, variance = model(l8_batch, l8_pan)
            mean = F.interpolate(mean, size = s2_batch.shape[2:], mode = "nearest")
            variance = F.interpolate(variance, size = s2_batch.shape[2:], mode = "nearest")
            loss = gaussian_nll_loss(mean, s2_batch, variance, full = False, distance = "mse")
        else:
            logit = model(l8_batch,l8_pan if cfg.fuse_pan else None)
            logit = F.interpolate(logit,size = s2_batch.shape[2:],mode = "nearest")
            loss  = calculate_loss(s2_batch,logit)
        loss.backward()
        opt.step()
        overall_training_loss.append(loss.item())
        # scaler.scale(loss).backward()
        # if (i + 1) % cfg.iters_to_accumulate == 0:
        #     scaler.step(opt)
        #     scaler.update()
        #     opt.zero_grad()
        #     overall_training_loss.append(loss.item())
        # del s2_batch
        # del l8_batch
        # del l8_pan
        # torch.cuda.empty_cache()
        if not i % 100:
            print(statistics.mean(overall_training_loss))
    return statistics.mean(overall_training_loss)


#reading_configs
cfg= get_cfg_defaults()
if not cfg.gpu_id is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
#hyper parameters
batch_size = cfg.batch_size
learning_rate = cfg.learning_rate
epochs = cfg.epochs
pre_interpolate = cfg.pre_interpolate
pan_highpass = cfg.high_pass

# Loss Function Declaration
if cfg.criterion == 'L1':
    loss_function = nn.L1Loss()
elif cfg.criterion == 'MGL':
    loss_function = mixed_gradient_error(criterion="L1")
elif cfg.criterion == 'SSIM':
    loss_function = SSIM(data_range=1,win_size = 7,channel=6).cuda()
else:
    print("Valid keys for criterion [L1, MGL(Mixed Gradient Loss), SSIM]")
    quit()

#Model Initializationd
model_kind = cfg.model_kind

if model_kind == 'UNet-Pan':
    model = SRUNet(in_channels=cfg.UNet_pan.in_channels,out_channels=cfg.UNet_pan.out_channels,
                   depth=cfg.UNet_pan.depth,SR_model=cfg.UNet_pan.SR_model,
                   up_mode=cfg.UNet_pan.up_mode)
elif model_kind == 'MB-RRDB':
    model = RRDB_pan_guided(nChannels=cfg.MB_RRDB.nchannels, nDenseLayers=cfg.MB_RRDB.DenseLayers,
                            nInitFeat=cfg.MB_RRDB.nInitFeat, GrowthRate=cfg.MB_RRDB.GrowthRate,
                            featureFusion=cfg.MB_RRDB.featureFusion, kernel_config=cfg.MB_RRDB.kernel_config,
                            pan_highpass=cfg.MB_RRDB.pan_highpass, pan_loss=cfg.MB_RRDB.pan_loss)
elif model_kind == "ESRT":
    model = ESRT(in_channels = cfg.ESRT.in_channels, out_channels = cfg.ESRT.out_channels,
                 hiddenDim=cfg.ESRT.hiddenDim, mlpDim=cfg.ESRT.mlpDim,
                 scaleFactor=cfg.ESRT.scaleFactor, num_heads = cfg.ESRT.num_heads)

elif model_kind == 'AUNet':
    model = AUNet(in_channels=cfg.AUNet.in_channels, out_channels=cfg.AUNet.out_channels,depth=cfg.AUNet.depth,
                  growth_factor=cfg.AUNet.growth_factor, interp_mode= cfg.AUNet.interp_mode, up_mode=cfg.AUNet.up_mode,
                  spatial_attention=cfg.AUNet.spatial_attention, ca_layer= cfg.AUNet.ca_layer)
elif model_kind == 'SMP_UNet_mit':
    model = smp.Unet(encoder_name = cfg.UNet_mit.encoder, encoder_depth = cfg.UNet_mit.encoder_depth, decoder_use_batchnorm = cfg.UNet_mit.decoder_use_batchnorm,
                     decoder_attention_type = cfg.UNet_mit.decoder_attention_type, in_channels = 3, classes = cfg.UNet_mit.num_classes,
                     activation = cfg.UNet_mit.activation, 
                    #  decoder_channels = cfg.UNet_mit.decoder_channels
                    )
    model.encoder.patch_embed1.proj = nn.Conv2d(in_channels=7, out_channels=64, kernel_size=(7,7), stride = (4,4), padding=(3,3))

elif model_kind == "AUNet_Gaussian":
    model = AUNet_Gaussian(in_channels=cfg.AUNet_G.in_channels, out_channels=cfg.AUNet_G.out_channels,depth=cfg.AUNet_G.depth,
                           growth_factor=cfg.AUNet_G.growth_factor, interp_mode= cfg.AUNet_G.interp_mode, up_mode=cfg.AUNet_G.up_mode,
                           spatial_attention=cfg.AUNet_G.spatial_attention, ca_layer= cfg.AUNet_G.ca_layer)
    # gnll = nn.GaussianNLLLoss(full = False)
else:
    print("valid model kinds [UNet-Pan, MB-RRDB]")
    quit()

if cfg.use_parallel:
    model = nn.DataParallel(model)

if cfg.use_cuda:
    model = model.cuda()

#Optimizers and Scheduler
opt = optim.Adam(model.parameters(),lr = learning_rate)
# scheduler = ReduceLROnPlateau(opt,cfg.scheduler_mode,factor=cfg.scheduler_factor,patience = cfg.scheduler_factor)
# scaler = GradScaler()

epochs_done = 0
if cfg.resume_training:
    weights = torch.load(cfg.resume_weight_path)
    model.load_state_dict(weights['model_state_dict'])
    opt.load_state_dict(weights['optimizer_state_dict'])
    epochs_done = weights['epoch']

#Training name and wandb initialization
training_name = cfg.training_name
if cfg.wandb_logging:
    init_parms = {"project":cfg.wandb_project,
                  "name":training_name,
                  "entity":cfg.wandb_entity}
    if cfg.wandb_resume:
        init_parms['id'] = cfg.wandb_id
        init_parms['resume'] = "allow"

    wandb.init(project=cfg.wandb_project,name = training_name,entity=cfg.wandb_entity)
    wandb.config = {
        "learning_rate":learning_rate,
        "epochs" : epochs,
        "batch_size": batch_size,
        "loss function": cfg.criterion,
        "model":model_kind
    }

# Dataset class and loaders
csv_path = cfg.csv_path
patch_path = cfg.patch_path

train_dataset = s2l8hDataset(
    csv_path=csv_path,
    patch_dir=patch_path,
    type=cfg.train_key,
    band_pass = cfg.band_pass,
    bandpass_model_path=cfg.bandpass_model_path,
    s2_array_size= cfg.s2_array_size,
    l8_array_size = cfg.l8_array_size,
    BOA=cfg.BOA
)
validation_dataset = s2l8hDataset(
    csv_path=csv_path,
    patch_dir=patch_path,
    type=cfg.validation_key,
    band_pass = cfg.band_pass,
    bandpass_model_path=cfg.bandpass_model_path,
    s2_array_size=cfg.s2_array_size,
    l8_array_size=cfg.l8_array_size,
    BOA = cfg.BOA
)

train_loader = DataLoader(dataset = train_dataset,batch_size=batch_size,num_workers=cfg.num_workers, collate_fn=collate_fn)
validation_loader = DataLoader(dataset = validation_dataset,batch_size=batch_size, collate_fn=collate_fn)
print(f"training : {training_name}")
print(f"train dataset : {len(train_dataset)}")
print(f"validation dataset : {len(validation_dataset)}")
if cfg.wandb_logging:
    wandb.watch(model)
old_validation_loss = 0

for epoch in range(epochs_done,epochs):
    opt.zero_grad()
    epoch_loss = train()
    validation_loss = validation_loop()
    test_network(epoch)
    if cfg.wandb_logging:
        wandb.log({"Training loss" : epoch_loss , "Validation Loss" : validation_loss})
    print(f"Epoch {epoch + 1} || Training loss ====> {epoch_loss} || Validation loss ====> {validation_loss}")
    if (old_validation_loss == 0) or (old_validation_loss > validation_loss):
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': loss_function}, os.path.join(cfg.checkpoint_path, training_name + ".pth"))
        print("model_saved")
        old_validation_loss = validation_loss
    # scheduler.step(validation_loss)
