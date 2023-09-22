import os
os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-bb1ccb6e-2bc9-c7a1-b25d-3eef9033e192'
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from Dataset import s2l8hDataset
from skimage.metrics import normalized_root_mse,peak_signal_noise_ratio,structural_similarity
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from models.RRDB import RRDB
from models.UNet import SRUNet
from models.pan_fusion import PANF_Generator
from models.RRDB_pan_fusion import RRDB_pan_guided
import numpy as np
from tqdm import tqdm
import json
from models.helpers import get_highpass
from models.AttenUNet.AUNet import AUNet

import torch.nn as nn
# from models.ESRT.ESRT import ESRT


def load_model_weights(kind,weight_path = None):
    '''
    :param kind:UNet-PAN or MB-RRDB
    :param weight_path: Full Path to model weight
    :return: model with loaded weights
    '''
    if kind == "UNet-Pan":
        model = SRUNet(in_channels=7,out_channels=6,depth=3,SR_model=False,up_mode='upconv').cuda()
    elif kind == "MB-RRDB":
        model = RRDB_pan_guided(nChannels=6,nDenseLayers=6,nInitFeat=6,GrowthRate=12,
                                featureFusion=True,kernel_config=[3,3,3,3],pan_highpass=False,pan_loss=False).cuda()
    elif kind == "SMP_UNet_mit":
        model = smp.Unet(encoder_name = "mit_b2", encoder_depth = 5, decoder_use_batchnorm = False,
                    decoder_attention_type = None, in_channels = 3, classes = 6,
                    activation = None, 
                #  decoder_channels = cfg.UNet_mit.decoder_channels
                ).cuda()
        model.encoder.patch_embed1.proj = nn.Conv2d(in_channels=7, out_channels=64, kernel_size=(7,7), stride = (4,4), padding=(3,3))
    if weight_path != None:
        weights = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(weights["model_state_dict"],strict = False)
        model = model.cuda()

    return model

def get_metrics(gt,predicted):
    '''
    Calculate the metrics between ground truths and predictions
    :param gt: ground truth image
    :param predicted: predicted image
    :return: metrics, bandwise metrics
    '''
    metrics_dict = {
        "NRMSE" : normalized_root_mse(gt,predicted).astype(float),
        "PSNR"  : peak_signal_noise_ratio(gt,predicted,data_range = 1).astype(float),
        "SSIM"  : structural_similarity(gt,predicted,channel_axis=0,data_range=1).astype(float)
    }
    metrics_bandwise = {}
    for i in range(0,6):
        gt_channel,pred_channel = np.squeeze(gt[i,:,:]),np.squeeze(predicted[i,:,:])
        metrics_bandwise[f"NRMSE_B{i+2}"] = normalized_root_mse(gt_channel,pred_channel).astype(float)
        metrics_bandwise[f"PSNR_B{i + 2}"] = peak_signal_noise_ratio(gt_channel, pred_channel,data_range = 2).astype(float)
        metrics_bandwise[f"SSIM_B{i + 2}"] = structural_similarity(gt_channel, pred_channel,win_size=5,data_range = 2).astype(float)
    return metrics_dict,metrics_bandwise


def get_derived_metrics(gt,predicted,derived):
    '''
    Calculate metrics for derived bands
    :param gt: ground truth image
    :param predicted: predicted image
    :param derived: "NDVI" or "NDWI1" or "NDWI2"
    :return: derived band metrics (dict)
    '''
    if derived == "NDVI":
        gt_derived = (gt[3]-gt[2])/(gt[3]+gt[2])
        predicted_derived = (predicted[3] - predicted[2]) / (predicted[3] + predicted[2])
    elif derived == "NDWI1":
        gt_derived = (gt[3] - gt[4])/(gt[3] + gt[4])
        predicted_derived = (predicted[3] - predicted[4]) / (predicted[3] + predicted[4])
    elif derived == "NDWI2":
        gt_derived = (gt[3] - gt[5]) / (gt[3] + gt[5])
        predicted_derived = (predicted[3] - predicted[5]) / (predicted[3] + predicted[5])

    gt_derived = gt_derived.clip(min = -1, max = 1)
    gt_derived[np.isnan(gt_derived)] = 0
    predicted_derived = predicted_derived.clip(min = -1,max = 1)
    predicted_derived[np.isnan(predicted_derived)] = 0

    metrics_dict = {
        "NRMSE" : normalized_root_mse(gt_derived, predicted_derived).astype(float),
        "PSNR" : peak_signal_noise_ratio(gt_derived, predicted_derived, data_range=1).astype(float),
        "SSIM" : structural_similarity(gt_derived.clip(min = 0,max = 1), predicted_derived.clip(min = 0,max = 1), data_range=1).astype(float),
        "MAE" : mean_absolute_error(gt_derived.flatten(order = "C"), predicted_derived.flatten(order = "C")).astype(float),
        "Pearson C" : pearsonr(gt_derived.flatten(order = "C"), predicted_derived.flatten(order = "C"))[0].astype(float)
    }
    return metrics_dict


def generate_baseline_results(path,test_loader, method = "bilinear",derived = None):

    baseline_dict = {}
    baseline_band_dict = {}

    for idx,batch in enumerate(tqdm(test_loader)):
        s2_batch,l8_batch = batch["s2_img"],batch["l8_img"]
        l8_batch = F.interpolate(l8_batch, size = s2_batch.shape[2:],mode = method)
        patch_name = os.path.split(batch["patch_path"][0])[-1][:-3]
        if derived is None:
            metrics, band_metrics = get_metrics(s2_batch.squeeze().numpy(), l8_batch.squeeze().numpy())
            baseline_dict[patch_name] = metrics
            baseline_band_dict[patch_name] = band_metrics
        else:
            metrics = get_derived_metrics(s2_batch.squeeze().numpy(), l8_batch.squeeze().numpy(),derived = derived)
            baseline_dict[patch_name] = metrics

    metric_path = os.path.join(path, method)
    os.makedirs(metric_path, exist_ok=True)
    if derived is None:
        with open(f"{metric_path}/metrics.json","w") as metrics_file:
            json.dump(baseline_dict,metrics_file,indent=4)
        with open(f"{metric_path}/metrics_bandwise.json","w") as metrics_file:
            json.dump(baseline_band_dict,metrics_file,indent=4)
    else:
        with open(f"{metric_path}/metrics_{derived}.json","w") as metrics_file:
            json.dump(baseline_dict,metrics_file,indent=4)

def make_prediction(model,kind,batch,pre_interpolated=True,high_pass = False):
    s2_batch, l8_batch, l8_pan = batch["s2_img"], batch["l8_img"].cuda(), batch["l8_pan"].cuda()
    if pre_interpolated:
        l8_batch = torch.nn.functional.interpolate(l8_batch, size=s2_batch.shape[2:], mode='bicubic')
        l8_pan = torch.nn.functional.interpolate(l8_pan, size=s2_batch.shape[2:], mode='bicubic')
    if high_pass:
        l8_pan = get_highpass(l8_pan)

    if kind == "UNet-Pan":
        logit = model(l8_batch,l8_pan)

    if kind == "MB-RRDB":
        logit = model(l8_batch,l8_pan)

    if kind == "ESRT":
        logit = model(l8_batch)

    if kind == "AUNet":
        logit = model(l8_batch,l8_pan)
    
    if kind == "SMP_UNet_mit":
        logit = model(torch.cat([l8_batch, l8_pan], dim = 1))


    logit = torch.nn.functional.interpolate(logit, size=s2_batch.shape[2:], mode='nearest')
    s2_img = s2_batch.squeeze().numpy()
    l8_img = l8_batch.squeeze().detach().cpu().numpy()
    logit = logit.squeeze().detach().cpu().numpy()

    return s2_img,l8_img,logit


def generate_model_results(kind, weight_path,metrics_path,test_loader,model=None,pre_interpolate=True,folder = None,high_pass = False,derived = None):
    '''
    generated the test metrics and dumps them to json
    :param kind: "UNet-PAN" or "MB-RRDB"
    :param weight_path: full path to weight
    :param pre_interpolate: Boolean to perform pre-interpolation
    :param folder: in case the json have to be dumped into a specific folder leave it None to use the model kind as the directory
    :param high_pass: Boolean to perform high-pass adjustment
    :param derived: "NDVI","NDWI1" or "NDWI2" if the derived bands have to be evaluated
    :return:
    '''
    if folder is None:
        folder = kind
    if model == None:
        model = load_model_weights(kind=kind,
                                   weight_path=weight_path)

    model.eval()
    model_dict = {}
    model_band_dict = {}

    for idx,batch in enumerate(tqdm(test_loader)):
        patch_name = os.path.split(batch["patch_path"][0])[-1][:-3]
        s2_img,l8_img,logit = make_prediction(model,kind = kind,
                                              batch=batch,
                                              pre_interpolated = pre_interpolate,high_pass = high_pass)
        if derived is None:
            metrics,band_metrics = get_metrics(s2_img,logit)
            model_dict[patch_name] = metrics
            model_band_dict[patch_name] = band_metrics
        else:
            metrics = get_derived_metrics(s2_img,logit,derived)
            model_dict[patch_name] = metrics
    os.makedirs(metrics_path,exist_ok = True)
    if derived is None:
        model_name = os.path.split(weight_path)[-1].split(".")[0]
        os.makedirs(os.path.join(metrics_path, folder), exist_ok=True)
        with open(os.path.join(metrics_path, folder, f"{model_name}_metrics_bandpass.json"),"w") as metrics_file:
            json.dump(model_dict, metrics_file, indent=4)
        with open(os.path.join(metrics_path, folder, f"{model_name}_band_metrics_bandpass.json"), "w") as metrics_file:
            json.dump(model_band_dict, metrics_file, indent=4)
    else:
        model_name = os.path.split(weight_path)[-1].split(".")[0]
        with open(os.path.join(metrics_path, f"{model_name}_metrics_bandpass_{derived}.json"),"w") as metrics_file:
            json.dump(model_dict, metrics_file, indent=4)


if __name__ == "__main__":

    train_dataset = s2l8hDataset(
        csv_path = "/data_fast/venkatesh/s2l8h/train_test_validation_patch_extended_final.csv",
        patch_dir = "/data_fast/venkatesh/s2l8h/DATA_L2_cloudless_patches",
        type = "Test",
        BOA = True,
        s2_array_size = 256,
        l8_array_size = 86,
        band_pass = False
    )
    test_loader = DataLoader(train_dataset,batch_size = 1)
    model_kind = "SMP_UNet_mit"
    weight_path = "/data_fast/venkatesh/s2l8h/training/model_checkpoints/AttUNets/SMP UNet mit_b5.pth"

    model = load_model_weights(kind = model_kind, weight_path=weight_path)
    generate_model_results(kind = model_kind,weight_path = weight_path,
                        metrics_path="/data_fast/venkatesh/s2l8h/model_metric_extended_data",
                        model=model,test_loader = test_loader)