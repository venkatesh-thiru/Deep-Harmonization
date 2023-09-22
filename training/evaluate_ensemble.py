import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "GPU-ff380879-d5b2-8469-da3d-71267d28a645"
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
import os
import numpy as np
from tqdm import tqdm
import json
from models.helpers import get_highpass
from models.AttenUNet.AUNet import AUNet
import segmentation_models_pytorch as smp
import torch.nn as nn
import torchmetrics


def load_model_weights(parameters, weight_path):
    model = AUNet(**parameters)
    weights = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(weights['model_state_dict'])
    return model.cuda()


def make_ensemble_predictions(models, batch, pre_interpolated = True):
       s2_batch, l8_batch, l8_pan = batch['s2_img'].cuda(), batch['l8_img'].cuda(), batch['l8_pan'].cuda()
       if pre_interpolated:
              l8_batch = torch.nn.functional.interpolate(l8_batch, size=s2_batch.shape[2:], mode='bicubic')
              l8_pan = torch.nn.functional.interpolate(l8_pan, size=s2_batch.shape[2:], mode='bicubic')
       
       logits = [model(l8_batch, l8_pan) for model in models]
       logits = [torch.nn.functional.interpolate(logit, size = s2_batch.shape[2:], mode="nearest") for logit in logits]
       logits = torch.cat(logits, dim = 0)
       variance, prediction = torch.var_mean(logits, dim = 0, keepdim = True)
       return s2_batch, l8_batch, logits, prediction, variance

# def generate_model_results(models, test_loader, metrics_path):
#        model_dict = {}
       
#        for idx, batch in enumerate(tqdm(test_loader)):
#               patch_name = os.path.split(batch['patch_path'][0])[-1][:-3]
#               s2_img, l8_img, prediction, variance = make_predictions(models, batch)
#               metrics, band_metrics = get_metrics(s2_img.detach().squeeze().cpu().numpy(), prediction.detach().squeeze().cpu().numpy())
#               metrics['variance'] = torch.mean(variance).item()

#               model_dict[patch_name] = metrics
#        with open(metrics_path, "w") as metrics_file:
#             json.dump(model_dict, metrics_file, indent = 4)

def generate_uncertainty_results(models, test_loader):
    uncertainty_metrics = {}
    for i, batch in enumerate(tqdm(test_loader)):
        SC = torchmetrics.SpearmanCorrCoef().cuda()
        PC = torchmetrics.PearsonCorrCoef().cuda()
        gnll = torch.nn.GaussianNLLLoss()
        uncertainty_metrics_per_image = {}
        patch_name = os.path.split(batch['patch_path'][0])[-1][:-3]
        s2_batch, l8_batch, logits, prediction, variance = make_ensemble_predictions(models, batch, pre_interpolated=True)
        abs_diff = torch.abs(s2_batch-prediction) #L1 Loss

        uncertainty_metrics_per_image['SSIM'] =  structural_similarity(s2_batch.detach().squeeze().cpu().numpy(), prediction.detach().squeeze().cpu().numpy(),channel_axis=0,data_range=1).astype(float)
        uncertainty_metrics_per_image['SC'] = SC(variance.detach().view(-1), abs_diff.detach().view(-1)).item()
        uncertainty_metrics_per_image['PC'] = PC(variance.detach().view(-1), abs_diff.detach().view(-1)).item()
        uncertainty_metrics_per_image['NRMSE'] = normalized_root_mse(s2_batch.detach().squeeze().cpu().numpy(), prediction.detach().squeeze().cpu().numpy()).astype(float)
        uncertainty_metrics_per_image['GNLL'] = gnll(prediction.detach().cpu(), s2_batch.detach().cpu(), variance.detach().cpu()).item()
        
        uncertainty_metrics[patch_name] = uncertainty_metrics_per_image

    with open("/data_fast/venkatesh/s2l8h/uncertainty_metrics/ensemble.json","w") as outfile:
        json.dump(uncertainty_metrics, outfile, indent = 4)



model_zoo = {
       "5_upsample":{"model_parameters":{"in_channels":7, "out_channels":6, "depth":5, 
                                    "spatial_attention":"None", "growth_factor":6,
                                    "interp_mode":"bicubic", "up_mode":"upsample",
                                    "ca_layer":False},
                     "model_weight":"/data_fast/venkatesh/s2l8h/training/model_checkpoints/AttUNets/5 depth/None/AUNet - standard No SA and No CA Scaler 5 depth pixel upsample.pth.pth"},
       "5_shuffle":{"model_parameters":{"in_channels":7, "out_channels":6, "depth":5, 
                                    "spatial_attention":"None", "growth_factor":6,
                                    "interp_mode":"bicubic", "up_mode":"shuffle",
                                    "ca_layer":False},
                     "model_weight":"/data_fast/venkatesh/s2l8h/training/model_checkpoints/AttUNets/5 depth/None/AUNet - standard No SA and No CA Scaler 5 depth pixel shuffle.pth.pth"},
       "6_upsample":{"model_parameters":{"in_channels":7, "out_channels":6, "depth":6, 
                                    "spatial_attention":"None", "growth_factor":6,
                                    "interp_mode":"bicubic", "up_mode":"upsample",
                                    "ca_layer":False},
                     "model_weight":"/data_fast/venkatesh/s2l8h/training/model_checkpoints/AttUNets/6 depth/AUNet - No CA No SA 6 depth pixel upsample.pth"},
       "6_shuffle":{"model_parameters":{"in_channels":7, "out_channels":6, "depth":6, 
                                    "spatial_attention":"None", "growth_factor":6,
                                    "interp_mode":"bicubic", "up_mode":"shuffle",
                                    "ca_layer":False},
                     "model_weight":"/data_fast/venkatesh/s2l8h/training/model_checkpoints/AttUNets/6 depth/AUNet - No CA No SA 6 depth pixel shuffle.pth"},
}


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
models = [load_model_weights(model_zoo[key]['model_parameters'], model_zoo[key]['model_weight']) for key in model_zoo.keys()]
# generate_model_results(models, test_loader, "/data_fast/venkatesh/s2l8h/model_metric_extended_data/ensemble/ensemble_model_metrics.json")


# generate_uncertainty_results(models, test_loader)


