import argparse
from yacs.config import CfgNode as CN
import os
#adding this command to check RSYNC behaviour
cfg = CN()

# GENERAL TRAINING HP
cfg.gpu_id = ""
cfg.training_name = "AUNet Gaussian bicubic shuffle depth 6 distance mse"
cfg.random_seed = 42
cfg.batch_size = 10
cfg.learning_rate = 0.00001
cfg.epochs = 200
cfg.scheduler_mode = 'min'
cfg.scheduler_factor = 0.5
cfg.scheduler_patience = 10
cfg.checkpoint_path = r"/data_fast/venkatesh/s2l8h/training/model_checkpoints/AttUNets"
cfg.criterion = 'L1'
cfg.pre_interpolate = True # Resamples the inputs to the target size using bicubic interpolation
cfg.fuse_pan = True
cfg.high_pass = False # Applies a high pass filter to the inputs
cfg.band_pass = False # applies the band-pass function to the inputs
cfg.BOA = True
cfg.bandpass_model_path = "/data_fast/venkatesh/s2l8h/training/SRF/models"
cfg.use_cuda = True
cfg.use_parallel = False
cfg.iters_to_accumulate = 1
cfg.resume_training = False
cfg.resume_weight_path = "/data_fast/venkatesh/s2l8h/training/model_checkpoints/AttUNets/AUNet - No SA No CA.pth"

#wandb
cfg.wandb_resume = False
cfg.wandb_logging = True #weights and biases logging
if cfg.wandb_logging:
    cfg.wandb_project = "s2l8h_transformers"
    cfg.wandb_entity = ""
    cfg.wandb_description = "Benchmarking Transformer models for the final iteration of the paper"


#Data Loader configs
cfg.csv_path = "/data_fast/venkatesh/s2l8h/train_test_validation_patch_extended_final.csv" # CSV containing patch details
cfg.patch_path = "/data_fast/venkatesh/s2l8h/DATA_L2_cloudless_patches" # Path to the directory containing patches
cfg.train_key = "Train" # Key for training patches in the CSV
cfg.test_key = "Test"
cfg.validation_key = "Validation"
cfg.num_workers = 4
cfg.s2_array_size = 256
cfg.l8_array_size = 86


#Architecture HP
cfg.model_kind = "AUNet_Gaussian"


#UNet-Pan configs
if cfg.model_kind == 'UNet-Pan':
    cfg.UNet_pan = CN()
    cfg.UNet_pan.in_channels = 7 if cfg.fuse_pan else 6
    cfg.UNet_pan.out_channels = 6
    cfg.UNet_pan.depth = 3
    cfg.UNet_pan.SR_model = False
    cfg.UNet_pan.up_mode = 'upconv'
#MB-RRDB parameters
elif cfg.model_kind == 'MB-RRDB':
    cfg.MB_RRDB = CN()
    cfg.MB_RRDB.nchannels = 6 if cfg.fuse_pan else 6
    cfg.MB_RRDB.DenseLayers = 6
    cfg.MB_RRDB.nInitFeat = 6
    cfg.MB_RRDB.GrowthRate = 12
    cfg.MB_RRDB.featureFusion = True
    cfg.MB_RRDB.kernel_config = [3,3,3,3]
    cfg.MB_RRDB.pan_highpass = False
    cfg.MB_RRDB.pan_loss = False
#ESRT parameters
elif cfg.model_kind == 'ESRT':
    cfg.ESRT = CN()
    cfg.ESRT.in_channels = 7
    cfg.ESRT.out_channels = 6
    cfg.ESRT.hiddenDim = 32
    cfg.ESRT.mlpDim = 128
    cfg.ESRT.scaleFactor = 1
    cfg.ESRT.num_heads = 4

elif cfg.model_kind == 'AUNet':
    cfg.AUNet = CN()
    cfg.AUNet.in_channels = 7
    cfg.AUNet.out_channels = 6
    cfg.AUNet.depth = 3
    cfg.AUNet.spatial_attention = 'standard'
    cfg.AUNet.growth_factor = 6
    cfg.AUNet.interp_mode = 'bicubic'
    cfg.AUNet.up_mode = 'shuffle'
    cfg.AUNet.ca_layer = False

elif cfg.model_kind == 'AUNet_Gaussian':
    cfg.AUNet_G = CN()
    cfg.AUNet_G.in_channels = 7
    cfg.AUNet_G.out_channels = 6
    cfg.AUNet_G.depth = 3
    cfg.AUNet_G.spatial_attention = 'None'
    cfg.AUNet_G.growth_factor = 6
    cfg.AUNet_G.interp_mode = 'bicubic'
    cfg.AUNet_G.up_mode = 'shuffle'
    cfg.AUNet_G.ca_layer = False

elif cfg.model_kind == "SMP_UNet_mit":
    cfg.UNet_mit = CN()
    cfg.UNet_mit.encoder = "mit_b3"
    cfg.UNet_mit.encoder_depth = 5
    cfg.UNet_mit.decoder_use_batchnorm = False
    cfg.UNet_mit.decoder_attention_type = None
    cfg.UNet_mit.in_channels = 7 
    cfg.UNet_mit.num_classes = 6
    cfg.UNet_mit.activation = None
    # cfg.UNet_mit.decoder_channels = [418,256,128,64]


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    return cfg, cfg_file

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    print(cfg)
