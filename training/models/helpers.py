import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from joblib import load
import os


def get_edge(X):
    edge = F.avg_pool2d(X,kernel_size = 5, stride = 1, padding = 2)
    edge = X-edge
    return edge

def get_highpass(X):
    '''
    Applies a highpass filter over the given image array, Uses torch convolution kernels
    :param X: image tensor
    :return: high-pass image
    '''
    if X.is_cuda:
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    kernel = [[1.,1.,1.],[1.,-8.,1.],[1.,1.,1.]]
    batch,channels = X.shape[:2]
    kernel = Tensor(kernel).expand(channels,channels,3,3)
    kernel = nn.Parameter(data=kernel,requires_grad=False)
    return F.conv2d(X,kernel,stride = 1,padding = 1)

def perform_band_pass(
                      l8_img,
                      model_path="SRF/models",
                      ):
    '''
    performs band-pass adjustments on the given landsat-8 image based on the generated poly-reg functions
    :param l8_img: landsat-8 image array
    :param model_path: path where bandwise regression model is stored
    :return: high-pass filter applied image
    '''
    bands = ["B2", "B3", "B4", "B5", "B6", "B7"]
    transformed_bands = []
    for i, band in enumerate(bands):
        model = load(os.path.join(model_path, "poly_reg", f"poly_reg_SRF_{band}.joblib"))
        image = l8_img[i]
        image_shape = image.shape
        flat = image.flatten(order="C").reshape(-1, 1)
        flat = np.squeeze(model.predict(flat))
        out = flat.reshape(image_shape, order="C")
        transformed_bands.append(out)
    return np.array(transformed_bands)

if __name__ == '__main__':
    from training.Dataset import s2l8hDataset
    test_dataset = s2l8hDataset(
        csv_path = "/home/local/DFPOC/thirugv/s2l8h/s2l8h/train_test_validation_patch.csv",
        patch_dir = "/home/local/DFPOC/thirugv/s2l8h/s2l8h/DATA_L2_cloudless_patches",
        type = "Test"
    )
    sample = test_dataset[128]
    array = sample['l8_pan'][0]
    edge = get_edge(torch.from_numpy(array).unsqueeze(dim= 0)).squeeze().numpy()
    hp = get_highpass(torch.from_numpy(array).unsqueeze(dim= 0).unsqueeze(dim = 0)).squeeze().numpy()
    # plt.imshow(hp,cmap = 'gray')
    plt.imshow(np.hstack((array,hp)),cmap = 'gray',vmin = 0,vmax = 0.1)
    plt.show()
