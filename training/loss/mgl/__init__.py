import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import L1Loss,MSELoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def custom_sobel(shape,axis,channels = 6):
    '''

    :param shape:
    :param axis:
    :return:
    '''
    kernel = np.zeros(shape)
    p = [(j,i) for j in range(shape[0])
               for i in range(shape[1]) if not (i == (shape[1]-1)/2. and j == (shape[0]-1)/2.)]
    for j,i in p:
        j_ = int(j-(shape[0]-1)/2.)
        i_ = int(i-(shape[1]-1)/2.)
        kernel[j,i] = (i_ if axis==0 else j_)/float(i_*i_+j_*j_)

    return torch.tensor(kernel,requires_grad=False).repeat(channels,channels,1,1)

def _generate_edges(image,filterx,filtery):
    kernel_size = filterx.shape[-1]

    dx = F.conv2d(image, weight=filterx, padding=(kernel_size - 1) // 2)
    dy = F.conv2d(image, weight=filtery, padding=(kernel_size - 1) // 2)
    edge = torch.sqrt(dx**2 + dy**2)
    return edge

class mixed_gradient_error(nn.Module):
    def __init__(self,criterion = "MSE"):
        super(mixed_gradient_error, self).__init__()
        self._filter_x = custom_sobel([3,3],axis = 0)
        self._filter_y = custom_sobel([3,3],axis = 1)
        self.criterion = criterion

    def forward(self,img1,img2):
        if self._filter_x.data.type() == img1.data.type():
            filterx = self._filter_x
            filtery = self._filter_y
        else:
            filterx = self._filter_x
            filtery = self._filter_y
            if img1.is_cuda:
                filterx = filterx.cuda(img1.get_device())
                filtery = filtery.cuda(img1.get_device())

            filterx = filterx.type_as(img1)
            filtery = filtery.type_as(img1)

        grad1 = _generate_edges(img1,filterx,filtery)
        grad2 = _generate_edges(img2,filterx,filtery)

        if self.criterion == 'MAE':
            loss_fn = L1Loss()
            return loss_fn(grad1,grad2).item()
        elif self.criterion == 'MSE' or 'RMSE':
            loss_fn = MSELoss()
            if self.criterion == 'MSE':
                return loss_fn(grad1,grad2).item()
            elif self.criterion == 'RMSE':
                return torch.sqrt(loss_fn(grad1,grad2)).item()