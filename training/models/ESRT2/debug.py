import torch
import torch.nn as nn

# Use for debug
class PrintLayer(nn.Module):
    def __init__(self, x = ''):
        super().__init__()
        self.msg = x
    
    def forward(self, x):
        print(self.msg, x.shape)
        return x