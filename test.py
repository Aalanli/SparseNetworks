# %%
import math

import torch
from convNeXtRef import convnext_tiny

from torchvision.models import resnet34

def calculate_param_size(model):
    params = 0
    for i in model.parameters():
        params += math.prod(list(i.shape))
    return params

print(calculate_param_size(resnet34()))

