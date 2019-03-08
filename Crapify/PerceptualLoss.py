
# coding: utf-8
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import fastai
import fastai.basics as fai
import fastai.vision as fv
from fastai.callbacks import hook_outputs, hook_output
from pathlib import Path
from shutil import copyfile
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.utils as utils
import torchvision.models
from torch.utils.data import DataLoader, Dataset
import PIL
from tqdm import tqdm_notebook as tqdm
import gc
import matplotlib.pyplot as plt
import numpy as np
import gc

vgg = torchvision.models.vgg16_bn(pretrained=True).features

vgg.eval()
fai.requires_grad(vgg, False)

gc.collect()

good_blocks = [i-1 for i,o in enumerate(vgg.children()) if isinstance(o,nn.MaxPool2d)]

class PerceptualLoss(nn.Module):
    def __init__(self, model, layer_ids, weights):
        super().__init__()
        self.model = model
        self.important_layers = [self.model[i] for i in layer_ids]
        self.hooks = hook_outputs(self.important_layers, detach=False)
        self.weights = weights

    def extract_features(self, x, clone=False):
        self.model(x)
        features = list(self.hooks.stored)
        
        if clone:
            features = [f.clone() for f in features]
        
        return features
    
    def forward(self, input, target):
        criterion = F.l1_loss
        
        input_features = self.extract_features(input)
        target_features = self.extract_features(target, clone=True)
        
        self.feat_losses = [criterion(input,target)]
        self.feat_losses += [criterion(in_f, targ_f)*w for in_f, targ_f, w in zip(input_features, target_features, self.weights)]
        
        return sum(self.feat_losses)
    
    def __del__(self): 
        self.hooks.remove() # necesario para que deje de guardar las cosas

perceptual_loss = PerceptualLoss(vgg, layer_ids=good_blocks[2:], weights=[5,15,2])
