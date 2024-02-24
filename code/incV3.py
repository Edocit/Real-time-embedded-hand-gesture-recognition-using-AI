import numpy as np
import os
import torch
import torchvision
from torchvision import transforms, models
from torchvision import datasets
import PIL
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
from torch.utils.data import Dataset
import time
import os
import copy
from torch import Tensor, nn
from typing import Dict
import tqdm as notebook_tqdm
from tqdm import tqdm

class Inception3(nn.Module):

    def __init__(self, num_classes=18):
        super(Inception3, self).__init__()

        model_ft = models.inception_v3(pretrained=True)



        model_ft.aux_logits = False
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

        #handle primary net

        for param in model_ft.parameters():
            param.requires_grad = True
        model_ft.fc = nn.Identity()

        self.backbone = model_ft
 
        self.gesture_classifier = nn.Sequential(
            nn.Linear(in_features=2048,out_features=512),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=512, out_features=num_classes),
          )


    def forward(self, x):
      #with torch.no_grad():
      x = self.backbone(x)
      
      #x = x.view(x.size(0), -1)
      x = self.gesture_classifier(x)

      return x
