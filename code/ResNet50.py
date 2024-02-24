
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import models 
from torchvision import transforms 
from torchvision.datasets import ImageFolder


class ResNet_Custom(nn.Module):
    def __init__(self, resnet_type=34):
        super(ResNet_Custom, self).__init__()
        fl = 512

        if(resnet_type == 34):
          backbone = models.resnet34(weights='IMAGENET1K_V1')
          fl = 512
        if(resnet_type == 50):
          backbone = models.resnet50(weights='IMAGENET1K_V1')
          fl = 2048
        if(resnet_type == 101):
          backbone = models.resnet101(weights='IMAGENET1K_V1')
        if(resnet_type == 151):
          backbone = models.resnet151(weights='IMAGENET1K_V1')

        for p in backbone.parameters(): 
          p.requires_grad = True 

        self.net =   nn.Sequential(*(list(backbone.children())[:-1]),
                                    nn.Flatten(start_dim=1, end_dim=-1),
                                    nn.Linear(in_features=fl, out_features=256) ,
                                    nn.ReLU(),
                                    nn.Linear(in_features=256, out_features=128) ,
                                    nn.ReLU(),
                                    nn.Linear(in_features=128 , out_features=18)
                                  )
        
      

    def forward(self, x):
      out = self.net(x)
      
      return out
