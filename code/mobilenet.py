from typing import Dict

import torchvision
from torch import Tensor, nn


class MobileNetV3(nn.Module):

    def __init__(self, num_classes: int,  mobnet_type: str = "large"):
        
        super(MobileNetV3, self).__init__()
        in_features = 960
        out_features = 1280

        if mobnet_type == "large":
            backbone = torchvision.models.mobilenet_v3_large(pretrained=True)
            in_features = 960
            out_features = 1280
        if mobnet_type == "small":
            backbone = torchvision.models.mobilenet_v3_small(pretrained=True)
            in_features = 576
            out_features = 1024


        for param in backbone.parameters():
            param.requires_grad = True
                
        self.backbone = nn.Sequential(backbone.features, backbone.avgpool)

        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=out_features, out_features=512),
            nn.Hardswish(),
            nn.Linear(in_features=512, out_features=num_classes),
        )                     




    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        gesture = self.net(x)
        return gesture
    