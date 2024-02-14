import torch.nn as nn
from torch.optim import Adam
import torch
from torchvision.models import resnet50, ResNet50_Weights

class ResnetDepthModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.result_history = []

        # First convolution to switch from an image with 1 channel to an image with 3 channels (needed for resnet)
        self.first_conv = nn.Conv2d(1, 3, stride=(1,1), padding=(1,1))

        # Using the weighted Resnet
        resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

