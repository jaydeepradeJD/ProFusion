import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_norm=False):
        super(ConvBlock, self).__init__()
        self.use_norm = use_norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if self.use_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.use_norm:
            self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.use_norm:
            x = self.bn1(x)
        x = F.relu(self.conv2(x))
        if self.use_norm:
            x = self.bn2(x)
        return x
# Small CNN model for image feature extraction
class SmallCNN(nn.Module):
    def __init__(self, in_dim=3, out_dim=768):
        super(SmallCNN, self).__init__()
        #input shape (3, 256, 256)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv1 = ConvBlock(self.in_dim, 64, 5, 2, 1, use_norm=True)
        #output shape (64, 128, 128)
        self.conv2 = ConvBlock(64, 128, 3, 2, 1, use_norm=True)
        #output shape (128, 64, 64)
        self.conv3 = ConvBlock(128, 256, 3, 2, 1, use_norm=True)
        #output shape (256, 32, 32)
        self.conv4 = ConvBlock(256, 512, 3, 2, 1, use_norm=True)
        #output shape (512, 16, 16)
        self.conv5 = ConvBlock(512, self.out_dim, 3, 1, 1, use_norm=True)
        #output shape (768, 16, 16)
    
    def forward(self, input_views):
        N = input_views.shape[1]
        reshaped = rearrange(input_views, "b n c h w -> (b n) c h w")
        x = self.conv1(reshaped) # (b*n, 64, 128, 128)
        x = self.conv2(x) # (b*n, 128, 64, 64)
        x = self.conv3(x) # (b*n, 256, 32, 32)
        x = self.conv4(x) # (b*n, 512, 16, 16)
        x = self.conv5(x) # (b*n, 768, 16, 16)
        x = rearrange(x, "(b n) c h w -> b n (h w) c", n=N) # (b, n, 256, 768)

        return x    

