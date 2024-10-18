import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor):
        return self.block(x)
    
class CopyAndCrop(nn.Module):
    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor):
        _, _, h, w = skip_connection.shape
        crop = CenterCrop((h, w))(x)
        residual = torch.cat((x, crop), dim=1)
        return residual
    
class UNET(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(UNET, self).__init__() 
        
        self.encoders = nn.ModuleList([
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        ])
        
        self.pool = nn.MaxPool2d(2)
        
        self.copyAndCrop = CopyAndCrop()
        
        self.decoders = nn.ModuleList([
            ConvBlock(1024, 512),
            ConvBlock(512, 256),
            ConvBlock(256, 128),
            ConvBlock(128, 64),
        ])
        
        self.up_samples = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])
        
        self.bottleneck = ConvBlock(512, 1024)
        
        self.finalconv = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)
        
    def forward(self, x): 
        skip_connections = []
        
        # Encoding 
        for enc in self.encoders:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        
        # Decoding
        for idx, dec in enumerate(self.decoders):
            x = self.up_samples[idx](x)
            skip_connection = skip_connections.pop()
            x = self.copyAndCrop(x, skip_connection)
            x = dec(x)
            
        x = self.finalconv(x)
        
        return x
