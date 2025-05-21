import torch.nn as nn
import torch

# 定义简单的UNet模型
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()
        
        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
        self.down1 = double_conv(in_channels, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.down4 = double_conv(256, 512)
        
        self.maxpool = nn.MaxPool2d(2)
        
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        
        self.up_conv4 = double_conv(512, 256)
        self.up_conv3 = double_conv(256, 128)
        self.up_conv2 = double_conv(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        # 编码器部分
        conv1 = self.down1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.down3(x)
        x = self.maxpool(conv3)
        
        x = self.down4(x)
        
        # 解码器部分
        x = self.up4(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_conv4(x)
        
        x = self.up3(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_conv3(x)
        
        x = self.up2(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv2(x)
        
        x = self.final_conv(x)
        return x