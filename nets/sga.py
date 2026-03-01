import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGatedAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, reduction=16):
        super(SpatialGatedAttention, self).__init__()
        
        # 空间注意力分支
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_conv(spatial_input)
        
        # 计算门控
        gate = self.gate(x)
        
        # 应用注意力和门控
        out = x * spatial_att
        out = out * gate
        
        return out
