import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Headings: Device Setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SobelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, requires_grad=False):
        assert kernel_size % 2 == 1, 'SobelConv2d kernel_size must be odd.'
        assert out_channels % 4 == 0, 'SobelConv2d out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d out_channels must be a multiple of groups.'
        super(SobelConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32), requires_grad=requires_grad) if bias and requires_grad else None
        self.sobel_weight = nn.Parameter(torch.zeros(out_channels, int(in_channels / groups), kernel_size, kernel_size), requires_grad=False)
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2
        self.sobel_factor = nn.Parameter(torch.ones(out_channels, 1, 1, 1, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        sobel_weight = self.sobel_weight * self.sobel_factor
        return F.conv2d(x, sobel_weight.to(x.device), self.bias.to(x.device) if self.bias is not None else None,
                        self.stride, self.padding, self.dilation, self.groups)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, max_val], dim=1)
        return self.sigmoid(self.conv(attn))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class LastDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(LastDecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ArterySegModel(nn.Module):
    def __init__(self, in_ch=1, sobel_ch=64):
        super(ArterySegModel, self).__init__()
        self.conv_sobel = SobelConv2d(in_ch, sobel_ch, kernel_size=3, stride=1, padding=1, bias=True, requires_grad=False)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.conv1 = nn.Conv2d(1 + sobel_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Backbone layers
        self.x0_conv = self.backbone.conv1      # Output: (B,64,672,672) for input (B,1344,1344)
        self.x0_bn = self.backbone.bn1
        self.x0_relu = self.backbone.relu
        self.x1_maxpool = self.backbone.maxpool  # Output: (B,64,336,336)
        self.enc1 = self.backbone.layer1         # (B,256,336,336)
        self.enc2 = self.backbone.layer2         # (B,512,168,168)
        self.enc3 = self.backbone.layer3         # (B,1024,84,84)
        self.enc4 = self.backbone.layer4         # (B,2048,42,42)
        
        # Decoder blocks (U-Net++ style)
        self.up4 = DecoderBlock(2048, 1024)       # Upsample to 84x84, combine with enc3
        self.up3 = DecoderBlock(1024, 512)        # Upsample to 168x168, combine with enc2
        self.up2 = DecoderBlock(512, 256)         # Upsample to 336x336, combine with enc1
        
        # Apply spatial attention to refine skip connections
        self.attention = SpatialAttention()
        
        self.up1 = LastDecoderBlock(256, 64, 128) # Upsample to 672x672, combine with x0 features
        self.final_upsample = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # Upsample to 1344x1344
        self.final_conv = nn.Conv2d(128, 5, kernel_size=1)  # 5-class segmentation output
        
    def forward(self, x):
        sobel_out = self.conv_sobel(x)
        x_cat = torch.cat([x, sobel_out], dim=1)
        x0 = self.x0_conv(x_cat)       # (B,64,672,672)
        x0 = self.x0_bn(x0)
        x0 = self.x0_relu(x0)
        x1 = self.x1_maxpool(x0)       # (B,64,336,336)
        e1 = self.enc1(x1)             # (B,256,336,336)
        e2 = self.enc2(e1)             # (B,512,168,168)
        e3 = self.enc3(e2)             # (B,1024,84,84)
        e4 = self.enc4(e3)             # (B,2048,42,42)
        d4 = self.up4(e4, e3)          # (B,1024,84,84)
        d3 = self.up3(d4, e2)          # (B,512,168,168)
        d2 = self.up2(d3, e1)          # (B,256,336,336)
        d2 = d2 * self.attention(d2)   # Apply attention to refine features
        d1 = self.up1(d2, x0)          # (B,128,672,672)
        d0 = self.final_upsample(d1)   # (B,128,1344,1344)
        out = self.final_conv(d0)      # (B,5,1344,1344)
        return out

