import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Headings: Device Setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Headings: Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

# Headings: Sobel Convolution Layer
class SobelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, requires_grad=False):
        super(SobelConv2d, self).__init__()
        assert kernel_size % 2 == 1
        assert out_channels % 4 == 0
        assert out_channels % groups == 0
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

# Headings: Stent Detection Model
class StentDetectionModel(nn.Module):
    def __init__(self, in_ch=1, sobel_ch=32):
        super(StentDetectionModel, self).__init__()
        self.conv_sobel = SobelConv2d(in_ch, sobel_ch, kernel_size=3, stride=1, padding=1, bias=True, requires_grad=False)
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.spatial_attention = SpatialAttention()
        self.backbone.conv1 = nn.Conv2d(1 + sobel_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer1 = nn.Sequential(*list(self.backbone.children())[:5])
        self.layer2 = nn.Sequential(*list(self.backbone.children())[5])
        self.layer3 = nn.Sequential(*list(self.backbone.children())[6])
        self.layer4 = nn.Sequential(*list(self.backbone.children())[7])
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.downsample = nn.Conv2d(16, 5, kernel_size=3, stride=1, padding=1)
    def unfreeze_layers(self):
        for param in self.parameters():
            param.requires_grad = True
    def forward(self, x):
        if x is None or x.numel() == 0:
            print("⚠ Warning: Empty tensor encountered in forward pass!")
            return None
        out = self.conv_sobel(x)
        out = torch.cat((x, out), dim=1)
        features1 = self.layer1(out)
        features2 = self.layer2(features1)
        features3 = self.layer3(features2)
        features4 = self.layer4(features3)
        attn_mask = self.spatial_attention(features4)
        deconv1_out = self.deconv1(attn_mask * features4)
        deconv2_out = self.deconv2(deconv1_out + features3)
        deconv3_out = self.deconv3(deconv2_out + features2)
        deconv4_out = self.deconv4(deconv3_out + features1)
        deconv5_out = self.deconv5(deconv4_out)
        return self.downsample(deconv5_out)

