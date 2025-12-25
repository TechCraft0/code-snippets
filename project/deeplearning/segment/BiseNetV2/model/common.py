import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loss.dice_loss import dice_loss_func

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemBlock, self).__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

        self.branch1 = nn.Sequential(
            Conv(out_channels, out_channels // 2, kernel_size=1, stride=1, padding=0),
            Conv(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
        )

        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )

        self.conv2 = Conv(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        out = self.conv1(x)
        branch1 = self.branch1(out)
        branch2 = self.branch2(out)
        out = torch.cat([branch1, branch2], dim=1)
        return self.conv2(out)

class GatherExpansionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio, stride=1):
        """
        参数：
        - in_channels: 输入通道数 (C)
        - expansion_channels: 扩展通道数 (higher-dimensional space)
        - out_channels: 输出通道数 (projection后的通道数)
        - stride: 步长，1或2，影响下采样
        """
        super(GatherExpansionLayer, self).__init__()
        mid_channels = int(in_channels * expansion_ratio)  # 计算扩展通道数
        self.stride = stride

        # 3x3 卷积扩展空间（普通卷积）
        self.conv1 = Conv(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

        if self.stride == 1:
            # 1x1 卷积扩展空间（普通卷积）
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.conv2 = Conv(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        elif self.stride == 2:
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.dwconv2 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.conv2 = Conv(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)  # 3x3普通卷积扩展维度

        if self.stride == 1:
            out = self.dwconv(out)  # 1x1普通卷积扩展维度
            out = self.conv2(out)  # 1x1普通卷积投影回原始维度
            out = out + x  # 残差连接
        elif self.stride == 2:
            out = self.dwconv(out)  # 1x1普通卷积扩展维度
            out = self.dwconv2(out)  # 1x1普通卷积扩展维度
            out = self.conv2(out)  # 1x1普通卷积投影回原始维度
            shortcut = self.shortcut(x)  # 1x1普通卷积投影回原始维度
            out = out + shortcut  # 残差连接

        out = self.relu(out)
        return out

class ContextEmphasis(nn.Module):
    def __init__(self, in_channels):
        super(ContextEmphasis, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv1 = Conv(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv1(feat)
        feat = feat + x
        return self.conv2(feat)

class BilateralGuidedAggregation(nn.Module):
    def __init__(self, in_channels):
        super(BilateralGuidedAggregation, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.left2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )

        self.right1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        )

        self.right2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        l1 = self.left1(x1)
        l2 = self.left2(x1)

        r1 = torch.sigmoid(self.right1(x2))
        r2 = torch.sigmoid(self.right2(x2))

        out1 = l1 * r1
        out2 = self.up(l2 * r2)

        out = out1 + out2
        out = self.conv(out)
        return out

class SegmentHead(nn.Module):

    def __init__(self, in_channels, mid_channels, n_classes):
        super(SegmentHead, self).__init__()
        self.conv1 = Conv(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(mid_channels, n_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, size=None):
        feat = self.conv1(x)
        feat = self.drop(feat)
        feat = self.conv2(feat)
        if size is not None:
            feat = F.interpolate(feat, size, mode='bilinear', align_corners=True)
        return feat

class DetailEnhance(nn.Module):
    def __init__(self):
        super(DetailEnhance, self).__init__()
        self.S1 = nn.Sequential(
            Conv(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            Conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        )

        self.S2 = nn.Sequential(
            Conv(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            Conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            Conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        )

        self.S3 = nn.Sequential(
            Conv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            Conv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            Conv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.S1(x)
        x = self.S2(x)
        x = self.S3(x)
        return x

class SegmentBranch(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SegmentBranch, self).__init__()

        self.s1s2 = StemBlock(in_channels, out_channels)

        self.s3_ge_layer = nn.Sequential(
            GatherExpansionLayer(16, 32, 6, 2),
            GatherExpansionLayer(32, 32, 6, 1),
        )
        self.s4_ge_layer = nn.Sequential(
            GatherExpansionLayer(32, 64, 6, 2),
            GatherExpansionLayer(64, 64, 6, 1),
        )
        self.s5_ge_layer = nn.Sequential(
            GatherExpansionLayer(64, 128, 6, 2),
            GatherExpansionLayer(128, 128, 6, 1),
            GatherExpansionLayer(128, 128, 6, 1),
            GatherExpansionLayer(128, 128, 6, 1),

        )
        self.s5_ce_layer = nn.Sequential(
            ContextEmphasis(128)
        )

    def forward(self, x):
        x1 = self.s1s2(x)
        x2 = self.s3_ge_layer(x1)
        x3 = self.s4_ge_layer(x2)
        x4 = self.s5_ge_layer(x3)
        x5 = self.s5_ce_layer(x4)
        return x1, x2, x3, x4, x5

class DetailAggregateLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DetailAggregateLoss, self).__init__()
        
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
        
        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6./10], [3./10], [1./10]],
            dtype=torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor))

    def forward(self, boundary_logits, gtmasks):

        # boundary_logits = boundary_logits.unsqueeze(1)
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)
        
        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)
    
        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')
        
        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
        
        
        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
       
        
        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0
        
        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
        
        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0
        
        # Compress multi-channel boundary_logits to single channel
        if boundary_logits.shape[1] > 1:
            boundary_logits = torch.mean(boundary_logits, dim=1, keepdim=True)
            
        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)
            
        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        return bce_loss,  dice_loss

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
                nowd_params += list(module.parameters())
        return nowd_params