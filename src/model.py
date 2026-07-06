"""
YOLO 目标检测模型定义

这是一个简化版本的 YOLO 模型，用于教学目的。
实际应用中可以使用更复杂的架构（如 Darknet53, ResNet 等）。n"""

import paddle
import paddle.nn as nn


class SimpleYOLO(nn.Layer):
    """
    简化的 YOLO 目标检测模型
    
    架构:
        输入: (batch_size, 3, 416, 416)
        ↓
        主干网络 (Backbone): 特征提取
        ↓
        颈部网络 (Neck): 特征融合
        ↓
        头部网络 (Head): 检测头
        ↓
        输出: (batch_size, num_classes+5, 13, 13)
    
    输出解释:
        - 13x13: 特征图大小
        - num_classes+5: 每个网格预测的数据
          - 4 维: 边界框位置 (x, y, w, h)
          - 1 维: 置信度
          - num_classes 维: 类别概率
    
    Args:
        num_classes (int): 类别数，默认为 2
    """
    
    def __init__(self, num_classes=2):
        super(SimpleYOLO, self).__init__()
        self.num_classes = num_classes
        
        # =====================
        # 主干网络 (Backbone)
        # =====================
        # 作用: 从输入图像提取特征
        self.backbone = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2D(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),  # 416 -> 208
            
            # Block 2: 32 -> 64
            nn.Conv2D(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),  # 208 -> 104
            
            # Block 3: 64 -> 128
            nn.Conv2D(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),  # 104 -> 52
            
            # Block 4: 128 -> 256
            nn.Conv2D(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),  # 52 -> 26
        )
        
        # =====================
        # 颈部网络 (Neck)
        # =====================
        # 作用: 融合不同尺度的特征
        self.neck = nn.Sequential(
            nn.Conv2D(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
        )
        
        # =====================
        # 头部网络 (Head)
        # =====================
        # 作用: 进行目标检测预测
        self.head = nn.Sequential(
            nn.Conv2D(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            # 最终输出层
            # 输出通道数 = num_classes + 5
            # 5 = x, y, w, h, confidence
            nn.Conv2D(256, num_classes + 5, kernel_size=1),
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, 3, 416, 416)
            
        Returns:
            output: 输出张量，形状为 (batch_size, num_classes+5, 13, 13)
        """
        # 通过主干网络
        backbone_out = self.backbone(x)  # (B, 256, 26, 26)
        
        # 通过颈部网络
        neck_out = self.neck(backbone_out)  # (B, 256, 26, 26)
        
        # 通过头部网络
        output = self.head(neck_out)  # (B, num_classes+5, 13, 13)
        
        return output


class ConvBlock(nn.Layer):
    """
    卷积块: Conv2D + BatchNorm + ReLU
    
    这是一个常用的构建块，可以简化模型定义。
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2D(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ImprovedYOLO(nn.Layer):
    """
    改进版 YOLO 模型，使用 ConvBlock 构建
    
    这个版本的代码更清晰，易于修改和扩展。
    """
    
    def __init__(self, num_classes=2):
        super(ImprovedYOLO, self).__init__()
        self.num_classes = num_classes
        
        # 主干网络
        self.backbone = nn.Sequential(
            ConvBlock(3, 32),      # 416 -> 416
            nn.MaxPool2D(2, 2),    # 416 -> 208
            ConvBlock(32, 64),     # 208 -> 208
            nn.MaxPool2D(2, 2),    # 208 -> 104
            ConvBlock(64, 128),    # 104 -> 104
            nn.MaxPool2D(2, 2),    # 104 -> 52
            ConvBlock(128, 256),   # 52 -> 52
            nn.MaxPool2D(2, 2),    # 52 -> 26
        )
        
        # 颈部网络
        self.neck = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 256),
        )
        
        # 头部网络
        self.head = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 256),
            nn.Conv2D(256, num_classes + 5, kernel_size=1),
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x
