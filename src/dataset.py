"""
数据集加载器

实现 YOLO 格式的数据加载和预处理
"""

import os
import cv2
import numpy as np
from paddle.io import Dataset


class YOLODataset(Dataset):
    """
    YOLO 格式的数据集加载器
    
    数据目录结构:
        dataset/
        ├── images/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── labels/
            ├── image1.txt
            ├── image2.txt
            └── ...
    
    标签文件格式 (YOLO 格式):
        每一行表示一个物体：
        class_id x_center y_center width height
        
        其中：
        - class_id: 类别 ID (从 0 开始)
        - x_center, y_center: 边界框中心坐标（归一化到 0-1）
        - width, height: 边界框宽度和高度（归一化到 0-1）
        
        例如:
        0 0.5 0.5 0.3 0.4
        1 0.7 0.3 0.2 0.2
    
    Args:
        image_dir (str): 图像目录路径
        label_dir (str): 标签目录路径
        image_size (int): 输入图像大小，默认 416
        transform (callable): 数据增强函数，默认 None
    """
    
    def __init__(self, image_dir, label_dir, image_size=416, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        
        # 获取所有图像文件名
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                   if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            image: 预处理后的图像张量，形状 (3, H, W)
            target: 目标张量，形状 (S, S, 5+num_classes)
                   其中 S 是网格大小，5 表示 (x, y, w, h, confidence)
        """
        # 1. 加载图像
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        
        # 转换 BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # 2. 预处理图像
        # 调整大小
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # 归一化到 [0, 1]
        image = image.astype('float32') / 255.0
        
        # 转换通道顺序: HWC -> CHW
        image = image.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        
        # 3. 加载标签
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        # 创建目标张量: (13, 13, 5+num_classes)
        # 假设网格大小为 13 (416 / 32)
        grid_size = 13
        num_classes = 2  # 可以从配置文件读取
        target = np.zeros((grid_size, grid_size, 5 + num_classes), dtype='float32')
        
        # 读取标签
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # 计算所在网格位置
                grid_x = int(x_center * grid_size)
                grid_y = int(y_center * grid_size)
                
                # 确保在有效范围内
                grid_x = min(grid_x, grid_size - 1)
                grid_y = min(grid_y, grid_size - 1)
                
                # 设置目标值
                target[grid_y, grid_x, 0] = x_center  # x_center
                target[grid_y, grid_x, 1] = y_center  # y_center
                target[grid_y, grid_x, 2] = width     # width
                target[grid_y, grid_x, 3] = height    # height
                target[grid_y, grid_x, 4] = 1.0       # confidence
                target[grid_y, grid_x, 5 + class_id] = 1.0  # class probability
        
        # 4. 数据增强（如果提供了 transform）
        if self.transform:
            image = self.transform(image)
        
        return image, target


class SimpleDataset(Dataset):
    """
    简化版数据集，仅用于测试
    """
    
    def __init__(self, num_samples=100, image_size=416):
        self.num_samples = num_samples
        self.image_size = image_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机图像
        image = np.random.rand(3, self.image_size, self.image_size).astype('float32')
        
        # 生成随机目标
        grid_size = 13
        num_classes = 2
        target = np.zeros((grid_size, grid_size, 5 + num_classes), dtype='float32')
        
        # 随机添加一些物体
        for _ in range(np.random.randint(1, 3)):
            grid_x = np.random.randint(0, grid_size)
            grid_y = np.random.randint(0, grid_size)
            class_id = np.random.randint(0, num_classes)
            
            target[grid_y, grid_x, 0] = np.random.rand()
            target[grid_y, grid_x, 1] = np.random.rand()
            target[grid_y, grid_x, 2] = np.random.rand() * 0.5
            target[grid_y, grid_x, 3] = np.random.rand() * 0.5
            target[grid_y, grid_x, 4] = 1.0
            target[grid_y, grid_x, 5 + class_id] = 1.0
        
        return image, target
