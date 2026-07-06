"""
模型训练脚本

使用方法:
    python train.py
    python train.py --config config/train_config.yaml
    python train.py --epochs 100 --batch-size 32
"""

import os
import sys
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import paddle
import paddle.nn as nn
from paddle.io import DataLoader

from model import SimpleYOLO
from dataset import YOLODataset, SimpleDataset


def load_config(config_path):
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_model(model, save_path):
    """
    保存模型权重
    
    Args:
        model: PaddlePaddle 模型
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    paddle.save(model.state_dict(), save_path)
    print(f"✓ 模型已保存: {save_path}")


def train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, config):
    """
    训练一个 epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        loss_fn: 损失函数
        optimizer: 优化器
        epoch: 当前 epoch 数
        config: 配置字典
        
    Returns:
        平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(enumerate(train_loader()), 
                       total=len(train_loader),
                       desc=f"Epoch {epoch+1}/{config['epochs']}")
    
    for batch_id, (images, targets) in progress_bar:
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        # 简化版本：直接使用 MSE 损失
        loss = nn.functional.mse_loss(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        # 记录损失
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}'
        })
        
        # 定期打印日志
        if (batch_id + 1) % config.get('log_interval', 10) == 0:
            print(f"  Batch {batch_id+1}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """
    主训练函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLO 目标检测模型训练')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                       help='训练配置文件路径')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（覆盖配置文件）')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批大小（覆盖配置文件）')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率（覆盖配置文件）')
    parser.add_argument('--use-test-data', action='store_true',
                       help='使用随机生成的测试数据')
    args = parser.parse_args()
    
    # 加载配置
    print("📋 加载配置...")
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    
    # 设置设备
    device = config.get('device', 'gpu')
    if device == 'gpu':
        paddle.set_device(f"gpu:{config.get('gpu_id', 0)}")
        print(f"✓ 使用 GPU: {config.get('gpu_id', 0)}")
    else:
        paddle.set_device('cpu')
        print("✓ 使用 CPU")
    
    # 创建模型
    print("\n🤖 创建模型...")
    model = SimpleYOLO(num_classes=config['num_classes'])
    print(f"✓ 模型创建成功")
    print(f"  类别数: {config['num_classes']}")
    
    # 创建优化器
    print("\n⚙️  配置优化器...")
    if config['optimizer'] == 'Adam':
        optimizer = paddle.optimizer.Adam(
            learning_rate=config['learning_rate'],
            parameters=model.parameters()
        )
    else:
        optimizer = paddle.optimizer.SGD(
            learning_rate=config['learning_rate'],
            momentum=config.get('momentum', 0.9),
            parameters=model.parameters()
        )
    print(f"✓ 优化器: {config['optimizer']}")
    
    # 创建损失函数
    loss_fn = nn.MSELoss()
    
    # 加载数据
    print("\n📦 加载数据...")
    if args.use_test_data:
        # 使用随机生成的测试数据
        train_dataset = SimpleDataset(
            num_samples=100,
            image_size=config['image_size']
        )
        print(f"✓ 使用测试数据集 (100 样本)")
    else:
        # 使用真实数据
        train_dataset = YOLODataset(
            image_dir=config['train_image_dir'],
            label_dir=config['train_label_dir'],
            image_size=config['image_size']
        )
        print(f"✓ 从 {config['train_image_dir']} 加载训练数据")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    print(f"✓ 数据加载器创建成功")
    print(f"  批大小: {config['batch_size']}")
    print(f"  批次数: {len(train_loader)}")
    
    # 开始训练
    print("\n🚀 开始训练...\n")
    print("="*60)
    
    best_loss = float('inf')
    patience = config.get('early_stopping_patience', 10)
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # 训练一个 epoch
        avg_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, epoch, config
        )
        
        print(f"Epoch {epoch+1}/{config['epochs']} - Avg Loss: {avg_loss:.4f}")
        
        # 保存最好的模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            checkpoint_path = os.path.join(
                config['output_dir'],
                'yolo_best.pdparams'
            )
            save_model(model, checkpoint_path)
        else:
            patience_counter += 1
        
        # 定期保存检查点
        if (epoch + 1) % config.get('save_interval', 5) == 0:
            checkpoint_path = os.path.join(
                config['output_dir'],
                f'yolo_epoch_{epoch+1}.pdparams'
            )
            save_model(model, checkpoint_path)
        
        # 早停
        if config.get('early_stopping', False) and patience_counter >= patience:
            print(f"\n⚠️  早停: 在 {patience} 个 epoch 后没有改进")
            break
        
        print("-" * 60)
    
    print("\n" + "="*60)
    print("✅ 训练完成！")
    print(f"最好模型保存在: {os.path.join(config['output_dir'], 'yolo_best.pdparams')}")


if __name__ == '__main__':
    main()
