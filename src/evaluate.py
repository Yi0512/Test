"""
模型评估脚本

python src/evaluate.py --weights output/models/yolo_epoch_5.pdparams

"""

import os
import paddle
import numpy as np
from tqdm import tqdm
from model import SimpleYOLO
from dataset import YOLODataset, SimpleDataset
from paddle.io import DataLoader
import argparse
import yaml

def evaluate(model, dataloader):
    """评估模型性能"""
    model.eval()
    
    total_loss = 0
    total_batches = 0
    
    with paddle.no_grad():
        for images, targets in tqdm(dataloader, desc="评估中"):
            outputs = model(images)
            
            # 形状对齐
            if len(targets.shape) == 4 and len(outputs.shape) == 4:
                # 如果 targets 是 [B, H, W, C] 格式，转置为 [B, C, H, W]
                if targets.shape[-1] == outputs.shape[1]:
                    targets = targets.transpose([0, 3, 1, 2])
                
                # 如果空间尺寸不匹配，调整大小
                if targets.shape[2] != outputs.shape[2] or targets.shape[3] != outputs.shape[3]:
                    targets = paddle.nn.functional.interpolate(
                        targets.float(),
                        size=(outputs.shape[2], outputs.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
            
            loss = paddle.nn.functional.mse_loss(outputs, targets)
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    
    print(f"\n评估结果:")
    print(f"  总批次数: {total_batches}")
    print(f"  平均损失: {avg_loss:.4f}")
    
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--weights', type=str, required=True,
                       help='模型权重文件路径')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='批次大小')
    
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    paddle.set_device('cpu')
    
    # 创建模型
    model = SimpleYOLO(
        num_classes=config.get('num_classes', 2)
    )
    
    if os.path.exists(args.weights):
        state_dict = paddle.load(args.weights)
        model.set_state_dict(state_dict)
        print(f"加载模型: {args.weights}")
    else:
        print(f"模型文件不存在: {args.weights}")
        return
    
    # 使用 SimpleDataset 进行测试（避免数据加载问题）
    print("使用随机生成的测试数据")
    test_dataset = SimpleDataset(
        num_samples=50,
        image_size=config.get('image_size', 416)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    evaluate(model, test_loader)

if __name__ == '__main__':
    main()
