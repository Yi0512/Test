# Paddle-YOLO 初学者项目

这是一个为初学者设计的 AI 模型训练项目，结合 **PaddlePaddle** 框架和 **YOLO** 目标检测算法。

## 🎓 项目特点

- ✅ **简化的 YOLO 实现** - 易于理解的模型结构
- ✅ **完整的训练流程** - 从数据加载到推理
- ✅ **详细的中文注释** - 适合初学者学习
- ✅ **配置化管理** - 易于调整参数
- ✅ **交互式教程** - Jupyter Notebook 示例

## 🚀 快速开始

### 1. 环境搭建

```bash
# 克隆项目
git clone https://github.com/Yi0512/Test.git
cd Test

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

```bash
# 使用测试数据快速开始
python src/train.py --use-test-data --epochs 10
```

### 3. 开始训练

```bash
# 使用默认配置训练
python src/train.py

# 或使用自定义参数
python src/train.py --epochs 100 --batch-size 32 --lr 0.0005
```

### 4. 模型推理

```bash
# 在单个图像上运行推理
python src/inference.py --image test_image.jpg --weights output/models/yolo_best.pdparams

# 在视频上运行推理
python src/inference.py --video test_video.mp4 --weights output/models/yolo_best.pdparams
```

## 📁 项目结构

```
Test/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖包列表
│
├── data/
│   ├── train/
│   │   ├── images/           # 训练图像
│   │   └── labels/           # 训练标签
│   └── val/
│       ├── images/           # 验证图像
│       └── labels/           # 验证标签
│
├── src/
│   ├── model.py              # YOLO 模型定义
│   ├── dataset.py            # 数据加载器
│   ├── train.py              # 训练脚本
│   ├── inference.py          # 推理脚本
│   └── utils.py              # 工具函数
│
├── config/
│   ├── data_config.yaml      # 数据集配置
│   └── train_config.yaml     # 训练参数配置
│
└── output/
    └── models/               # 保存训练的模型
```

## 📚 学习路径

1. **了解基础** - 阅读 README 和代码注释
2. **运行示例** - 使用测试数据进行训练
3. **调整参数** - 修改配置文件实验不同参数
4. **使用真实数据** - 用你自己的数据集训练
5. **深入学习** - 阅读源代码实现细节

## 🛠️ 配置说明

### train_config.yaml

```yaml
epochs: 50                  # 训练轮数
batch_size: 16              # 批大小
learning_rate: 0.001        # 学习率
image_size: 416             # 输入图像大小
```

**调整建议：**
- GPU 内存小？减小 `batch_size`（如 8 或 4）
- 训练不收敛？降低 `learning_rate`（如 0.0001）
- 过拟合？增加 `epochs` 并启用 `augmentation`

## 🎯 YOLO 基础知识

### 什么是 YOLO？

**YOLO（You Only Look Once）** 是一种实时目标检测算法，特点是：n- **一次检测** - 在一个前向传播中检测所有物体
- **实时性强** - 运行速度快
- **准确度高** - 特别是对大物体

### YOLO 输出格式

对于每个检测到的物体输出：
- `x_center, y_center` - 边界框中心坐标
- `width, height` - 边界框宽度和高度
- `confidence` - 检测置信度
- `class_probabilities` - 每个类别的概率

### 标签格式

YOLO 标签文件格式（txt）：
```
class_id x_center y_center width height
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.2
```

其中所有坐标都归一化到 [0, 1]。

## 🐛 常见问题

### Q1: 显存不足怎么办？

**A:** 减小 `batch_size` 或 `image_size`

```yaml
batch_size: 8          # 从 16 改为 8
image_size: 320        # 从 416 改为 320
```

### Q2: 训练很慢怎么办？

**A:** 检查是否使用了 GPU

```python
import paddle
print(paddle.device.get_device())  # 应该显示 'gpu:0'
```

### Q3: 损失值不下降？

**A:** 尝试以下方案：
- 降低学习率
- 增加训练轮数
- 检查数据标注是否正确

## 📖 推荐资源

- [PaddlePaddle 官方文档](https://www.paddlepaddle.org.cn/)
- [YOLO 论文解读](https://arxiv.org/abs/1506.02640)
- [目标检测入门指南](https://github.com/ultralytics/yolov5)

## 📝 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**Happy Learning! 🎉**
