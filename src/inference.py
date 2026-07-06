"""
模型推理脚本

使用方法:
    python inference.py --image test.jpg --weights model.pdparams
    python inference.py --video test.mp4 --weights model.pdparams
"""

import os
import argparse
import cv2
import numpy as np
import paddle
from model import SimpleYOLO


def load_model(model_path):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型权重文件路径
        
    Returns:
        model: 加载好的模型
    """
    model = SimpleYOLO(num_classes=2)
    state_dict = paddle.load(model_path)
    model.set_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path, image_size=416):
    """
    图像预处理
    
    Args:
        image_path: 图像路径
        image_size: 输入图像大小
        
    Returns:
        image: 预处理后的图像张量
        original_image: 原始图像
    """
    # 读取图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    
    # 转换 BGR -> RGB
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # 调整大小
    image = cv2.resize(image, (image_size, image_size))
    
    # 归一化
    image = image.astype('float32') / 255.0
    
    # 转换通道顺序: HWC -> CHW
    image = image.transpose(2, 0, 1)
    
    # 添加批维度
    image = np.expand_dims(image, axis=0)
    
    return paddle.to_tensor(image), original_image


def post_process(output, confidence_threshold=0.5):
    """
    后处理模型输出
    
    Args:
        output: 模型输出，形状 (1, num_classes+5, 13, 13)
        confidence_threshold: 置信度阈值
        
    Returns:
        detections: 检测结果列表，每个元素为 (x, y, w, h, confidence, class_id)
    """
    # 将输出转换为 numpy
    output = output.numpy()
    batch_size = output.shape[0]
    
    detections = []
    
    for b in range(batch_size):
        grid_output = output[b]  # (num_classes+5, 13, 13)
        grid_size = grid_output.shape[1]
        
        for y in range(grid_size):
            for x in range(grid_size):
                # 获取这个网格的预测
                pred = grid_output[:, y, x]
                
                # 提取信息
                x_center = pred[0]
                y_center = pred[1]
                width = pred[2]
                height = pred[3]
                confidence = pred[4]
                
                # 检查置信度
                if confidence < confidence_threshold:
                    continue
                
                # 获取类别
                class_probs = pred[5:]
                class_id = np.argmax(class_probs)
                class_prob = class_probs[class_id]
                
                # 保存检测
                detections.append({
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_prob': class_prob
                })
    
    return detections


def draw_detections(image, detections, class_names=None):
    """
    在图像上绘制检测结果
    
    Args:
        image: 原始图像
        detections: 检测结果列表
        class_names: 类别名称列表
        
    Returns:
        image_with_detections: 绘制了检测结果的图像
    """
    if class_names is None:
        class_names = ['person', 'car']
    
    image_with_detections = image.copy()
    height, width = image.shape[:2]
    
    for detection in detections:
        # 获取边界框坐标
        x_center = detection['x_center'] * width
        y_center = detection['y_center'] * height
        box_width = detection['width'] * width
        box_height = detection['height'] * height
        
        # 计算左上角和右下角坐标
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)
        
        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # 绘制边界框
        color = (0, 255, 0)  # 绿色
        thickness = 2
        cv2.rectangle(image_with_detections, (x1, y1), (x2, y2), color, thickness)
        
        # 绘制标签
        class_id = detection['class_id']
        class_name = class_names[class_id]
        confidence = detection['confidence']
        label = f"{class_name}: {confidence:.2f}"
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            image_with_detections,
            (x1, y1 - label_size[1] - 4),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        cv2.putText(
            image_with_detections,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return image_with_detections


def infer_image(image_path, model_path, output_path=None):
    """
    在单张图像上进行推理
    
    Args:
        image_path: 输入图像路径
        model_path: 模型权重路径
        output_path: 输出图像路径
    """
    print(f"\n📷 推理图像: {image_path}")
    
    # 加载模型
    print("加载模型...")
    model = load_model(model_path)
    
    # 预处理图像
    print("预处理图像...")
    image_tensor, original_image = preprocess_image(image_path)
    
    # 推理
    print("运行推理...")
    with paddle.no_grad():
        output = model(image_tensor)
    
    # 后处理
    print("后处理...")
    detections = post_process(output, confidence_threshold=0.5)
    print(f"检测到 {len(detections)} 个物体")
    
    # 绘制结果
    print("绘制结果...")
    image_with_detections = draw_detections(original_image, detections)
    
    # 保存结果
    if output_path is None:
        output_path = image_path.replace('.jpg', '_detected.jpg')
    
    cv2.imwrite(output_path, image_with_detections)
    print(f"✓ 结果已保存: {output_path}")


def infer_video(video_path, model_path, output_path=None):
    """
    在视频上进行推理
    
    Args:
        video_path: 输入视频路径
        model_path: 模型权重路径
        output_path: 输出视频路径
    """
    print(f"\n🎬 推理视频: {video_path}")
    
    # 加载模型
    print("加载模型...")
    model = load_model(model_path)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    
    # 获取视频信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {frame_width}x{frame_height}, {fps} fps, 总帧数: {total_frames}")
    
    # 创建视频写入器
    if output_path is None:
        output_path = video_path.replace('.mp4', '_detected.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # 处理视频帧
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 保存原始帧
        original_frame = frame.copy()
        
        # 预处理
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.resize(rgb_frame, (416, 416))
        rgb_frame = rgb_frame.astype('float32') / 255.0
        rgb_frame = rgb_frame.transpose(2, 0, 1)
        rgb_frame = np.expand_dims(rgb_frame, axis=0)
        image_tensor = paddle.to_tensor(rgb_frame)
        
        # 推理
        with paddle.no_grad():
            output = model(image_tensor)
        
        # 后处理
        detections = post_process(output, confidence_threshold=0.5)
        
        # 绘制结果
        frame_with_detections = draw_detections(original_frame, detections)
        
        # 写入视频
        out.write(frame_with_detections)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"处理进度: {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    print(f"✓ 视频已保存: {output_path}")


def main():
    """
    主推理函数
    """
    parser = argparse.ArgumentParser(description='YOLO 目标检测推理')
    parser.add_argument('--image', type=str, default=None,
                       help='输入图像路径')
    parser.add_argument('--video', type=str, default=None,
                       help='输入视频路径')
    parser.add_argument('--weights', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出结果路径')
    args = parser.parse_args()
    
    # 检查输入
    if args.image is None and args.video is None:
        print("❌ 错误: 必须指定 --image 或 --video")
        return
    
    if not os.path.exists(args.weights):
        print(f"❌ 错误: 模型文件不存在: {args.weights}")
        return
    
    # 执行推理
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ 错误: 图像文件不存在: {args.image}")
            return
        infer_image(args.image, args.weights, args.output)
    
    if args.video:
        if not os.path.exists(args.video):
            print(f"❌ 错误: 视频文件不存在: {args.video}")
            return
        infer_video(args.video, args.weights, args.output)
    
    print("\n✅ 推理完成！")


if __name__ == '__main__':
    main()
