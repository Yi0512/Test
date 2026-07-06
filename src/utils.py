"""
工具函数
"""

import numpy as np
import cv2


def load_class_names(class_names_file):
    """
    从文件加载类别名称
    
    Args:
        class_names_file: 类别文件路径（每行一个类别名）
        
    Returns:
        class_names: 类别名称列表
    """
    with open(class_names_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


def compute_iou(box1, box2):
    """
    计算 IoU (Intersection over Union)
    
    Args:
        box1: 边界框 1，格式 (x1, y1, x2, y2)
        box2: 边界框 2，格式 (x1, y1, x2, y2)
        
    Returns:
        iou: IoU 值
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    # 计算并集
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    # 计算 IoU
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou


def non_max_suppression(detections, iou_threshold=0.5):
    """
    非极大值抑制 (NMS)
    
    Args:
        detections: 检测结果列表
        iou_threshold: IoU 阈值
        
    Returns:
        filtered_detections: 过滤后的检测结果
    """
    if len(detections) == 0:
        return []
    
    # 按置信度排序
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    filtered_detections = []
    
    for detection in detections:
        is_duplicate = False
        
        for kept_detection in filtered_detections:
            # 转换为边界框坐标
            box1 = (
                detection['x_center'] - detection['width'] / 2,
                detection['y_center'] - detection['height'] / 2,
                detection['x_center'] + detection['width'] / 2,
                detection['y_center'] + detection['height'] / 2,
            )
            box2 = (
                kept_detection['x_center'] - kept_detection['width'] / 2,
                kept_detection['y_center'] - kept_detection['height'] / 2,
                kept_detection['x_center'] + kept_detection['width'] / 2,
                kept_detection['y_center'] + kept_detection['height'] / 2,
            )
            
            iou = compute_iou(box1, box2)
            
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_detections.append(detection)
    
    return filtered_detections


def create_annotation_file(image_file, annotations, output_file):
    """
    创建 YOLO 格式的注解文件
    
    Args:
        image_file: 图像文件路径
        annotations: 注解列表，每个元素为 (class_id, x_center, y_center, width, height)
        output_file: 输出注解文件路径
    """
    with open(output_file, 'w') as f:
        for annotation in annotations:
            line = ' '.join(map(str, annotation))
            f.write(line + '\n')


def draw_bounding_boxes(image, boxes, labels=None, colors=None):
    """
    在图像上绘制边界框
    
    Args:
        image: 输入图像
        boxes: 边界框列表，格式 [(x1, y1, x2, y2), ...]
        labels: 标签列表（可选）
        colors: 颜色列表（可选）
        
    Returns:
        image_with_boxes: 绘制了边界框的图像
    """
    image_with_boxes = image.copy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        color = colors[i] if colors else (0, 255, 0)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        if labels:
            label = labels[i]
            cv2.putText(image_with_boxes, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image_with_boxes
