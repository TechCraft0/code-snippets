import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch.onnx

import os
import sys
import argparse

# =========================================================
# 项目路径设置（允许在 scripts/ 下直接运行）
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from model.bisenetv2 import BisenetV2
from model.bisenetv2_stdc import BisenetV2STDC

# 类别颜色定义（BGR）
class_colors = {
    0: (0, 0, 0),         # 背景 - 黑色
    1: (0, 255, 255),     # 可行驶区域 - 黄色
    2: (0, 255, 0),       # 草坪 - 绿色
    3: (255, 0, 0),       # 车道线 - 蓝色
    4: (0, 128, 0),       # 植被 - 深绿色
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(weight_path, num_classes=5, device=device):
    """
    加载 BisenetV2 模型权重
    weight_path: .pth 文件路径
    num_classes: 分类数
    device: 'cpu' 或 'cuda'
    """
    model = BisenetV2STDC(in_channels=3, out_channels=16, n_classes=num_classes, mode="test")
    state = torch.load(weight_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        # 从 checkpoint 中提取模型权重
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict):
        # 直接是 state_dict
        model.load_state_dict(state)
    else:
        # 如果直接保存了整个模型
        model = state

    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, input_size=(512, 512)):
    """
    input_size: (width, height)
    """
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(input_size),   # (W, H)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)  # (1,3,H,W)
    print("After transform, img_tensor.shape:", img_tensor.shape)
    return img_tensor, np.array(img)


def decode_segmentation(segmentation, class_colors):
    h, w = segmentation.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        color_image[segmentation == class_id] = color
    # BGR->RGB for plt
    color_image = color_image[:, :, ::-1]
    return color_image

def compute_iou(pred, label, num_classes=5):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        label_inds = (label == cls)
        intersection = (pred_inds & label_inds).sum()
        union = (pred_inds | label_inds).sum()
        if union == 0:
            ious.append(float('nan'))  # ignore this class
        else:
            ious.append(intersection / union)
    return ious

def infer_and_evaluate(model, image_path, label_path, input_size=(512, 1024), device=device):
    """
    input_size: (width, height)
    """
    img_tensor, original_img = preprocess_image(image_path, input_size)
    img_tensor = img_tensor.to(device)

    label_img = Image.open(label_path)
    label_img = label_img.resize((960, 512), Image.NEAREST) # width height
    label_np = np.array(label_img)
    print("label_np shape:", label_np.shape)

    with torch.no_grad():
        output = model(img_tensor)[0]  # (num_classes, H, W)
        print("model output shape:", output.shape)
        pred = torch.argmax(output, dim=0).cpu().numpy()  # (H, W)
        print("pred shape:", pred.shape)

    # 若shape不一致，可以转置标签
    if pred.shape != label_np.shape:
        print("Warning: pred and label shape mismatch, transposing label.")
        label_np = label_np.T

    ious = compute_iou(pred, label_np)
    print(f"IoU per class: {ious}")
    print(f"Mean IoU: {np.nanmean(ious):.4f}")

    pred_color = decode_segmentation(pred, class_colors)
    pred_color_resized = cv2.resize(pred_color, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 可视化
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_color_resized)
    plt.title("Predicted Segmentation")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(label_img)
    plt.title("Ground Truth")
    plt.axis('off')

    plt.show()

    return ious


if __name__ == "__main__":
    # 示例用法
    input_size = (512, 960)  # (height, width)
    model = load_model("runs/train_003/save_model/model_iter_149000.pth", num_classes=5, device=device)
    ious = infer_and_evaluate(model,
                             "datasets/segdata/images/test/1_2025-05-30_2025-05-30_camera_camera_1748590537_745044.1581_2445360.7154_2.4193.jpg",
                             "datasets/segdata/labels/test/1_2025-05-30_2025-05-30_camera_camera_1748590537_745044.1581_2445360.7154_2.4193.png",
                             input_size=input_size, device=device)