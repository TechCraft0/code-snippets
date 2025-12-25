import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# 类别颜色（BGR）
class_colors = {
    0: (0, 0, 0),
    1: (0, 255, 255),
    2: (0, 255, 0),
    3: (255, 0, 0),
    4: (0, 128, 0),
}

input_size = (512, 960) # height, width

def decode_segmentation(segmentation, class_colors):
    h, w = segmentation.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        color_image[segmentation == class_id] = color
    return color_image[:, :, ::-1]  # BGR -> RGB

def preprocess_image(image_path, input_size=input_size):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).numpy()
    return img_tensor, np.array(img)

def merge_four_images_and_masks(image_paths, mask_paths):
    imgs = []
    masks = []

    for img_path, mask_path in zip(image_paths, mask_paths):
        img = cv2.imread(img_path)                      # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 2, h // 2))          # 缩小一半
        imgs.append(img)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (w // 2, h // 2),
                           interpolation=cv2.INTER_NEAREST)
        masks.append(mask)

    # 2x2 拼接
    top_img = np.hstack([imgs[0], imgs[1]])
    bot_img = np.hstack([imgs[2], imgs[3]])
    merged_img = np.vstack([top_img, bot_img])

    top_mask = np.hstack([masks[0], masks[1]])
    bot_mask = np.hstack([masks[2], masks[3]])
    merged_mask = np.vstack([top_mask, bot_mask])

    return merged_img, merged_mask


# 模型加载
onnx_path = "weights/bisenetv2_5_cls_stdc_960_512.onnx"
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])


# ===== 单张图像路径 =====
image_paths = [
    'datasets/segdata/images/test/1_2025-05-30_2025-05-30_camera_camera_1748590537_745044.1581_2445360.7154_2.4193.jpg',
    'datasets/segdata/images/test/1_2025-05-30_2025-05-30_camera_camera_1748591901_744651.2723_2445540.9775_2.8620.jpg',
    'datasets/segdata/images/test/1_2025-05-30_2025-05-30_camera_camera_1748592243_744566.1845_2445156.8405_1.7973.jpg',
    'datasets/segdata/images/test/1_2025-05-30_2025-05-30_camera_camera_1748592664_744831.0892_2445452.6143_2.1471.jpg',
]

mask_paths = [
    'datasets/segdata/labels/test/1_2025-05-30_2025-05-30_camera_camera_1748590537_745044.1581_2445360.7154_2.4193.png',
    'datasets/segdata/labels/test/1_2025-05-30_2025-05-30_camera_camera_1748591901_744651.2723_2445540.9775_2.8620.png',
    'datasets/segdata/labels/test/1_2025-05-30_2025-05-30_camera_camera_1748592243_744566.1845_2445156.8405_1.7973.png',
    'datasets/segdata/labels/test/1_2025-05-30_2025-05-30_camera_camera_1748592664_744831.0892_2445452.6143_2.1471.png',
]

# ===== 合并图像预测（由四张单图拼接）=====
merged_rgb_img, merged_gt = merge_four_images_and_masks(
    image_paths, mask_paths
)

# 送入模型（使用原 preprocess，只是输入换成拼接图）
merged_pil = Image.fromarray(merged_rgb_img)
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(merged_pil).unsqueeze(0).numpy()

merged_out = session.run(None, {session.get_inputs()[0].name: input_tensor})
merged_pred_mask = merged_out[0][0, 0].astype(np.uint8)
merged_pred_color = decode_segmentation(merged_pred_mask, class_colors)

# GT resize 对齐预测
merged_gt = cv2.resize(
    merged_gt,
    (merged_pred_mask.shape[1], merged_pred_mask.shape[0]),
    interpolation=cv2.INTER_NEAREST
)
merged_gt_color = decode_segmentation(merged_gt, class_colors)


# ===== 单张图像、标签、预测结果列表 =====
single_imgs = []
single_gt_colors = []
single_pred_colors = []

for img_path, mask_path in zip(image_paths, mask_paths):
    input_tensor, rgb_img = preprocess_image(img_path)
    single_imgs.append(rgb_img)

    # 读取并 decode mask
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.resize(gt_mask, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_color = decode_segmentation(gt_mask, class_colors)
    single_gt_colors.append(gt_color)

    # 推理并 decode 预测
    output = session.run(None, {session.get_inputs()[0].name: input_tensor})
    pred_mask = output[0][0, 0].astype(np.uint8)
    pred_color = decode_segmentation(pred_mask, class_colors)
    single_pred_colors.append(pred_color)

plt.figure(figsize=(18, 14))

# 第一行: merged image, gt, pred + 一个空格
plt.subplot(4, 4, 1)
plt.imshow(merged_rgb_img)
plt.title("Merged Image")
plt.axis('off')

plt.subplot(4, 4, 2)
plt.imshow(merged_gt_color)
plt.title("Merged GT")
plt.axis('off')

plt.subplot(4, 4, 3)
plt.imshow(merged_pred_color)
plt.title("Merged Prediction")
plt.axis('off')

plt.subplot(4, 4, 4)
plt.axis('off')  # 填充空位

# 第二行: 4张单图原图
for i in range(4):
    plt.subplot(4, 4, 5 + i)
    plt.imshow(single_imgs[i])
    plt.title(f"Image {i+1}")
    plt.axis('off')

# 第三行: 4张标签
for i in range(4):
    plt.subplot(4, 4, 9 + i)
    plt.imshow(single_gt_colors[i])
    plt.title(f"GT {i+1}")
    plt.axis('off')

# 第四行: 4张预测
for i in range(4):
    plt.subplot(4, 4, 13 + i)
    plt.imshow(single_pred_colors[i])
    plt.title(f"Pred {i+1}")
    plt.axis('off')

plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.02, hspace=0.03)
plt.show()
