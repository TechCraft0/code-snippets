import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from rknn.api import RKNN

# 类别颜色定义（BGR）
class_colors = {
    0: (0, 0, 0),         # 背景 - 黑色
    1: (0, 255, 255),     # 可行驶区域 - 黄色
    2: (0, 255, 0),       # 草坪 - 绿色
    3: (255, 0, 0),       # 车道线 - 蓝色
    4: (0, 128, 0),       # 植被 - 深绿色
}

def preprocess_image(image_path, input_size=(512, 1024)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size, Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0

    # 归一化，保持和训练一致
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_np - mean) / std

    # HWC -> CHW
    img_chw = img_norm.transpose(2, 0, 1)

    # 添加 batch 维度
    input_tensor = np.expand_dims(img_chw, axis=0)
    return input_tensor, np.array(img)

def decode_segmentation(segmentation, class_colors):
    h, w = segmentation.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        color_image[segmentation == class_id] = color
    # BGR->RGB 转换以便用 plt 显示
    return color_image[:, :, ::-1]

def rknn_inference(rknn_model_path, image_path, input_size=(512, 1024)):
    rknn = RKNN()

    print('--> Loading RKNN model')
    ret = rknn.load_rknn(rknn_model_path)
    if ret != 0:
        print('Load RKNN model failed')
        return

    print('--> Initializing runtime environment')
    ret = rknn.init_runtime(target="rk3588")
    if ret != 0:
        print('Init runtime environment failed')
        return

    # 预处理图片
    input_tensor, original_img = preprocess_image(image_path, input_size)

    print('--> Running inference')
    outputs = rknn.inference(inputs=[input_tensor])

    # 假设输出是 shape = (1, 1, H, W)，且 dtype 是 int 或 uint8
    pred_mask = outputs[0][0, 0, :, :].astype(np.uint8)

    # 解码颜色
    pred_color = decode_segmentation(pred_mask, class_colors)

    # 可视化
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pred_color)
    plt.title("RKNN Segmentation")
    plt.axis('off')

    plt.show()

    rknn.release()

if __name__ == "__main__":
    rknn_model_path = "/home/only/company/yolov5-7.0/weights/segment/bisenetv2_5class.rknn"    # 你的 RKNN 模型路径
    test_image_path = "/home/only/company/yolov5-7.0/datasets/segment/segdata/images/val/2024-10-02_data_cleansing_8.jpg"  # 测试图片路径

    rknn_inference(rknn_model_path, test_image_path)