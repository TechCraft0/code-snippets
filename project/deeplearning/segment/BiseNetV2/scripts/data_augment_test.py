import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
from torchvision.transforms import ColorJitter
from torchvision import transforms

train_img_path = "segdata/images/train"
train_label_path = "segdata/labels/train"


class LoadImageAndLabels(Dataset):
    def __init__(self, image_paths, label_paths, in_shape: tuple = (512, 1024), mode: str = 'train',
                 crop_size: tuple = (640, 480)):
        super(LoadImageAndLabels, self).__init__()
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.in_shape = tuple(in_shape)
        self.crop_size = crop_size
        self.mode = mode

        if len(self.in_shape) == 3:  # e.g. (C, H, W)
            self.in_shape = (self.in_shape[2], self.in_shape[1])  # (W, H)

        self.img_list, self.label_list = self.load_data()

        self.color_jitter = ColorJitter(
            brightness=0.25,
            contrast=0.2,
            saturation=0.15,
            hue=0.1
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # 1. Load image and label
        if self.mode == 'train' and random.random() < 0.5:  # 50% 使用 mosaic
            image, mask = self.load_mosaic(idx)
        else:
            image = Image.open(self.img_list[idx]).convert('RGB')
            mask = Image.open(self.label_list[idx])

        if self.mode == 'train':
            # Step 0: Color jitter
            image = self.color_jitter(image)
            image = self.apply_gaussian_blur(image)
            image = self.apply_gaussian_noise(image)
            image = self.apply_motion_blur(image)

            # Step 1: data augmentation
            image, mask = self.random_crop(image, mask, self.crop_size)
            image, mask = self.random_horizontal_flip(image, mask)
            image, mask = self.random_resize(image, mask)

            # Step 2: 弹性形变，确保图像和mask尺寸一致
            image, mask = self.apply_elastic_deform(image, mask, alpha=34, sigma=4, p=0.5)

            # Step 3: resize to in_shape
            image = image.resize(self.in_shape, Image.BILINEAR)
            mask = mask.resize(self.in_shape, Image.NEAREST)

        elif self.mode == 'val':
            image = image.resize(self.in_shape, Image.BILINEAR)
            mask = mask.resize(self.in_shape, Image.NEAREST)

        # Step 3: convert to tensor
        image = self.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()  # 语义分割 mask 是整数 class id

        return image, mask

    def to_tensor(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        return transform(image)

    def random_resize(self, image, mask):
        scales = [0.75, 1, 1.25, 1.5, 1.75, 2.0]
        scale = random.choice(scales)

        if isinstance(image, Image.Image):
            w, h = image.size
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)
            mask = mask.resize((new_w, new_h), Image.NEAREST)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return image, mask

    def random_crop(self, image, mask, crop_size):
        crop_w, crop_h = crop_size
        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        if w < crop_w or h < crop_h:
            raise ValueError(f"Crop size ({crop_w}, {crop_h}) exceeds image size ({w}, {h})")

        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)
        crop_box = (x1, y1, x1 + crop_w, y1 + crop_h)
        image = image.crop(crop_box)
        mask = mask.crop(crop_box)

        return image, mask

    def random_horizontal_flip(self, image, mask, p=0.5):
        if random.random() < p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask

    def load_data(self):
        img_names = sorted(os.listdir(self.image_paths))
        mask_names = sorted(os.listdir(self.label_paths))
        img_list = [os.path.join(self.image_paths, name) for name in img_names]
        mask_list = [os.path.join(self.label_paths, name) for name in mask_names]
        return img_list, mask_list

    def load_mosaic(self, index):
        """使用 mosaic 数据增强将四张图像拼成一张图像以及对应的 mask"""
        # 1. mosaic 输出大小（比 crop 大一些防止信息丢失）
        input_h, input_w = 1080, 1920
        mosaic_img = Image.new('RGB', (input_w, input_h), (114, 114, 114))
        mosaic_mask = Image.new('L', (input_w, input_h), 0)

        # 2. 随机中心点坐标（避免都集中在中间）
        yc = random.randint(int(0.25 * input_h), int(0.75 * input_h))
        xc = random.randint(int(0.25 * input_w), int(0.75 * input_w))

        # 3. 拼接四张图片
        indices = [index] + random.choices(range(len(self.img_list)), k=3)
        for i, idx in enumerate(indices):
            img = Image.open(self.img_list[idx]).convert('RGB')
            mask = Image.open(self.label_list[idx])

            # resize 原图像，使其大致匹配 mosaic 区域大小
            img = img.resize((input_w, input_h), Image.BILINEAR)
            mask = mask.resize((input_w, input_h), Image.NEAREST)

            # 确定放置位置
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = 0, 0, xc, yc
                x1b, y1b, x2b, y2b = input_w - xc, input_h - yc, input_w, input_h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, 0, input_w, yc
                x1b, y1b, x2b, y2b = 0, input_h - yc, input_w - xc, input_h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = 0, yc, xc, input_h
                x1b, y1b, x2b, y2b = input_w - xc, 0, input_w, input_h - yc
            else:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, input_w, input_h
                x1b, y1b, x2b, y2b = 0, 0, input_w - xc, input_h - yc

            # crop 源图像和 mask 并粘贴
            img_crop = img.crop((x1b, y1b, x2b, y2b))
            mask_crop = mask.crop((x1b, y1b, x2b, y2b))
            mosaic_img.paste(img_crop, (x1a, y1a, x2a, y2a))
            mosaic_mask.paste(mask_crop, (x1a, y1a, x2a, y2a))

        return mosaic_img, mosaic_mask

    def apply_gaussian_blur(self, image, p=0.3):
        """对图像应用高斯模糊"""
        if random.random() < p:
            radius = random.uniform(0.5, 1.5)  # 模糊程度
            image = image.filter(ImageFilter.GaussianBlur(radius))
        return image

    def apply_gaussian_noise(self, image, p=0.3):
        """对图像添加高斯噪声"""
        if random.random() < p:
            img_array = np.array(image).astype(np.float32)
            noise = np.random.normal(0, 10, img_array.shape)  # 均值0，标准差10
            img_array += noise
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        return image

    def apply_motion_blur(self, image, p=0.3):
        """对图像应用运动模糊"""
        if random.random() < p:
            # 随机选择核的大小（运动强度）
            kernel_size = random.choice([3, 5, 7, 9])
            # 随机选择方向：0=水平，1=垂直
            direction = random.choice(['horizontal', 'vertical'])

            # 创建运动模糊核
            kernel = np.zeros((kernel_size, kernel_size))
            if direction == 'horizontal':
                kernel[kernel_size // 2, :] = np.ones(kernel_size)
            else:
                kernel[:, kernel_size // 2] = np.ones(kernel_size)

            kernel /= kernel_size

            # 转为 NumPy 数组并应用滤波
            img_np = np.array(image)
            blurred = cv2.filter2D(img_np, -1, kernel)
            image = Image.fromarray(np.uint8(blurred))

        return image

    def load_mixup(self, index):
        """MixUp 数据增强（随机混合两张图像和 mask）"""
        # 选两张图像
        index2 = random.randint(0, len(self.img_list) - 1)
        img1 = Image.open(self.img_list[index]).convert("RGB").resize(self.in_shape, Image.BILINEAR)
        mask1 = Image.open(self.label_list[index]).resize(self.in_shape, Image.NEAREST)

        img2 = Image.open(self.img_list[index2]).convert("RGB").resize(self.in_shape, Image.BILINEAR)
        mask2 = Image.open(self.label_list[index2]).resize(self.in_shape, Image.NEAREST)

        # Sample mixup ratio λ
        lam = np.random.beta(0.8, 0.8)  # 可以改成 (0.8, 0.8) 更激进

        # Convert to numpy
        img1 = np.array(img1).astype(np.float32)
        img2 = np.array(img2).astype(np.float32)
        mix_img = lam * img1 + (1 - lam) * img2
        mix_img = np.clip(mix_img, 0, 255).astype(np.uint8)
        mix_img = Image.fromarray(mix_img)

        mask1 = np.array(mask1).astype(np.float32)
        mask2 = np.array(mask2).astype(np.float32)
        mix_mask = lam * mask1 + (1 - lam) * mask2
        mix_mask = Image.fromarray(mix_mask.astype(np.uint8))  # 可以保留 float 用 soft loss

        return mix_img, mix_mask

    def apply_elastic_deform(self, image, mask, alpha=34, sigma=4, p=0.5):
        if random.random() > p:
            return image, mask

        img_np = np.array(image)
        mask_np = np.array(mask)
        shape = img_np.shape[:2]

        dx = (np.random.rand(*shape) * 2 - 1) * alpha
        dy = (np.random.rand(*shape) * 2 - 1) * alpha

        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        img_deformed = cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_deformed = cv2.remap(mask_np, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        image_deformed = Image.fromarray(img_deformed)
        mask_deformed = Image.fromarray(mask_deformed)

        return image_deformed, mask_deformed


# 创建数据集实例
dataset = LoadImageAndLabels(
    image_paths=train_img_path,
    label_paths=train_label_path,
    in_shape=(1024, 512),  # (W, H)
    mode='train',
    crop_size=(640, 640)
)

index = 0
mosaic_img, mosaic_mask = dataset.load_mosaic(index)
mixup_img, mixup_mask = dataset.load_mixup(index)

img = Image.open(dataset.img_list[index]).convert('RGB')
mask = Image.open(dataset.label_list[index])
img_deform, mask_deform = dataset.apply_elastic_deform(img, mask, p=1.0)  # 强制 deform


# 显示函数：将灰度 mask 显示为伪彩色
def decode_mask(mask: Image.Image):
    mask_np = np.array(mask)
    colormap = np.array([
        [0, 0, 0],  # class 0
        [128, 0, 0],  # class 1
        [0, 128, 0],  # class 2
        [128, 128, 0],  # class 3
        [0, 0, 128],  # class 4
    ], dtype=np.uint8)
    mask_color = colormap[mask_np % len(colormap)]
    return mask_color


# 可视化 Mosaic 和 MixUp 结果
plt.figure(figsize=(12, 12))

plt.subplot(3, 2, 1)
plt.title("Mosaic Image")
plt.imshow(mosaic_img)
plt.axis('off')

plt.subplot(3, 2, 2)
plt.title("Mosaic Mask (pseudo color)")
plt.imshow(decode_mask(mosaic_mask))
plt.axis('off')

plt.subplot(3, 2, 3)
plt.title("MixUp Image")
plt.imshow(mixup_img)
plt.axis('off')

plt.subplot(3, 2, 4)
plt.title("MixUp Mask (pseudo color)")
plt.imshow(decode_mask(mixup_mask))
plt.axis('off')

plt.subplot(3, 2, 5)
plt.title("Elastic Deform Image")
plt.imshow(img_deform)
plt.axis('off')

plt.subplot(3, 2, 6)
plt.title("Elastic Deform Mask")
plt.imshow(decode_mask(mask_deform))
plt.axis('off')

plt.tight_layout()
plt.show()