import os

# 设备配置（默认用cuda:0，有需求可改）
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# 路径配置
PATHS = {
    "train_img_path": "datasets/segdata/images/train",
    "train_label_path": "datasets/segdata/labels/train",
    "val_img_path": "datasets/segdata/images/val",
    "val_label_path": "datasets/segdata/labels/val",
    "test_img_path": "datasets/segdata/images/test",
    "test_label_path": "datasets/segdata/labels/test",
    "save_dir": "save_model",
    "log_dir": "logs",
    "visualization_dir": "visualization",
    "name": "train",
    "root_dir": "runs"
}

# 训练参数
TRAIN_PARAMS = {
    "lr": 0.02,
    "batch_size": 14,
    "num_workers": 8,
    "pin_memory": False,
    "drop_last": True,
    "input_size": (1024, 512),  # (width, height)
    "total_iters": 200000,
    "power": 0.9,
    "val_interval": 1000,
    "test_interval": 1000,
    "checkpoint_interval": 1000,
    "mode": "train",
    "crop_size": (1080, 1080),
    "auto_resume": True,  # 自动断点续训
    "enable_visualization": True,  # 启用可视化
    "max_vis_samples": 40,  # 最大可视化样本数
}

# 优化器参数
OPTIMIZER_PARAMS = {
    "momentum": 0.9,
    "weight_decay": 5e-4
}

# 模型参数
MODEL_PARAMS = {
    "in_channels": 3,
    "out_channels": 16,
    "num_classes": 5
}

# 断点文件名
CHECKPOINT_FILENAME = "checkpoint.pth"

CLASS_NAME = {
    0: {"name": "background", "color": (0, 0, 0)},         # 黑色
    1: {"name": "lane", "color": (0, 255, 255)},           # 黄色 (BGR)
    2: {"name": "ground", "color": (255, 255, 0)},         # 青色 (BGR)
    3: {"name": "vegetation", "color": (0, 128, 0)},       # 深绿色
    4: {"name": "lawn", "color": (255, 0, 255)},           # 品红
}

LOSS_PARAMS = {
    "type": "ce" # ce, ohem, focal_loss, dice,
}

OPTIMIZER_PARAMS = "sgd"

LR_SCHEDULER_PARAMS = {
    "type": "poly",  # 支持: 'linear', 'step', 'poly', 'cos', 'warmcos'
    "step_size": 1000,        # 仅 step 有效
    "gamma": 0.1,             # step 的学习率衰减因子
    "power": 0.9,             # poly 衰减指数
    "warmup_iters": 500,      # warmcos 用
    "total_iters": 200000     # warmcos 和 poly 都用，应与TRAIN_PARAMS一致
}


TEST_PARAMS = {
    "batch_size": 1,
    "num_workers": 4,
    "pin_memory": True,
    "input_size": (1024, 512),
    "root_dir": "runs",
    "name": "test",
    "log_dir": "logs",
    "visualization_dir": "visualization",
    "checkpoint_path": "runs/train_001/save_model/model_iter_9800.pth"  # 需要根据实际路径修改
}