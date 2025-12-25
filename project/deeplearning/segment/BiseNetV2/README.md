# BiseNetV2 — 语义分割

## 简要说明
这是 BiseNetV2 语义分割模型的完整实现，支持轻量级实时语义分割。本项目提供完整的训练、验证、测试和可视化功能，包含自动化实验管理和断点续训功能。

## 项目特性
- 🚀 完整的 BiseNetV2 模型实现
- 📊 自动化训练监控和日志记录
- 🔄 断点续训功能
- 🎨 实时可视化结果生成
- 📈 多种学习率调度策略
- 🔍 验证集监控 + 测试集评估
- 📁 自动实验目录管理

## 目录结构
```
BiseNetV2/
├── cfg/                    # 配置文件
│   └── config.py          # 主配置文件（路径、模型、训练参数等）
├── data/                   # 数据处理模块
│   └── data_load.py       # 数据加载器和预处理
├── model/                  # 模型定义
│   └── bisenetv2.py       # BiseNetV2 网络架构
├── tools/                  # 核心工具脚本
│   ├── train.py           # 训练脚本（支持断点续训）
│   └── val.py             # 验证和指标计算
├── utils/                  # 辅助工具
│   ├── common.py          # 通用函数（目录创建等）
│   └── visualization.py   # 可视化结果生成
└── README.md              # 项目说明
```

### 各模块详细说明

#### cfg/ - 配置管理
- `config.py`: 统一配置文件，包含：
  - `PATHS`: 数据路径、保存路径配置
  - `TRAIN_PARAMS`: 训练参数（学习率、批次大小、迭代次数等）
  - `MODEL_PARAMS`: 模型参数（输入通道、类别数等）
  - `LOSS_PARAMS`: 损失函数配置
  - `OPTIMIZER_PARAMS`: 优化器选择
  - `LR_SCHEDULER_PARAMS`: 学习率调度参数
  - `CLASS_NAME`: 类别名称和颜色定义

#### data/ - 数据处理
- `data_load.py`: 数据加载和预处理
  - `LoadImageAndLabels`: 图像和标签加载类
  - 支持训练时数据增强和验证时标准化
  - 自动处理图像尺寸调整和归一化

#### model/ - 模型架构
- `bisenetv2.py`: BiseNetV2 网络实现
  - 轻量级双分支架构
  - 支持多尺度特征融合
  - 包含主分割头和辅助分割头

#### tools/ - 核心功能
- `train.py`: 完整训练流程
  - 自动实验目录创建（带编号避免覆盖）
  - 支持多种优化器（SGD、Adam、AdamW）
  - 多种学习率调度（linear、step、poly、cos、warmcos）
  - 断点续训功能
  - 训练过程中验证监控
  - 定期测试评估和可视化
  - 详细日志记录（支持表情符号）
  - tqdm 进度条显示
- `val.py`: 验证和测试
  - `validate()`: 计算验证/测试损失和指标
  - `compute_metrics()`: mIoU 和像素准确率计算

#### utils/ - 辅助工具
- `common.py`: 通用函数
  - `create_experiment_dirs()`: 自动创建带编号的实验目录
- `visualization.py`: 结果可视化
  - `save_segmentation_results()`: 生成原图-标签-预测对比图
  - 支持自定义类别颜色
  - 自动反归一化图像显示
- `plot_curves.py`: 训练曲线绘制
  - `plot_training_curves()`: 绘制训练损失和验证指标曲线
  - 支持训练损失、验证损失、mIoU、像素准确率曲线

## 环境与依赖

### 系统要求
- Python 3.8+
- CUDA 支持的 GPU（推荐）

### 核心依赖
```bash
pip install torch torchvision numpy opencv-python tqdm albumentations matplotlib
```

### 完整依赖列表
- `torch`: 深度学习框架
- `torchvision`: 图像处理和模型
- `numpy`: 数值计算
- `opencv-python`: 图像处理和可视化
- `tqdm`: 进度条显示
- `albumentations`: 数据增强（需要 pydantic>=2.0）
- `matplotlib`: 绘制训练曲线

### 安装说明
如果遇到 albumentations 兼容性问题：
```bash
# 方法1：升级 pydantic
pip install pydantic>=2.0

# 方法2：降级 albumentations
pip install albumentations==1.3.1
```

## 数据集准备

### 支持格式
- 语义分割数据集（Cityscapes、Pascal VOC、自定义数据集）
- 图像格式：JPG、PNG
- 标签格式：PNG（像素值对应类别ID）

### 数据组织结构
```
dataset/
├── train/
│   ├── images/     # 训练图像
│   └── masks/      # 训练标签
├── val/
│   ├── images/     # 验证图像
│   └── masks/      # 验证标签
└── test/
    ├── images/     # 测试图像
    └── masks/      # 测试标签
```

### 配置数据路径
在 `cfg/config.py` 中修改 `PATHS` 配置：
```python
PATHS = {
    'train_img_path': '/path/to/dataset/train/images',
    'train_label_path': '/path/to/dataset/train/masks',
    'val_img_path': '/path/to/dataset/val/images',
    'val_label_path': '/path/to/dataset/val/masks',
    'test_img_path': '/path/to/dataset/test/images',
    'test_label_path': '/path/to/dataset/test/masks',
    # 其他配置...
}
```

## 训练

### 基本训练命令
```bash
# 默认训练（自动断点续训）
python tools/train.py

# 强制从头开始训练
python tools/train.py --no-resume

# 显式启用断点续训
python tools/train.py --resume
```

### 训练特性
- **自动实验管理**: 自动创建带编号的实验目录（如 `exp_001`、`exp_002`）
- **断点续训**: 自动检测最新检查点并继续训练
- **实时监控**: tqdm 进度条 + 详细日志记录
- **定期验证**: 可配置验证间隔（默认1000次迭代）
- **测试评估**: 可配置测试间隔（默认2000次迭代）
- **自动可视化**: 测试时自动生成预测结果对比图
- **训练曲线**: 训练结束后自动生成损失和指标曲线图

### 训练配置
在 `cfg/config.py` 中配置训练参数：
```python
TRAIN_PARAMS = {
    "total_iters": 10000,      # 总迭代次数
    "batch_size": 8,           # 批次大小
    "lr": 0.001,               # 学习率
    "val_interval": 500,       # 验证间隔
    "test_interval": 1000,     # 测试间隔
    "auto_resume": True,       # 自动断点续训
    # 更多参数...
}
```

### 断点续训控制
- **自动模式**: 默认启用，自动检测并恢复最新checkpoint
- **命令行控制**: 使用 `--resume` 或 `--no-resume` 参数
- **配置文件**: 设置 `auto_resume: False` 禁用自动续训

### 输出结构
训练会自动创建以下目录结构：
```
experiments/
└── exp_001/
    ├── models/         # 模型检查点
    ├── logs/          # 训练日志
    └── visualizations/ # 可视化结果
        ├── iter_2000/  # 按迭代次数的预测结果
        ├── iter_4000/
        └── training_curves.png  # 训练曲线图
```

## 评估与测试

### 自动评估
训练过程中会自动进行：
- **验证集评估**: 监控训练进度，不生成可视化
- **测试集评估**: 评估模型性能，生成可视化结果

### 评估指标
- **mIoU**: 平均交并比
- **Pixel Accuracy**: 像素准确率
- **Loss**: 验证/测试损失

### 独立评估脚本
```python
from tools.val import validate

# 加载模型和数据
val_loss, miou, pixel_acc = validate(
    model, val_loader, loss_function, device, num_classes
)
print(f"mIoU: {miou:.4f}, Pixel Acc: {pixel_acc:.4f}")
```

## 可视化功能

### 自动可视化
训练过程中会自动生成测试集可视化结果：
- 原图 + 真实标签 + 预测结果的三联图
- 按迭代次数保存（如 `iter_2000/`、`iter_4000/`）
- 支持自定义类别颜色

### 训练曲线
训练结束后会自动生成训练曲线图：
- **训练损失曲线**: 显示训练过程中的损失变化
- **验证/测试损失曲线**: 对比验证和测试集上的损失
- **mIoU 曲线**: 显示验证和测试集上的 mIoU 指标变化
- **像素准确率曲线**: 显示验证和测试集上的像素准确率变化

曲线图保存为 `training_curves.png`，包含 2x2 子图布局，方便对比分析。

### 可视化配置
在 `cfg/config.py` 中配置类别颜色：
```python
CLASS_NAME = {
    0: {'name': '背景', 'color': (0, 0, 0)},
    1: {'name': '道路', 'color': (128, 64, 128)},
    2: {'name': '建筑', 'color': (70, 70, 70)},
    # 更多类别...
}
```

### 手动生成可视化
```python
from utils.visualization import save_segmentation_results

save_segmentation_results(
    model, test_loader, device, "custom_name", 
    class_colors, save_dir
)
```

## 模型与检查点

### 自动保存
- 每1000次迭代自动保存检查点
- 检查点包含：模型权重、优化器状态、调度器状态、迭代计数
- 文件命名：`model_iter_1000.pth`、`model_iter_2000.pth`

### 断点续训
- 自动检测最新检查点
- 恢复训练状态（模型、优化器、学习率调度器）
- 从中断处继续训练

### 检查点结构
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'iter_count': iter_count
}
```

## 配置系统

### 统一配置管理
所有配置集中在 `cfg/config.py`：

```python
# 路径配置
PATHS = {
    'root_dir': './experiments',
    'name': 'bisenetv2_exp',
    # 数据路径...
}

# 训练参数
TRAIN_PARAMS = {
    'total_iters': 10000,
    'batch_size': 8,
    'lr': 0.001,
    # 更多参数...
}

# 模型参数
MODEL_PARAMS = {
    'in_channels': 3,
    'num_classes': 19,
    # 更多参数...
}

# 学习率调度
LR_SCHEDULER_PARAMS = {
    'type': 'poly',  # linear/step/poly/cos/warmcos
    'power': 0.9,
    # 更多参数...
}
```

### 实验复现
- 自动保存训练日志（带时间戳和表情符号）
- 实验目录自动编号，避免覆盖
- 完整的训练状态保存

## 常见问题

### 环境问题
**Q: ImportError: cannot import name 'ValidationInfo' from 'pydantic'**
```bash
# 解决方案1：升级 pydantic
pip install pydantic>=2.0

# 解决方案2：降级 albumentations
pip install albumentations==1.3.1
```

### 训练问题
**Q: CUDA 显存不足**
- 减小 `batch_size`
- 减小 `input_size`
- 使用梯度累积

**Q: 模型收敛缓慢**
- 检查学习率设置
- 尝试不同的学习率调度策略
- 检查数据预处理和增强

**Q: 可视化图像显示异常（马赛克）**
- 已修复：自动进行反归一化处理
- 确保 ImageNet 标准化参数正确

### 配置问题
**Q: 如何修改验证/测试间隔？**
```python
TRAIN_PARAMS = {
    'val_interval': 500,    # 验证间隔
    'test_interval': 1000,  # 测试间隔
}
```

## 快速开始

1. **环境准备**
```bash
pip install torch torchvision numpy opencv-python tqdm albumentations
```

2. **数据准备**
- 按照数据集结构组织数据
- 修改 `cfg/config.py` 中的路径配置

3. **开始训练**
```bash
python tools/train.py
```

4. **查看结果**
- 训练日志：`experiments/exp_001/logs/train.log`
- 模型权重：`experiments/exp_001/models/`
- 可视化结果：`experiments/exp_001/visualizations/`
- 训练曲线：`experiments/exp_001/visualizations/training_curves.png`

## 项目亮点

- 🔧 **开箱即用**: 无需复杂配置，直接运行训练
- 📊 **完整监控**: 训练损失、验证指标、测试性能全程跟踪
- 🎨 **实时可视化**: 自动生成预测结果对比图
- 📈 **训练曲线**: 自动绘制损失和指标曲线，直观分析训练过程
- 💾 **智能保存**: 自动断点续训，实验目录管理
- 📝 **详细日志**: 支持表情符号的美观日志输出
- ⚙️ **灵活配置**: 统一配置文件，支持多种训练策略

## 引用

如果本项目对您的研究有帮助，请引用 BiseNetV2 原论文：

```bibtex
@article{yu2021bisenet,
  title={BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation},
  author={Yu, Changqian and Gao, Changxin and Wang, Jingbo and Yu, Gang and Shen, Chunhua and Sang, Nong},
  journal={International Journal of Computer Vision},
  year={2021}
}
```

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 贡献

欢迎提交 Issue 和 Pull Request！请确保：
- 提供详细的问题描述或改进说明
- 包含必要的测试和文档
- 遵循现有的代码风格

## 版权声明

本仓库仅供学习和研究使用。使用时请遵守相应数据集和第三方库的许可协议。
