import os
import sys
import math
import torch
import logging
import argparse
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import psutil
import gc
from memory_profiler import profile
import tracemalloc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cfg.config_stdc import *

# 日志控制配置
DEBUG_CONFIG = {
    'enable_detailed_logs': False,  # 详细日志开关
    'enable_memory_logs': False,  # 内存监控日志开关
    'enable_validation_logs': False,  # 验证过程详细日志
    'enable_visualization_logs': False,  # 可视化过程详细日志
}
from torch import optim
from tqdm import tqdm
from data.data_load import *
from model.bisenetv2_stdc import BisenetV2STDC
from utils.common import create_experiment_dirs
from torch.utils.data import Dataset, DataLoader
from tools.val import validate
from utils.visualization import save_segmentation_results
from utils.plot_curves import plot_training_curves
from model.common import DetailAggregateLoss


def log_memory_usage(stage=""):
    """记录内存使用情况"""
    try:
        # 系统内存
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # GPU内存
        gpu_allocated = 0
        gpu_reserved = 0
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024 ** 3
            gpu_reserved = torch.cuda.memory_reserved() / 1024 ** 3

        logging.info(
            f"🧠 [MEMORY] {stage} | RAM: {mem.percent:.1f}% ({mem.used / 1024 ** 3:.1f}GB/{mem.total / 1024 ** 3:.1f}GB) | "
            f"Swap: {swap.percent:.1f}% ({swap.used / 1024 ** 3:.1f}GB) | "
            f"GPU: {gpu_allocated:.1f}GB allocated, {gpu_reserved:.1f}GB reserved")

        # 内存异常检测
        if mem.percent > 85:
            logging.warning(f"⚠️ [MEMORY] High RAM usage: {mem.percent:.1f}%")
        if swap.percent > 50:
            logging.warning(f"⚠️ [MEMORY] High swap usage: {swap.percent:.1f}%")
        if gpu_allocated > 12:  # RTX 4060 Ti 16GB的80%
            logging.warning(f"⚠️ [MEMORY] High GPU usage: {gpu_allocated:.1f}GB")

    except Exception as e:
        logging.error(f"🧠 [MEMORY] Failed to log memory usage: {str(e)}")


def train(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        loss_function,
        device,
        model_save_path,
        visualization_save_path,
        class_colors,
        max_iter,
        start_iter=0,
        resume_metrics=None,
):
    model.train()
    train_losses = []
    iter_count = start_iter

    # Initialize edge loss function once
    edge_loss_fn = DetailAggregateLoss().to(device)

    # Clear GPU cache
    torch.cuda.empty_cache()

    # 记录训练指标（支持断点续训）
    if resume_metrics is not None:
        metrics_history = resume_metrics
        print(f"📈 Resumed metrics history with {len(metrics_history['train_total_loss'])} training records")
    else:
        metrics_history = {
            'train_total_loss': [],
            'train_seg_loss': [],
            'train_edge_loss': [],
            'train_main_loss': [],
            'train_aux0_loss': [],
            'train_aux1_loss': [],
            'train_aux2_loss': [],
            'train_aux3_loss': [],
            'train_iters': [],
            'learning_rate': [],
            'val_loss': [],
            'val_miou': [],
            'val_pixel_acc': [],
            'val_iters': [],
            'test_loss': [],
            'test_miou': [],
            'test_pixel_acc': [],
            'test_iters': []
        }

    # 启动内存跟踪
    tracemalloc.start()
    log_memory_usage("Training Start")

    # Simple progress tracking without tqdm
    print(f"🚀 Training started: {start_iter}/{max_iter} iterations")

    while iter_count < max_iter:
        for images, labels in train_loader:
            iter_count += 1
            if iter_count >= max_iter:
                break

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            main_head, aux_heads, x1 = model(images)
            bce_loss, dice_loss = edge_loss_fn(x1, labels)
            edge_loss = bce_loss + dice_loss
            losses = [loss_function(main_head, labels)]
            for head in aux_heads:
                losses.append(loss_function(head, labels))
            seg_loss = torch.stack(losses).sum()
            loss = seg_loss + edge_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

            # Calculate metrics for logging
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = sum(train_losses[-50:]) / min(50, len(train_losses))

            # 训练损失日志 - 仅显示关键信息
            if iter_count % 50 == 0:
                progress = iter_count / max_iter
                bar_length = 20
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                percentage = progress * 100

                # 详细的训练损失日志
                loss_details = " | ".join([f"H{i}: {l.item():.4f}" for i, l in enumerate(losses)])
                print(
                    f"📊 [{bar}] {percentage:5.1f}% Iter {iter_count:>6} | Total: {loss.item():.4f} | Seg: {seg_loss.item():.4f} | Edge: {edge_loss.item():.4f} | {loss_details} | LR: {current_lr:.2e}")

                # 记录训练损失分解
                metrics_history['train_total_loss'].append(loss.item())
                metrics_history['train_seg_loss'].append(seg_loss.item())
                metrics_history['train_edge_loss'].append(edge_loss.item())
                metrics_history['train_main_loss'].append(losses[0].item())
                for i, aux_loss in enumerate(losses[1:]):
                    if i < 4:
                        metrics_history[f'train_aux{i}_loss'].append(aux_loss.item())
                metrics_history['train_iters'].append(iter_count)
                metrics_history['learning_rate'].append(current_lr)

                # 内存监控
                if DEBUG_CONFIG['enable_memory_logs'] and iter_count % 200 == 0:
                    log_memory_usage(f"Iter {iter_count}")

            # Validation
            if (
                    iter_count % TRAIN_PARAMS.get("val_interval", 1000) == 0
                    or iter_count == max_iter
            ):
                try:
                    # 清理GPU缓存和内存
                    gc.collect()
                    torch.cuda.empty_cache()
                    if DEBUG_CONFIG['enable_memory_logs']:
                        log_memory_usage(f"Before Validation Iter {iter_count}")

                    val_loss, miou, pixel_acc = validate(
                        model,
                        val_loader,
                        loss_function,
                        device,
                        MODEL_PARAMS["num_classes"],
                        stdc=True
                    )

                    # 简化的验证结果输出
                    print(
                        f"🔍 Validation | Iter {iter_count} | Loss: {val_loss:.4f} | mIoU: {miou:.4f} | PixelAcc: {pixel_acc:.4f}")

                    # 记录验证指标
                    metrics_history['val_loss'].append(val_loss)
                    metrics_history['val_miou'].append(miou)
                    metrics_history['val_pixel_acc'].append(pixel_acc)
                    metrics_history['val_iters'].append(iter_count)

                except Exception as e:
                    error_msg = f"⚠️ Validation failed at iter {iter_count}: {str(e)}"
                    print(error_msg)
                    logging.error(error_msg)
                    import traceback
                    logging.error(f"⚠️ Validation traceback: {traceback.format_exc()}")

                model.train()

            # Test evaluation and visualization
            if (
                    iter_count % TRAIN_PARAMS.get("test_interval", 200) == 0
                    or iter_count == max_iter
            ):
                try:
                    # 清理GPU缓存
                    torch.cuda.empty_cache()
                    if DEBUG_CONFIG['enable_memory_logs']:
                        log_memory_usage(f"Before Test Iter {iter_count}")

                    test_loss, test_miou, test_pixel_acc = validate(
                        model,
                        test_loader,
                        loss_function,
                        device,
                        MODEL_PARAMS["num_classes"],
                        stdc=True,
                    )

                    # 简化的测试结果输出
                    print(
                        f"🏆 Test | Iter {iter_count} | Loss: {test_loss:.4f} | mIoU: {test_miou:.4f} | PixelAcc: {test_pixel_acc:.4f}")

                    # 记录测试指标
                    metrics_history['test_loss'].append(test_loss)
                    metrics_history['test_miou'].append(test_miou)
                    metrics_history['test_pixel_acc'].append(test_pixel_acc)
                    metrics_history['test_iters'].append(iter_count)

                    # 可视化生成
                    if TRAIN_PARAMS.get("enable_visualization", True):
                        torch.cuda.empty_cache()
                        if DEBUG_CONFIG['enable_memory_logs']:
                            log_memory_usage(f"Before Visualization Iter {iter_count}")

                        max_samples = TRAIN_PARAMS.get("max_vis_samples", 5)
                        saved_count = save_segmentation_results(model, test_loader, device, f"iter_{iter_count}",
                                                                class_colors, visualization_save_path,
                                                                max_samples=max_samples)
                        print(f"🎨 Visualizations saved: {saved_count} samples")

                except Exception as e:
                    error_msg = f"⚠️ Test evaluation/visualization failed at iter {iter_count}: {str(e)}"
                    print(error_msg)
                    logging.error(error_msg)
                    import traceback
                    logging.error(f"⚠️ Test traceback: {traceback.format_exc()}")

                model.train()

            # Save checkpoint
            if iter_count % TRAIN_PARAMS.get("checkpoint_interval", 1000) == 0 or iter_count == max_iter:
                try:
                    logging.info(f"💾 [DEBUG] Starting checkpoint save at iter {iter_count}...")
                    checkpoint_data = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'iter_count': iter_count,
                        'metrics_history': metrics_history
                    }

                    checkpoint_path = os.path.join(model_save_path, f'model_iter_{iter_count}.pth')
                    logging.info(f"💾 [DEBUG] Saving checkpoint to: {checkpoint_path}")
                    torch.save(checkpoint_data, checkpoint_path)

                    save_msg = f"💾 Checkpoint saved at iter {iter_count}"
                    print(save_msg)
                    logging.info(save_msg)
                    logging.info(f"💾 [DEBUG] Checkpoint save completed successfully")
                except Exception as e:
                    error_msg = f"⚠️ Checkpoint save failed at iter {iter_count}: {str(e)}"
                    print(error_msg)
                    logging.error(error_msg)
                    import traceback
                    logging.error(f"⚠️ Checkpoint save traceback: {traceback.format_exc()}")

    print(f"✅ Training completed! Final iteration: {iter_count}/{max_iter}")
    logging.info("✅ Training completed!")

    # 最终内存报告
    log_memory_usage("Training Completed")

    # 显示内存增长最大的前10个位置
    try:
        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"🧠 [MEMORY] Peak memory usage: {peak / 1024 ** 2:.1f} MB")

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        logging.info("🧠 [MEMORY] Top 5 memory allocations:")
        for index, stat in enumerate(top_stats[:5], 1):
            logging.info(f"🧠 [MEMORY] #{index}: {stat}")

        tracemalloc.stop()
    except Exception as e:
        logging.error(f"🧠 [MEMORY] Failed to get memory trace: {str(e)}")

    # 绘制训练曲线
    try:
        logging.info("📈 [DEBUG] Starting to plot training curves...")
        print("📈 Plotting training curves...")
        plot_training_curves(metrics_history, visualization_save_path)
        print(f"📈 Training curves saved to {visualization_save_path}/training_curves.png")
        logging.info(f"📈 Training curves saved to {visualization_save_path}/training_curves.png")
        logging.info("📈 [DEBUG] Training curves plotting completed successfully")
    except Exception as e:
        error_msg = f"⚠️ Training curves plotting failed: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        import traceback
        logging.error(f"⚠️ Training curves traceback: {traceback.format_exc()}")


@profile
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='BiseNetV2 Training')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--no-resume', action='store_true', help='Force start training from scratch')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug logs')
    parser.add_argument('--debug-memory', action='store_true', help='Enable memory monitoring logs')
    parser.add_argument('--debug-val', action='store_true', help='Enable validation debug logs')
    parser.add_argument('--debug-vis', action='store_true', help='Enable visualization debug logs')
    args = parser.parse_args()

    # 更新日志配置
    if args.debug:
        DEBUG_CONFIG.update({
            'enable_detailed_logs': True,
            'enable_memory_logs': True,
            'enable_validation_logs': True,
            'enable_visualization_logs': True,
        })
    else:
        if args.debug_memory:
            DEBUG_CONFIG['enable_memory_logs'] = True
        if args.debug_val:
            DEBUG_CONFIG['enable_validation_logs'] = True
        if args.debug_vis:
            DEBUG_CONFIG['enable_visualization_logs'] = True

    # get data path
    train_img_path = PATHS['train_img_path']
    train_label_path = PATHS['train_label_path']
    val_img_path = PATHS['val_img_path']
    val_label_path = PATHS['val_label_path']
    test_img_path = PATHS['test_img_path']
    test_label_path = PATHS['test_label_path']

    # create train result path or find existing one for resume
    enable_resume = not args.no_resume and (args.resume or TRAIN_PARAMS.get("auto_resume", True))

    if enable_resume:
        # 查找最新的实验目录
        root_dir = PATHS["root_dir"]
        if os.path.exists(root_dir):
            exp_dirs = [d for d in os.listdir(root_dir) if
                        d.startswith(PATHS["name"] + "_") and os.path.isdir(os.path.join(root_dir, d))]
            if exp_dirs:
                latest_exp = sorted(exp_dirs, key=lambda x: int(x.split("_")[-1]))[-1]
                exp_path = os.path.join(root_dir, latest_exp)
                model_save_path = os.path.join(exp_path, PATHS["save_dir"])
                logs_save_path = os.path.join(exp_path, PATHS["log_dir"])
                visualization_save_path = os.path.join(exp_path, PATHS["visualization_dir"])
                print(f"🔄 Found existing experiment: {latest_exp}")
            else:
                model_save_path, logs_save_path, visualization_save_path = create_experiment_dirs(
                    PATHS["root_dir"], PATHS["name"], PATHS["save_dir"], PATHS["log_dir"], PATHS["visualization_dir"])
        else:
            model_save_path, logs_save_path, visualization_save_path = create_experiment_dirs(
                PATHS["root_dir"], PATHS["name"], PATHS["save_dir"], PATHS["log_dir"], PATHS["visualization_dir"])
    else:
        model_save_path, logs_save_path, visualization_save_path = create_experiment_dirs(
            PATHS["root_dir"], PATHS["name"], PATHS["save_dir"], PATHS["log_dir"], PATHS["visualization_dir"])

    # setup logging with emoji support
    log_file = os.path.join(logs_save_path, 'train.log')
    logging.basicConfig(
        level=logging.INFO,  # 改为DEBUG级别以获取更多信息
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True  # 强制重新配置logging
    )

    # 添加内存和GPU信息监控
    if torch.cuda.is_available():
        logging.info(f"🔧 [DEBUG] GPU: {torch.cuda.get_device_name()}")
        logging.info(f"🔧 [DEBUG] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    logging.info(
        f"🚀 Training started! 📁 Model save path: {model_save_path} 📊 Logs save path: {logs_save_path} 🎨 Visualization save path: {visualization_save_path}"
    )

    # get class names and colors
    classes_name_color_pairs = CLASS_NAME
    # Convert to BGR format for visualization
    class_colors = {cls_id: tuple(cls_info['color'][::-1]) for cls_id, cls_info in classes_name_color_pairs.items()}

    # setting max iteration
    max_iter = TRAIN_PARAMS["total_iters"]

    # 后面可以使用这个字典，比如绘制的时候：
    for cls_id, cls_info in classes_name_color_pairs.items():
        logging.info(f"🌈 Class {cls_id}: {cls_info['name']}, color: {cls_info['color']}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_set = LoadImageAndLabels(
        train_img_path,
        train_label_path,
        TRAIN_PARAMS["input_size"],
        TRAIN_PARAMS["mode"],
        TRAIN_PARAMS["crop_size"])
    train_loader = DataLoader(
        train_set,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=True,
        num_workers=TRAIN_PARAMS["num_workers"],
        pin_memory=TRAIN_PARAMS["pin_memory"],
        drop_last=TRAIN_PARAMS["drop_last"])

    val_set = LoadImageAndLabels(
        val_img_path,
        val_label_path,
        TRAIN_PARAMS["input_size"],
        mode="val")
    val_loader = DataLoader(
        val_set,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=TRAIN_PARAMS["num_workers"],
        pin_memory=TRAIN_PARAMS["pin_memory"],
        drop_last=False)

    test_set = LoadImageAndLabels(
        test_img_path,
        test_label_path,
        TRAIN_PARAMS["input_size"],
        mode="val")
    test_loader = DataLoader(
        test_set,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=TRAIN_PARAMS["num_workers"],
        pin_memory=TRAIN_PARAMS["pin_memory"],
        drop_last=False)

    # model config
    model = BisenetV2STDC(
        in_channels=MODEL_PARAMS["in_channels"],
        out_channels=MODEL_PARAMS["out_channels"],
        n_classes=MODEL_PARAMS["num_classes"]).to(device)

    # loss function
    if LOSS_PARAMS["type"] == "ce":
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.CrossEntropyLoss()

    if OPTIMIZER_PARAMS == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=TRAIN_PARAMS["lr"], momentum=0.9, weight_decay=5e-4)
    elif OPTIMIZER_PARAMS == "adam":
        optimizer = optim.Adam(model.parameters(), lr=TRAIN_PARAMS["lr"], weight_decay=5e-4)
    elif OPTIMIZER_PARAMS == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=TRAIN_PARAMS["lr"], weight_decay=5e-4)
    else:
        raise ValueError(f"🚨 Unsupported optimizer type: {LOSS_PARAMS['type']}")

    # learning rate scheduler
    lr_scheduler_type = LR_SCHEDULER_PARAMS["type"]
    if lr_scheduler_type == "linear":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda it: 1.0 - it / LR_SCHEDULER_PARAMS["total_iters"]
        )
    elif lr_scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=LR_SCHEDULER_PARAMS["step_size"],
            gamma=LR_SCHEDULER_PARAMS["gamma"]
        )
    elif lr_scheduler_type == "poly":
        def poly_decay(it):
            return (1 - it / LR_SCHEDULER_PARAMS["total_iters"]) ** LR_SCHEDULER_PARAMS["power"]

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_decay)
    elif lr_scheduler_type == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=LR_SCHEDULER_PARAMS["total_iters"]
        )
    elif lr_scheduler_type == "warmcos":
        def warm_cosine(it):
            warmup_iters = LR_SCHEDULER_PARAMS["warmup_iters"]
            total_iters = LR_SCHEDULER_PARAMS["total_iters"]
            if it < warmup_iters:
                return it / warmup_iters
            else:
                return 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (total_iters - warmup_iters)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_cosine)
    else:
        raise ValueError(f"🚨 Unsupported LR scheduler type: {lr_scheduler_type}")

    # checkpoint resume
    iter_count = 0
    resume_metrics = None

    if enable_resume and os.path.exists(model_save_path) and os.listdir(model_save_path):
        checkpoint_files = [f for f in os.listdir(model_save_path) if
                            f.startswith("model_iter_") and f.endswith(".pth")]
        if checkpoint_files:
            latest_ckpt = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            ckpt_path = os.path.join(model_save_path, latest_ckpt)
            try:
                checkpoint = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                iter_count = checkpoint['iter_count']
                # 恢复指标历史（如果存在）
                if 'metrics_history' in checkpoint:
                    resume_metrics = checkpoint['metrics_history']
                logging.info(f"🔄 Resumed from checkpoint: {latest_ckpt}, iter: {iter_count}")
                print(f"🔄 Resumed training from iteration {iter_count}")
                # Clear GPU memory after loading checkpoint
                # del checkpoint
                # torch.cuda.empty_cache()
                # torch.cuda.synchronize()
            except Exception as e:
                logging.error(f"⚠️ Failed to load checkpoint {latest_ckpt}: {str(e)}")
                logging.info("🆕 Starting training from scratch")
                iter_count = 0
        else:
            logging.info("🆕 No checkpoints found, starting training from scratch")
    else:
        if args.no_resume:
            logging.info("🆕 Force starting training from scratch (--no-resume)")
        elif not enable_resume:
            logging.info("🆕 Auto-resume disabled, starting training from scratch")
        else:
            logging.info("🆕 Starting training from scratch")

    # Start training
    try:
        logging.info("🚀 [DEBUG] Starting training function...")
        train(
            model,
            train_loader,
            val_loader,
            test_loader,
            optimizer,
            scheduler,
            loss_function,
            device,
            model_save_path,
            visualization_save_path,
            class_colors,
            max_iter,
            iter_count,
            resume_metrics,
        )
        logging.info("✅ [DEBUG] Training function completed successfully")
    except Exception as e:
        error_msg = f"💥 Training failed: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        import traceback
        logging.error(f"💥 Training traceback: {traceback.format_exc()}")
        raise


if __name__ == '__main__':
    main()