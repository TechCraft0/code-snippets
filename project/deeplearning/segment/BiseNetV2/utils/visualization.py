import os
import torch
import numpy as np
import cv2
import logging
import random
import psutil
from torchvision.transforms import ToPILImage
from tqdm import tqdm


def save_segmentation_results(model, data_loader, device, epoch, class_colors, save_dir="visualization", max_samples=10):
    """保存分割结果可视化"""
    try:
        # logging.info(f"🎨 [DEBUG] Starting visualization for {epoch}...")
        model.eval()
        vis_dir = os.path.join(save_dir, f"epoch_{epoch}")
        os.makedirs(vis_dir, exist_ok=True)
        # logging.info(f"🎨 [DEBUG] Created visualization directory: {vis_dir}")

        # 计算总样本数并随机采样
        total_samples = len(data_loader.dataset)
        if max_samples < total_samples:
            # 随机选择样本索引
            selected_indices = random.sample(range(total_samples), max_samples)
            selected_indices.sort()  # 排序以便高效遍历
            # logging.info(f"🎨 [DEBUG] Randomly selected {max_samples} samples from {total_samples} total samples")
        else:
            selected_indices = list(range(min(max_samples, total_samples)))
            # logging.info(f"🎨 [DEBUG] Using first {len(selected_indices)} samples")

        total_saved = 0
        current_sample_idx = 0
        selected_set = set(selected_indices)
        
        with torch.no_grad():
            for idx, (images, labels) in enumerate(data_loader):
                try:
                    # logging.info(f"🎨 [DEBUG] Processing batch {idx+1}/{len(data_loader)}")
                    
                    # 限制样本数量以避免内存问题
                    if total_saved >= max_samples:
                        logging.info(f"🎨 [DEBUG] Reached max samples limit ({max_samples}), stopping")
                        break
                    
                    # 移动到GPU并预测
                    images_gpu = images.to(device)
                    # logging.info(f"🎨 [DEBUG] Images moved to device, shape: {images_gpu.shape}")
                    
                    # 获取预测结果
                    outputs = model(images_gpu)
                    if isinstance(outputs, tuple):
                        preds = outputs[0]  # 主头输出
                    else:
                        preds = outputs
                    
                    preds = torch.argmax(preds, dim=1).cpu().numpy()
                    # logging.info(f"🎨 [DEBUG] Predictions computed, shape: {preds.shape}")
                    
                    # 移动到CPU并立即清理GPU变量
                    labels_np = labels.cpu().numpy()
                    images_cpu = images.cpu()
                    
                    # 立即删除GPU tensor
                    del images_gpu, outputs
                    torch.cuda.empty_cache()
                    # logging.info(f"🎨 [DEBUG] GPU tensors cleared")

                    # 处理每个样本
                    batch_size = images_cpu.size(0)
                    for i in range(batch_size):
                        # 检查当前样本是否被选中
                        if current_sample_idx not in selected_set:
                            current_sample_idx += 1
                            continue
                            
                        if total_saved >= max_samples:
                            break
                            
                        try:
                            # logging.info(f"🎨 [DEBUG] Processing selected sample {current_sample_idx} ({total_saved+1}/{max_samples})")
                            
                            # 内存监控
                            if total_saved % 5 == 0:  # 每5个样本记录一次
                                mem = psutil.virtual_memory()
                                gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                                # logging.info(f"🧠 [VIS-MEMORY] Sample {total_saved}: RAM {mem.percent:.1f}%, GPU {gpu_mem:.1f}GB")
                            
                            # 反归一化图像
                            img_tensor = images_cpu[i].clone()
                            img_tensor = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                            img_tensor = torch.clamp(img_tensor, 0, 1)
                            
                            # 转换为 PIL 图像再转 BGR
                            to_pil = ToPILImage()
                            img = to_pil(img_tensor)
                            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                            
                            # 立即删除临时tensor
                            del img_tensor, img
                            # logging.info(f"🎨 [DEBUG] Image converted, shape: {img_bgr.shape}")

                            # 构造可视化 label 和 pred 彩图
                            h, w = labels_np[i].shape
                            label_color = np.zeros((h, w, 3), dtype=np.uint8)
                            pred_color = np.zeros((h, w, 3), dtype=np.uint8)

                            for class_id, color in class_colors.items():
                                if class_id < len(class_colors):  # 安全检查
                                    label_color[labels_np[i] == class_id] = color
                                    pred_color[preds[i] == class_id] = color
                            
                            # logging.info(f"🎨 [DEBUG] Color maps created")

                            # 保存三图拼接
                            vis_img = np.concatenate([
                                img_bgr,
                                label_color,
                                pred_color
                            ], axis=1)
                            
                            save_path = os.path.join(vis_dir, f"sample_{total_saved:04d}.png")
                            # logging.info(f"🎨 [DEBUG] Saving to: {save_path}")
                            
                            success = cv2.imwrite(save_path, vis_img)
                            if success:
                                total_saved += 1
                                # logging.info(f"🎨 [DEBUG] Sample saved successfully ({total_saved}/{max_samples})")
                            else:
                                logging.error(f"🎨 [ERROR] Failed to save image: {save_path}")
                            
                            # 清理numpy数组
                            del img_bgr, label_color, pred_color, vis_img
                            
                            # 强制垃圾回收
                            if total_saved % 10 == 0:
                                import gc
                                gc.collect()
                            
                        except Exception as e:
                            logging.error(f"🎨 [ERROR] Failed to process sample {current_sample_idx}: {str(e)}")
                        finally:
                            current_sample_idx += 1
                                

                    
                    # 清理批次数据
                    del labels_np, images_cpu, preds
                    # logging.info(f"🎨 [DEBUG] Batch {idx+1} completed, total saved: {total_saved}")
                    
                    # 如果已经保存足够的样本，提前退出
                    if total_saved >= max_samples:
                        # logging.info(f"🎨 [DEBUG] Reached target sample count, stopping early")
                        break
                    
                except Exception as e:
                    # logging.error(f"🎨 [ERROR] Failed to process batch {idx}: {str(e)}")
                    # 清理可能的GPU内存
                    torch.cuda.empty_cache()
                    continue
        
        # logging.info(f"🎨 [DEBUG] Visualization completed, total saved: {total_saved}")
        return total_saved
        
    except Exception as e:
        # logging.error(f"🎨 [ERROR] Visualization function failed: {str(e)}")
        import traceback
        logging.error(f"🎨 [ERROR] Traceback: {traceback.format_exc()}")
        return 0
    
    finally:
        # 确保清理资源
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info(f"🎨 [DEBUG] Visualization function cleanup completed")