import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import gc

from utils.evaluation_metrics import compute_metrics_simple
from model.common import DetailAggregateLoss


def validate(model, val_loader, loss_function, device, num_classes, stdc: bool = False):
    """Validation function for semantic segmentation"""
    model.eval()
    val_loss = 0
    total_correct = 0
    total_pixels = 0
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    edge_loss_fn = DetailAggregateLoss().to(device)

    # 静默验证，不输出过程日志

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            try:
                # 静默处理，不输出过程日志
                images, labels = images.to(device), labels.to(device)
                if stdc:
                    main_head, aux_heads, edge_map = model(images)

                    main_loss = loss_function(main_head, labels)
                    aux_losses = [loss_function(head, labels) for head in aux_heads]
                    batch_loss = (main_loss + 0.4 * sum(aux_losses)).item()
                    bce_loss, dice_loss = edge_loss_fn(edge_map, labels)
                    edge_loss = (bce_loss + dice_loss).item()
                    val_loss += batch_loss + edge_loss

                    # 计算指标而不保存数据
                    preds = torch.argmax(main_head, dim=1)

                    # 像素准确率
                    correct = (preds == labels).sum().item()
                    pixels = labels.numel()
                    total_correct += correct
                    total_pixels += pixels

                    # 类别 IoU - 移到CPU计算避免GPU内存累积
                    preds_cpu = preds.cpu()
                    labels_cpu = labels.cpu()
                    for c in range(num_classes):
                        pred_c = (preds_cpu == c)
                        label_c = (labels_cpu == c)
                        intersection = (pred_c & label_c).sum().item()
                        union = (pred_c | label_c).sum().item()
                        if union > 0:
                            class_correct[c] += intersection
                            class_total[c] += union

                    # 清理临时变量
                    del preds_cpu, labels_cpu, pred_c, label_c

                    # 立即清理内存
                    del images, labels, main_head, aux_heads, preds, main_loss, aux_losses
                    if batch_idx % 5 == 0:  # 每5个batch清理一次，避免频繁清理影响性能
                        torch.cuda.empty_cache()
                        gc.collect()

                else:
                    main_head, aux_heads = model(images)

                    main_loss = loss_function(main_head, labels)
                    aux_losses = [loss_function(head, labels) for head in aux_heads]
                    batch_loss = (main_loss + 0.4 * sum(aux_losses)).item()
                    val_loss += batch_loss

                    # 计算指标而不保存数据
                    preds = torch.argmax(main_head, dim=1)

                    # 像素准确率
                    correct = (preds == labels).sum().item()
                    pixels = labels.numel()
                    total_correct += correct
                    total_pixels += pixels

                    # 类别 IoU - 移到CPU计算避免GPU内存累积
                    preds_cpu = preds.cpu()
                    labels_cpu = labels.cpu()
                    for c in range(num_classes):
                        pred_c = (preds_cpu == c)
                        label_c = (labels_cpu == c)
                        intersection = (pred_c & label_c).sum().item()
                        union = (pred_c | label_c).sum().item()
                        if union > 0:
                            class_correct[c] += intersection
                            class_total[c] += union

                    # 清理临时变量
                    del preds_cpu, labels_cpu, pred_c, label_c

                    # 立即清理内存
                    del images, labels, main_head, aux_heads, preds, main_loss, aux_losses
                    if batch_idx % 5 == 0:  # 每5个batch清理一次，避免频繁清理影响性能
                        torch.cuda.empty_cache()
                        gc.collect()

            # 静默处理
            except Exception as e:
                logging.error(f"🔍 [ERROR] Validation batch {batch_idx} failed: {str(e)}")
                continue

    # 计算最终指标
    val_loss /= len(val_loader)
    pixel_acc = total_correct / total_pixels if total_pixels > 0 else 0

    # 计算 mIoU
    ious = []
    for c in range(num_classes):
        if class_total[c] > 0:
            iou = class_correct[c] / class_total[c]
            ious.append(iou.item())
    miou = sum(ious) / len(ious) if ious else 0

    # 最终清理
    gc.collect()
    torch.cuda.empty_cache()

    # 验证完成，结果由调用方输出

    return val_loss, miou, pixel_acc
