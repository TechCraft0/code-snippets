import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def compute_metrics(preds, labels, num_classes):
    """Compute comprehensive metrics for semantic segmentation"""
    preds = torch.argmax(preds, dim=1).cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    
    # Basic metrics
    intersection = np.diag(cm)
    union = cm.sum(1) + cm.sum(0) - intersection
    
    # Per-class IoU
    iou = intersection / np.maximum(union, 1)
    
    # Mean IoU
    miou = np.mean(iou)
    
    # Pixel Accuracy
    pixel_acc = np.sum(intersection) / np.sum(cm)
    
    # Per-class Accuracy
    class_acc = intersection / np.maximum(cm.sum(1), 1)
    mean_class_acc = np.mean(class_acc)
    
    # Frequency Weighted IoU
    freq = cm.sum(1) / np.sum(cm)
    fwiou = np.sum(freq * iou)
    
    # Dice Score (F1 Score)
    dice = 2 * intersection / np.maximum(cm.sum(1) + cm.sum(0), 1)
    mean_dice = np.mean(dice)
    
    return {
        'miou': miou,
        'pixel_acc': pixel_acc,
        'mean_class_acc': mean_class_acc,
        'fwiou': fwiou,
        'mean_dice': mean_dice,
        'per_class_iou': iou,
        'per_class_acc': class_acc,
        'per_class_dice': dice
    }


def compute_metrics_simple(preds, labels, num_classes):
    """Simple version for backward compatibility"""
    metrics = compute_metrics(preds, labels, num_classes)
    return metrics['miou'], metrics['pixel_acc']


