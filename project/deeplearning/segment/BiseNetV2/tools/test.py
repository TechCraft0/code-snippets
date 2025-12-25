import os
import sys
import torch
import logging
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cfg.config import *
from data.data_load import LoadImageAndLabels
from model.bisenetv2 import BisenetV2
from utils.common import create_experiment_dirs
from torch.utils.data import DataLoader
from tools.val import validate
from utils.visualization import save_segmentation_results
from utils.evaluation_metrics import compute_metrics


def test():
    """Test function for evaluating trained model on test dataset"""
    
    # Create test result directories
    test_save_path, test_logs_path, test_vis_path = create_experiment_dirs(
        TEST_PARAMS["root_dir"],
        TEST_PARAMS["name"],
        "results",
        TEST_PARAMS["log_dir"],
        TEST_PARAMS["visualization_dir"],
    )
    
    # Setup logging
    log_file = os.path.join(test_logs_path, 'test.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"🧪 Testing started! 📁 Results path: {test_save_path} 📊 Logs path: {test_logs_path} 🎨 Visualization path: {test_vis_path}")
    
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"🖥️ Using device: {device}")
    
    # Load test dataset
    test_img_path = PATHS['test_img_path']
    test_label_path = PATHS['test_label_path']
    
    test_set = LoadImageAndLabels(
        test_img_path,
        test_label_path,
        TEST_PARAMS["input_size"],
        mode="val"
    )
    test_loader = DataLoader(
        test_set,
        batch_size=TEST_PARAMS["batch_size"],
        shuffle=False,
        num_workers=TEST_PARAMS["num_workers"],
        pin_memory=TEST_PARAMS["pin_memory"],
        drop_last=False
    )
    
    logging.info(f"📊 Test dataset loaded: {len(test_set)} samples")
    
    # Load model
    model = BisenetV2(
        in_channels=MODEL_PARAMS["in_channels"],
        out_channels=MODEL_PARAMS["out_channels"],
        n_classes=MODEL_PARAMS["num_classes"]
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = TEST_PARAMS["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"✅ Model loaded from: {checkpoint_path}")
    
    # Loss function
    loss_function = nn.CrossEntropyLoss()
    
    # Get class colors for visualization
    class_colors = {cls_id: tuple(cls_info['color'][::-1]) for cls_id, cls_info in CLASS_NAME.items()}
    
    # Run detailed evaluation
    logging.info("🔍 Starting comprehensive test evaluation...")
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="🧪 Testing", leave=False):
            images, labels = images.to(device), labels.to(device)
            main_head, aux_heads = model(images)
            
            main_loss = loss_function(main_head, labels)
            aux_losses = [loss_function(head, labels) for head in aux_heads]
            test_loss += (main_loss + 0.4 * sum(aux_losses)).item()
            
            all_preds.append(main_head.detach().cpu())
            all_labels.append(labels.detach().cpu())
    
    test_loss /= len(test_loader)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute comprehensive metrics
    metrics = compute_metrics(all_preds, all_labels, MODEL_PARAMS["num_classes"])
    
    # Log detailed results
    logging.info("=" * 70)
    logging.info("🏆 COMPREHENSIVE TEST RESULTS")
    logging.info("=" * 70)
    logging.info(f"📊 Test Loss: {test_loss:.4f}")
    logging.info(f"🎯 Mean IoU (mIoU): {metrics['miou']:.4f}")
    logging.info(f"✅ Pixel Accuracy: {metrics['pixel_acc']:.4f}")
    logging.info(f"🎨 Mean Class Accuracy: {metrics['mean_class_acc']:.4f}")
    logging.info(f"⚖️ Frequency Weighted IoU: {metrics['fwiou']:.4f}")
    logging.info(f"🎲 Mean Dice Score: {metrics['mean_dice']:.4f}")
    logging.info("-" * 70)
    
    # Per-class results
    logging.info("📈 Per-Class Results:")
    for cls_id, cls_info in CLASS_NAME.items():
        logging.info(f"  {cls_info['name']:>12}: IoU={metrics['per_class_iou'][cls_id]:.4f} | "
                    f"Acc={metrics['per_class_acc'][cls_id]:.4f} | "
                    f"Dice={metrics['per_class_dice'][cls_id]:.4f}")
    logging.info("=" * 70)
    
    # Generate visualizations
    logging.info("🎨 Generating test visualizations...")
    save_segmentation_results(
        model, test_loader, device, "test_results", class_colors, test_vis_path
    )
    logging.info(f"🎨 Visualizations saved to: {test_vis_path}")
    
    # Save comprehensive results to file
    results_file = os.path.join(test_save_path, "test_results.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Comprehensive Test Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test Samples: {len(test_set)}\n")
        f.write("\n")
        f.write("Overall Metrics:\n")
        f.write(f"  Test Loss: {test_loss:.4f}\n")
        f.write(f"  Mean IoU (mIoU): {metrics['miou']:.4f}\n")
        f.write(f"  Pixel Accuracy: {metrics['pixel_acc']:.4f}\n")
        f.write(f"  Mean Class Accuracy: {metrics['mean_class_acc']:.4f}\n")
        f.write(f"  Frequency Weighted IoU: {metrics['fwiou']:.4f}\n")
        f.write(f"  Mean Dice Score: {metrics['mean_dice']:.4f}\n")
        f.write("\n")
        f.write("Per-Class Results:\n")
        for cls_id, cls_info in CLASS_NAME.items():
            f.write(f"  {cls_info['name']:>12}: IoU={metrics['per_class_iou'][cls_id]:.4f} | "
                   f"Acc={metrics['per_class_acc'][cls_id]:.4f} | "
                   f"Dice={metrics['per_class_dice'][cls_id]:.4f}\n")
        f.write("=" * 50 + "\n")
    
    logging.info(f"📄 Results saved to: {results_file}")
    logging.info("✅ Testing completed!")


if __name__ == '__main__':
    test()