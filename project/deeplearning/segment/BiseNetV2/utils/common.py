import os
import sys


def create_experiment_dirs(root_dir, train_name, save_dir, log_dir, visualization_dir):
    """Create experiment directory structure with unique numbering"""
    # Find next available experiment number
    exp_num = 1
    while True:
        train_dir = os.path.join(root_dir, f"{train_name}_{exp_num:03d}")
        if not os.path.exists(train_dir):
            break
        exp_num += 1
    
    os.makedirs(train_dir, exist_ok=True)
    
    model_save_path = os.path.join(train_dir, save_dir)
    logs_save_path = os.path.join(train_dir, log_dir)
    visualization_save_path = os.path.join(train_dir, visualization_dir)
    
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(logs_save_path, exist_ok=True)
    os.makedirs(visualization_save_path, exist_ok=True)
    
    return model_save_path, logs_save_path, visualization_save_path

