"""
Reproducibility utilities for setting random seeds.
"""

import os
import random
import numpy as np
import tensorflow as tf


def set_global_seed(seed=42):
    """
    Set random seeds for Python, NumPy, and TensorFlow.
    
    Args:
        seed: Integer seed value
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # TensorFlow deterministic operations (slower but reproducible)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    print(f"Global seed set to {seed}")


def configure_gpu(device='auto', gpu_memory_growth=True):
    """
    Configure GPU/CPU execution.
    
    Args:
        device: 'auto', 'cpu', or 'gpu'
        gpu_memory_growth: If True, allocate GPU memory as needed
    """
    physical_gpus = tf.config.list_physical_devices('GPU')
    
    if device == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
        print("Forcing CPU-only execution")
        return
    
    if not physical_gpus:
        print("No GPUs detected, using CPU")
        return
    
    if gpu_memory_growth:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(physical_gpus)} GPU(s)")
    
    print(f"GPU configuration complete: {len(physical_gpus)} GPU(s) available")