"""
Efficient data loading utilities.
Handles image loading, preprocessing, and train/val/test splitting.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_metadata_csv(data_dir, output_path, random_state=42):
    """
    Scan image directory and create metadata CSV with train/val/test splits.
    
    Args:
        data_dir: Path to directory containing cat.*.jpg and dog.*.jpg
        output_path: Where to save metadata CSV
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: [path, class, label_numeric, split]
    """
    image_paths = []
    labels = []
    
    # Scan directory for images
    for img_file in Path(data_dir).glob('*.jpg'):
        img_name = img_file.name
        
        # Extract class from filename (cat.123.jpg -> cat)
        if img_name.startswith('cat'):
            label = 'cat'
        elif img_name.startswith('dog'):
            label = 'dog'
        else:
            continue
        
        image_paths.append(str(img_file))
        labels.append(label)
    
    # Create DataFrame
    df = pd.DataFrame({
        'path': image_paths,
        'class': labels
    })
    
    # Create numeric labels
    label_map = {'cat': 0, 'dog': 1}
    df['label_numeric'] = df['class'].map(label_map)
    
    # Stratified 70/20/10 split
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['label_numeric'], 
        random_state=random_state
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.333, stratify=temp_df['label_numeric'],
        random_state=random_state
    )
    
    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Combine and save
    df_final = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_final.to_csv(output_path, index=False)
    
    print(f"Metadata created: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    return df_final


def load_split_as_numpy(metadata_csv, split='train', img_size=(128, 128), 
                        limit=None, random_state=42):
    """
    Load a specific split (train/val/test) as NumPy arrays.
    
    Args:
        metadata_csv: Path to metadata CSV
        split: 'train', 'val', or 'test'
        img_size: Target (height, width) for resizing
        limit: Maximum number of samples to load (for debugging)
        random_state: Random seed for sampling
    
    Returns:
        (x, y) as NumPy arrays: x.shape = (N, H, W, C), y.shape = (N,)
    """
    df = pd.read_csv(metadata_csv)
    df_split = df[df['split'] == split].copy()
    
    if limit is not None:
        df_split = df_split.sample(n=min(limit, len(df_split)), 
                                   random_state=random_state)
    
    x_list, y_list = [], []
    
    for _, row in df_split.iterrows():
        try:
            img = cv2.imread(row['path'])
            if img is None:
                print(f"Warning: Could not read {row['path']}")
                continue
            
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            x_list.append(img)
            y_list.append(row['label_numeric'])
        except Exception as e:
            print(f"Error loading {row['path']}: {e}")
            continue
    
    x = np.array(x_list)
    y = np.array(y_list)
    
    print(f"Loaded {split}: {x.shape[0]} samples, shape={x.shape}")
    return x, y


def create_tf_dataset(metadata_csv, split='train', img_size=(128, 128), 
                     batch_size=32, shuffle=True, random_state=42):
    """
    Create a TensorFlow Dataset for memory-efficient training.
    
    Args:
        metadata_csv: Path to metadata CSV
        split: 'train', 'val', or 'test'
        img_size: Target (height, width)
        batch_size: Batch size
        shuffle: Whether to shuffle
        random_state: Random seed
    
    Returns:
        tf.data.Dataset
    """
    import tensorflow as tf
    
    df = pd.read_csv(metadata_csv)
    df_split = df[df['split'] == split]
    
    paths = df_split['path'].values
    labels = df_split['label_numeric'].values
    
    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=random_state)
    
    dataset = dataset.map(load_and_preprocess, 
                         num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset