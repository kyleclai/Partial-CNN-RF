"""
Extract features from all CNN layers and save as .npz files.
Uses streaming to avoid RAM exhaustion.
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
import tensorflow as tf
from utils.model_builders import create_feature_extractor, get_strategic_vgg16_layers, get_all_layer_names
from utils.data_loaders import create_tf_dataset
from utils.seeds import set_global_seed, configure_gpu


def flatten_features(features):
    """Flatten 4D tensor (N, H, W, C) to 2D (N, H*W*C)."""
    if features.ndim == 2:
        return features
    elif features.ndim == 4:
        n_samples = features.shape[0]
        return features.reshape(n_samples, -1)
    else:
        raise ValueError(f"Unexpected feature dimensionality: {features.ndim}D")


def extract_features_streaming(extractor, dataset, total_samples, extraction_batch_size=8):
    """
    Extract features using tf.data pipeline (no RAM loading).
    
    Args:
        extractor: Feature extractor model
        dataset: tf.data.Dataset yielding (images, labels)
        total_samples: Total number of samples for progress tracking
        extraction_batch_size: Batch size for extraction
    
    Returns:
        features (ndarray), labels (ndarray)
    """
    feature_batches = []
    label_batches = []
    processed = 0
    
    for batch_images, batch_labels in dataset:
        # Extract features for this batch
        batch_features = extractor.predict(batch_images, verbose=0)
        
        feature_batches.append(batch_features)
        label_batches.append(batch_labels.numpy())
        
        processed += len(batch_labels)
        if len(feature_batches) % 100 == 0:
            print(f"  Processed {processed}/{total_samples} samples", end='\r')
    
    print(f"  Processed {total_samples}/{total_samples} samples")
    
    # Concatenate all batches
    all_features = np.concatenate(feature_batches, axis=0)
    all_labels = np.concatenate(label_batches, axis=0)
    
    return all_features, all_labels


def get_split_size(metadata_path, split):
    """Count samples in a split without loading images."""
    import pandas as pd
    df = pd.read_csv(metadata_path)
    return len(df[df['split'] == split])


def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    set_global_seed(config['run']['seed'])
    configure_gpu(config['run']['device'])
    
    # Paths
    run_name = config['run']['name']
    artifacts_dir = Path(config['run']['artifacts_dir']) / run_name
    models_dir = artifacts_dir / 'models'
    features_dir = artifacts_dir / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trained CNN
    arch = config['train']['arch']
    model_path = models_dir / f'{arch}_baseline.keras'
    
    print(f"Loading trained model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Get strategic layers based on architecture
    if arch == 'vgg16':
        layer_names = get_strategic_vgg16_layers()
        print(f"\nUsing {len(layer_names)} strategic VGG16 conv block endpoints:")
        extraction_batch_size = 8  # Small batches for VGG16
    else:
        layer_names = get_all_layer_names(model)
        print(f"\nFound {len(layer_names)} layers for extraction:")
        extraction_batch_size = 32  # LeNet can use larger batches
    
    for i, name in enumerate(layer_names):
        print(f"  {i+1}. {name}")
    
    print(f"\nUsing extraction batch size: {extraction_batch_size}")
    
    # Paths
    metadata_path = Path(config['data']['data_dir']) / 'metadata.csv'
    img_size = tuple(config['data']['image_size'])
    
    # Extract features for each split
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print('='*60)
        
        # Get split size
        split_size = get_split_size(metadata_path, split)
        print(f"Split size: {split_size} samples")
        
        # Create streaming dataset
        dataset = create_tf_dataset(
            metadata_path, 
            split=split, 
            img_size=img_size,
            batch_size=extraction_batch_size,
            shuffle=False,
            random_state=config['run']['seed']
        )
        
        # Extract from each layer
        for layer_name in layer_names:
            print(f"\nExtracting features from: {layer_name}")
            
            # Create extractor
            extractor = create_feature_extractor(model, layer_name)
            
            # Extract features using streaming
            features_raw, labels = extract_features_streaming(
                extractor, dataset, split_size, extraction_batch_size
            )
            
            # Flatten for RF compatibility
            features_flat = flatten_features(features_raw)
            
            print(f"  Raw shape: {features_raw.shape}")
            print(f"  Flattened shape: {features_flat.shape}")
            
            # Save as compressed NPZ
            output_path = features_dir / f'{split}_{layer_name}_features.npz'
            np.savez_compressed(
                output_path,
                features=features_flat,
                labels=labels
            )
            
            print(f"  Saved to {output_path}")
            
            # Clean up
            del extractor, features_raw, features_flat, labels
            tf.keras.backend.clear_session()
        
        # Recreate dataset for next layer (iterator exhausted)
        del dataset
    
    print(f"\n{'='*60}")
    print("Feature extraction complete!")
    print(f"Features saved to {features_dir}")
    print('='*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract CNN features')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML')
    args = parser.parse_args()
    
    main(args.config)