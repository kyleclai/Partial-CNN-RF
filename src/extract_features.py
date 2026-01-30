"""
Extract features from all CNN layers and save as .npz files.
Uses streaming + global pooling to handle all layers efficiently.
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
import tensorflow as tf
from utils.model_builders import get_all_layer_names, get_all_vgg16_conv_layers#, get_strategic_vgg16_layers
from utils.pooling import create_feature_extractor_with_pooling
from utils.data_loaders import create_tf_dataset
from utils.seeds import set_global_seed, configure_gpu


def extract_features_streaming(extractor, dataset, total_samples):
    """
    Extract features using tf.data pipeline (no RAM loading).
    Features are already pooled, so no flattening needed.
    
    Args:
        extractor: Feature extractor model (with pooling)
        dataset: tf.data.Dataset yielding (images, labels)
        total_samples: Total number of samples for progress tracking
    
    Returns:
        features (ndarray), labels (ndarray)
    """
    feature_batches = []
    label_batches = []
    processed = 0
    
    for batch_images, batch_labels in dataset:
        # Extract features for this batch (already pooled)
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
    
    # Get layer names based on architecture
    if arch == 'vgg16':
        layer_names = get_all_vgg16_conv_layers()
        print(f"\nExtracting from ALL {len(layer_names)} VGG16 conv layers (with GAP):")
        extraction_batch_size = 32  # Can use larger batches with GAP
        use_pooling = True
    else:
        layer_names = get_all_layer_names(model)
        print(f"\nFound {len(layer_names)} layers for extraction:")
        extraction_batch_size = 32
        use_pooling = config.get('features', {}).get('use_global_pooling', False)
    
    for i, name in enumerate(layer_names):
        print(f"  {i+1}. {name}")
    
    if use_pooling:
        print("\n✓ Using Global Average Pooling to reduce feature dimensions")
    
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
            
            # Create extractor with pooling
            if use_pooling and arch == 'vgg16':
                extractor = create_feature_extractor_with_pooling(
                    model, layer_name, pooling_type='avg'
                )
            else:
                from utils.model_builders import create_feature_extractor
                extractor = create_feature_extractor(model, layer_name)
            
            # Extract features using streaming
            features, labels = extract_features_streaming(
                extractor, dataset, split_size
            )
            
            print(f"  Feature shape: {features.shape}")
            
            # Save as compressed NPZ
            output_path = features_dir / f'{split}_{layer_name}_features.npz'
            np.savez_compressed(
                output_path,
                features=features,
                labels=labels
            )
            
            print(f"  Saved to {output_path}")
            
            # Clean up
            del extractor, features, labels
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