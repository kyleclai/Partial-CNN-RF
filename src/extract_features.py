"""
Extract features from all CNN layers and save as .npz files.
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
import tensorflow as tf
from utils.model_builders import get_all_layer_names, create_feature_extractor
from utils.data_loaders import load_split_as_numpy
from utils.seeds import set_global_seed, configure_gpu


def flatten_features(features):
    """
    Flatten 4D tensor (N, H, W, C) to 2D (N, H*W*C).
    Handles already-flat features gracefully.
    """
    if features.ndim == 2:
        return features
    elif features.ndim == 4:
        n_samples = features.shape[0]
        return features.reshape(n_samples, -1)
    else:
        raise ValueError(f"Unexpected feature dimensionality: {features.ndim}D")


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
    
    # Get all extractable layers
    layer_names = get_all_layer_names(model)
    print(f"\nFound {len(layer_names)} layers for extraction:")
    for i, name in enumerate(layer_names):
        print(f"  {i+1}. {name}")
    
    # Load data
    metadata_path = Path(config['data']['data_dir']) / 'metadata.csv'
    img_size = tuple(config['data']['image_size'])
    batch_size = config['train']['batch_size']
    
    # Extract features for each split
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print('='*60)
        
        # Load images
        x_data, y_data = load_split_as_numpy(
            metadata_path, split=split, img_size=img_size,
            limit=config['data'].get('limit_total'),
            random_state=config['run']['seed']
        )
        
        # Extract from each layer
        for layer_name in layer_names:
            print(f"\nExtracting features from: {layer_name}")
            
            # Create extractor
            extractor = create_feature_extractor(model, layer_name)
            
            # Extract features
            features_raw = extractor.predict(x_data, batch_size=batch_size, verbose=0)
            
            # Flatten for RF compatibility
            features_flat = flatten_features(features_raw)
            
            print(f"  Raw shape: {features_raw.shape}")
            print(f"  Flattened shape: {features_flat.shape}")
            
            # Save as compressed NPZ
            output_path = features_dir / f'{split}_{layer_name}_features.npz'
            np.savez_compressed(
                output_path,
                features=features_flat,
                labels=y_data
            )
            
            print(f"  Saved to {output_path}")
    
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