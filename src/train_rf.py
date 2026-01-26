"""
Train Random Forest on extracted CNN features OR raw pixels (baseline).
"""

import argparse
import yaml
import json
import time
import joblib
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.data_loaders import load_split_as_numpy
from utils.seeds import set_global_seed


def train_baseline_rf(config):
    """Train RF on raw pixels + PCA (baseline comparison)."""
    print("\n" + "="*60)
    print("Training BASELINE RF (raw pixels + PCA)")
    print("="*60)
    
    # Load raw images
    metadata_path = Path(config['data']['data_dir']) / 'metadata.csv'
    img_size = tuple(config['data']['image_size'])
    
    print("Loading training data...")
    x_train, y_train = load_split_as_numpy(
        metadata_path, split='train', img_size=img_size,
        limit=config['data'].get('limit_total'),
        random_state=config['run']['seed']
    )
    
    print("Loading validation data...")
    x_val, y_val = load_split_as_numpy(
        metadata_path, split='val', img_size=img_size,
        random_state=config['run']['seed']
    )
    
    # Flatten images
    n_train = x_train.shape[0]
    n_val = x_val.shape[0]
    x_train_flat = x_train.reshape(n_train, -1)
    x_val_flat = x_val.reshape(n_val, -1)
    
    print(f"Flattened shape: {x_train_flat.shape}")
    
    # Apply PCA
    n_components = config['baseline_rf']['pca_components']
    print(f"Applying PCA (n_components={n_components})...")
    
    pca = PCA(n_components=n_components, random_state=config['run']['seed'])
    x_train_pca = pca.fit_transform(x_train_flat)
    x_val_pca = pca.transform(x_val_flat)
    
    print(f"PCA shape: {x_train_pca.shape}")
    
    # Train RF
    print("Training Random Forest...")
    start_time = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=config['rf']['n_estimators'],
        max_depth=config['rf']['max_depth'],
        random_state=config['rf']['random_state'],
        n_jobs=-1,
        verbose=1
    )
    rf.fit(x_train_pca, y_train)
    
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = rf.predict(x_val_pca)
    
    metrics = {
        'model_type': 'baseline_rf_pca',
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1_score': f1_score(y_val, y_pred, zero_division=0),
        'train_time_seconds': train_time
    }
    
    print(f"\nBaseline RF Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Train time: {train_time:.2f}s")
    
    # Save model and PCA
    run_name = config['run']['name']
    artifacts_dir = Path(config['run']['artifacts_dir']) / run_name
    models_dir = artifacts_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(rf, models_dir / 'baseline_rf.pkl')
    joblib.dump(pca, models_dir / 'baseline_pca.pkl')
    
    # Save metrics
    with open(artifacts_dir / 'baseline_rf_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def train_hybrid_rf(config):
    """Train RF models on all extracted CNN features."""
    print("\n" + "="*60)
    print("Training HYBRID RF (CNN features)")
    print("="*60)
    
    # Paths
    run_name = config['run']['name']
    artifacts_dir = Path(config['run']['artifacts_dir']) / run_name
    features_dir = artifacts_dir / 'features'
    models_dir = artifacts_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all feature files
    feature_files = sorted(features_dir.glob('train_*_features.npz'))
    layer_names = [f.stem.replace('train_', '').replace('_features', '') 
                   for f in feature_files]
    
    print(f"Found {len(layer_names)} layers with extracted features")
    
    all_metrics = []
    
    for layer_name in layer_names:
        print(f"\n{'='*60}")
        print(f"Training RF on layer: {layer_name}")
        print('='*60)
        
        # Load train features
        train_data = np.load(features_dir / f'train_{layer_name}_features.npz')
        x_train = train_data['features']
        y_train = train_data['labels']
        
        # Load val features
        val_data = np.load(features_dir / f'val_{layer_name}_features.npz')
        x_val = val_data['features']
        y_val = val_data['labels']
        
        print(f"Train features shape: {x_train.shape}")
        print(f"Val features shape: {x_val.shape}")
        
        # Train RF
        print("Training Random Forest...")
        start_time = time.time()
        
        rf = RandomForestClassifier(
            n_estimators=config['rf']['n_estimators'],
            max_depth=config['rf']['max_depth'],
            random_state=config['rf']['random_state'],
            n_jobs=-1,
            verbose=0
        )
        rf.fit(x_train, y_train)
        
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = rf.predict(x_val)
        
        metrics = {
            'model_type': 'hybrid_rf',
            'layer_name': layer_name,
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1_score': f1_score(y_val, y_pred, zero_division=0),
            'train_time_seconds': train_time,
            'n_features': x_train.shape[1]
        }
        
        all_metrics.append(metrics)
        
        print(f"\nResults for {layer_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Train time: {train_time:.2f}s")
        
        # Save model
        joblib.dump(rf, models_dir / f'hybrid_rf_{layer_name}.pkl')
    
    # Save all metrics
    with open(artifacts_dir / 'hybrid_rf_all_layers_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Hybrid RF training complete!")
    print(f"Trained {len(all_metrics)} RF models")
    print('='*60)
    
    return all_metrics


def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_global_seed(config['run']['seed'])
    
    # Train baseline RF if enabled
    if config.get('baseline_rf', {}).get('enabled', False):
        baseline_metrics = train_baseline_rf(config)
    
    # Train hybrid RF models
    hybrid_metrics = train_hybrid_rf(config)
    
    print("\nAll RF training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Random Forest models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML')
    args = parser.parse_args()
    
    main(args.config)