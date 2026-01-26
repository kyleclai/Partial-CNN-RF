"""
Train CNN baseline (VGG16 or LeNet) and save to artifacts.
"""

import argparse
import yaml
import json
import time
from pathlib import Path
import tensorflow as tf
from utils.model_builders import build_vgg16, build_lenet
from utils.data_loaders import load_split_as_numpy
from utils.seeds import set_global_seed, configure_gpu


def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    set_global_seed(config['run']['seed'])
    configure_gpu(config['run']['device'])
    
    # Create artifacts directory
    run_name = config['run']['name']
    artifacts_dir = Path(config['run']['artifacts_dir']) / run_name
    models_dir = artifacts_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
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
    
    # Build model
    input_shape = (*img_size, config['data']['channels'])
    arch = config['train']['arch']
    
    if arch == 'vgg16':
        model = build_vgg16(input_shape=input_shape, num_classes=1)
    elif arch == 'lenet':
        model = build_lenet(input_shape=input_shape, num_classes=1)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config['train']['learning_rate']
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nModel architecture: {arch}")
    model.summary()
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=config['train']['epochs'],
        batch_size=config['train']['batch_size'],
        verbose=1
    )
    
    train_time = time.time() - start_time
    
    # Save model
    model_path = models_dir / f'{arch}_baseline.keras'
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save training history
    history_dict = {
        'train_loss': [float(x) for x in history.history['loss']],
        'train_accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'train_time_seconds': train_time
    }
    
    history_path = artifacts_dir / f'{arch}_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved to {history_path}")
    
    # Final metrics
    print(f"\nTraining completed in {train_time:.2f}s")
    print(f"Final train accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN baseline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML')
    args = parser.parse_args()
    
    main(args.config)