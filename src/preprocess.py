"""
Data preprocessing: create metadata CSV with 70/20/10 split.
"""

import argparse
import yaml
from pathlib import Path
from utils.data_loaders import create_metadata_csv
from utils.seeds import set_global_seed


def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_global_seed(config['run']['seed'])
    
    # Paths
    data_dir = Path(config['data']['data_dir']) / 'dogs-vs-cats' / 'train'
    metadata_path = Path(config['data']['data_dir']) / 'metadata.csv'
    
    # Create metadata with stratified split
    print(f"Scanning images in {data_dir}")
    df = create_metadata_csv(
        data_dir=data_dir,
        output_path=metadata_path,
        random_state=config['run']['seed']
    )
    
    # Print split statistics
    print("\nSplit distribution:")
    print(df.groupby(['split', 'class']).size().unstack(fill_value=0))
    
    print(f"\nMetadata saved to {metadata_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Cats vs Dogs dataset')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML')
    args = parser.parse_args()
    
    main(args.config)