"""
prepare_data.py — Robust metadata.csv generator for IMP exploration.

Handles any dataset size, including the 6-image local dev subset.
Falls back to simple sequential splitting when the dataset is too small
for sklearn's stratified split (< 6 samples per class).

Usage:
    python explorations/lottery_ticket/prepare_data.py
    python explorations/lottery_ticket/prepare_data.py --data-dir data/dogs-vs-cats/train
    python explorations/lottery_ticket/prepare_data.py --config explorations/lottery_ticket/configs/imp_lenet.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_exploration_root = Path(__file__).resolve().parent
_src_path = _exploration_root.parent.parent / 'src'
sys.path.insert(0, str(_src_path))


def create_metadata_csv_robust(
    data_dir: str | Path,
    output_path: str | Path,
    train_split: float = 0.70,
    val_split: float = 0.20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Scan image directory and create metadata CSV with train/val/test splits.

    Works for any dataset size. For datasets too small for stratified splitting,
    falls back to a simple sequential per-class assignment.

    Args:
        data_dir:     Directory containing cat.*.jpg / dog.*.jpg images.
        output_path:  Where to write metadata.csv.
        train_split:  Fraction for training (default 0.70).
        val_split:    Fraction for validation (default 0.20).
        random_state: Random seed.

    Returns:
        DataFrame with columns [path, class, label_numeric, split].
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths, labels = [], []
    for img_file in sorted(data_dir.glob('*.jpg')):
        name = img_file.name
        if name.startswith('cat'):
            labels.append('cat')
        elif name.startswith('dog'):
            labels.append('dog')
        else:
            continue
        image_paths.append(str(img_file))

    if not image_paths:
        raise FileNotFoundError(
            f"No cat/dog .jpg images found in {data_dir}. "
            "Expected filenames like cat.0.jpg and dog.0.jpg."
        )

    df = pd.DataFrame({'path': image_paths, 'class': labels})
    df['label_numeric'] = df['class'].map({'cat': 0, 'dog': 1})

    rng = np.random.default_rng(random_state)
    df['split'] = 'train'

    # Per-class split to maintain rough balance.
    # For tiny datasets (< 6 per class) we force at least 1 sample into each
    # of train/val/test rather than using the percentage fractions, which can
    # round to zero on 3-sample classes.
    for cls in ['cat', 'dog']:
        idx = df.index[df['class'] == cls].tolist()
        rng.shuffle(idx)
        n = len(idx)

        if n < 6:
            # Tiny dataset: 1 val, 1 test, rest train (minimum 1 train)
            n_val  = 1 if n >= 2 else 0
            n_test = 1 if n >= 3 else 0
            n_train = n - n_val - n_test
        else:
            n_train = max(1, round(n * train_split))
            n_val   = max(1, round(n * val_split))
            n_test  = n - n_train - n_val

        for i, sample_idx in enumerate(idx):
            if i < n_train:
                df.at[sample_idx, 'split'] = 'train'
            elif i < n_train + n_val:
                df.at[sample_idx, 'split'] = 'val'
            else:
                df.at[sample_idx, 'split'] = 'test'

    df.to_csv(output_path, index=False)

    counts = df['split'].value_counts()
    print(f"metadata.csv → {output_path}")
    print(f"  train={counts.get('train', 0)}, val={counts.get('val', 0)}, test={counts.get('test', 0)}")
    print(f"  Total: {len(df)} images ({df['class'].value_counts().to_dict()})")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='Generate metadata.csv for IMP exploration.')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to image directory (default: auto-detected)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for metadata.csv (default: data/metadata.csv)')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional YAML config to read data_dir from')
    parser.add_argument('--train-split', type=float, default=0.70)
    parser.add_argument('--val-split', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = _exploration_root.parent.parent

    data_dir = args.data_dir
    output = args.output

    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        if data_dir is None:
            data_dir = project_root / config['data']['data_dir'] / 'dogs-vs-cats' / 'train'
        if output is None:
            output = project_root / config['data']['data_dir'] / 'metadata.csv'

    if data_dir is None:
        data_dir = project_root / 'data' / 'dogs-vs-cats' / 'train'
    if output is None:
        output = project_root / 'data' / 'metadata.csv'

    create_metadata_csv_robust(
        data_dir=data_dir,
        output_path=output,
        train_split=args.train_split,
        val_split=args.val_split,
        random_state=args.seed,
    )


if __name__ == '__main__':
    main()
