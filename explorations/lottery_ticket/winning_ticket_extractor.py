"""
winning_ticket_extractor.py — Extract and verify IMP winning ticket subnetworks.

After IMP training completes, this module:
  1. Identifies the winning round (best val_accuracy among pruned rounds).
  2. Loads the winning model weights + mask.
  3. Permanently applies the mask (zeros pruned weights in place).
  4. Saves a canonical winning_ticket_model.keras + winning_ticket_mask.npz.
  5. Verifies accuracy matches logged metrics.
  6. Optionally plugs the pruned CNN into the hybrid CNN-RF pipeline to
     check whether pruned features are still RF-compatible (Question 4).

Usage:
    python winning_ticket_extractor.py --results explorations/lottery_ticket/results/lenet
    python winning_ticket_extractor.py --results explorations/lottery_ticket/results/vgg16 \\
        --verify --plug-into-hybrid
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_exploration_root = Path(__file__).resolve().parent
_src_path = _exploration_root.parent.parent / 'src'
sys.path.insert(0, str(_src_path))

from utils.data_loaders import load_split_as_numpy

from mask_utils import MaskManager
from imp_trainer import _build_lenet_local, build_vgg16


# ---------------------------------------------------------------------------
# WinningTicketExtractor
# ---------------------------------------------------------------------------

class WinningTicketExtractor:
    """Loads, verifies, and saves the IMP winning ticket subnetwork.

    Args:
        results_dir: Path to the round artifacts produced by IMPTrainer.
    """

    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)
        self._results_path = self.results_dir / 'sparsity_vs_accuracy.json'
        self._meta_path    = self.results_dir / 'winning_ticket_meta.json'

        if not self._results_path.exists():
            raise FileNotFoundError(
                f"Results file not found: {self._results_path}\n"
                "Run imp_trainer.py first."
            )

        with open(self._results_path) as f:
            self.results: list[dict] = json.load(f)

        # Load pre-identified winning meta if available
        self._winning_meta: dict | None = None
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                self._winning_meta = json.load(f)

    # ------------------------------------------------------------------
    # Round selection
    # ------------------------------------------------------------------

    def find_winning_round(self, metric: str = 'val_accuracy') -> dict:
        """Return the round dict with the best `metric` among rounds ≥ 1.

        Args:
            metric: Key to maximise. Default 'val_accuracy'.

        Returns:
            Round dict (same structure as sparsity_vs_accuracy.json entries).
        """
        pruned = [r for r in self.results if r['round'] >= 1]
        if not pruned:
            raise ValueError("No pruned rounds found in results.")
        winning = max(pruned, key=lambda r: r[metric])
        print(
            f"Winning ticket: round {winning['round']} | "
            f"{metric}={winning[metric]:.4f} | "
            f"global_sparsity={winning['global_sparsity']:.4f}"
        )
        return winning

    # ------------------------------------------------------------------
    # Model + mask loading
    # ------------------------------------------------------------------

    def load_winning_ticket(
        self,
        winning_round: int,
        arch: str,
        config: dict | None = None,
    ) -> tuple[tf.keras.Model, MaskManager]:
        """Reconstruct model + MaskManager for the winning round.

        Args:
            winning_round: Round index to load.
            arch:          'vgg16' or 'lenet'.
            config:        Optional config dict for model shape. If None, uses
                           defaults (128×128×3 input, binary output).

        Returns:
            (model, mask_manager) with weights and masks loaded.
        """
        arch = arch.lower()
        img_size = (128, 128)
        if config:
            img_size = tuple(config['data']['image_size'])
        input_shape = (*img_size, 3)

        if arch == 'vgg16':
            model = build_vgg16(input_shape=input_shape, num_classes=1, freeze_base=True)
            prunable_layers = ['fc1', 'fc2', 'output']
        elif arch == 'lenet':
            model = _build_lenet_local(input_shape=input_shape, num_classes=1)
            prunable_layers = ['conv_1', 'conv_2', 'dense_1', 'dense_2', 'output']
        else:
            raise ValueError(f"Unknown arch: '{arch}'")

        # Load mask
        mask_path = self.results_dir / f'mask_round_{winning_round}.npz'
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask_manager = MaskManager.load(mask_path, model, prunable_layers)

        # Load weights
        weights_path = self.results_dir / f'model_round_{winning_round}.weights.h5'
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        model.load_weights(str(weights_path))

        print(f"Loaded round {winning_round} model ({arch}) and mask from {self.results_dir}")
        sparsity = mask_manager.get_sparsity_report()
        print(f"  Global sparsity: {sparsity['global']['sparsity']:.4f}")

        return model, mask_manager

    # ------------------------------------------------------------------
    # Permanent mask application + save
    # ------------------------------------------------------------------

    def extract_subnetwork(
        self,
        model: tf.keras.Model,
        mask_manager: MaskManager,
    ) -> tf.keras.Model:
        """Permanently zero pruned weights and save canonical winning ticket.

        Applies mask in-place (kernel * mask), then saves:
          - results_dir/winning_ticket_model.keras
          - results_dir/winning_ticket_mask.npz

        Args:
            model:        Model loaded by load_winning_ticket().
            mask_manager: Corresponding MaskManager.

        Returns:
            model with pruned weights permanently zeroed.
        """
        # Apply mask permanently
        for name in mask_manager.prunable_layers:
            layer = model.get_layer(name)
            mask_tensor = tf.constant(mask_manager.masks[name], dtype=tf.float32)
            layer.kernel.assign(layer.kernel * mask_tensor)

        # Compile (needed to call model.evaluate later)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        # Save model
        model_out = self.results_dir / 'winning_ticket_model.keras'
        model.save(str(model_out))
        print(f"Saved winning ticket model → {model_out}")

        # Save mask
        mask_out = self.results_dir / 'winning_ticket_mask.npz'
        mask_manager.save(mask_out)
        print(f"Saved winning ticket mask  → {mask_out}")

        return model

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_ticket(
        self,
        model: tf.keras.Model,
        x_val: np.ndarray,
        y_val: np.ndarray,
        logged_accuracy: float,
        tolerance: float = 0.005,
    ) -> bool:
        """Evaluate saved model and compare to logged round accuracy.

        Args:
            model:            Model returned by extract_subnetwork().
            x_val, y_val:     Validation split arrays.
            logged_accuracy:  val_accuracy recorded during IMP training.
            tolerance:        Max allowed absolute difference (default 0.5%).

        Returns:
            True if verification passes, False otherwise.
        """
        y_pred = (model.predict(x_val, verbose=0).ravel() >= 0.5).astype(int)
        live_acc = float(np.mean(y_pred == y_val))
        diff = abs(live_acc - logged_accuracy)

        status = "PASS" if diff <= tolerance else "FAIL"
        print(
            f"Verification {status}: "
            f"live_acc={live_acc:.4f} | logged={logged_accuracy:.4f} | diff={diff:.4f}"
        )
        return diff <= tolerance

    # ------------------------------------------------------------------
    # Hybrid pipeline compatibility check (Question 4)
    # ------------------------------------------------------------------

    def plug_into_hybrid_pipeline(
        self,
        model: tf.keras.Model,
        x_data: np.ndarray,
        arch: str,
        layer_names: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Extract GAP features from pruned CNN for RF compatibility check.

        Answers experimental question 4: can the winning ticket's feature maps
        still serve as useful RF inputs?

        Builds sub-models up to each layer, applies Global Average Pooling,
        and returns feature arrays. These can be passed directly to sklearn RF.

        Args:
            model:       Pruned winning ticket model.
            x_data:      Input images, shape (N, H, W, C).
            arch:        'vgg16' or 'lenet'.
            layer_names: Layers to extract from. Defaults to prunable layers.

        Returns:
            {layer_name: np.ndarray of shape (N, C)} — one entry per layer.
        """
        arch = arch.lower()
        if layer_names is None:
            if arch == 'lenet':
                layer_names = ['conv_1', 'conv_2', 'dense_1', 'dense_2']
            else:
                layer_names = ['fc1', 'fc2']

        features: dict[str, np.ndarray] = {}

        for name in layer_names:
            try:
                layer = model.get_layer(name)
                extractor = tf.keras.Model(
                    inputs=model.input,
                    outputs=layer.output,
                )
                acts = extractor.predict(x_data, verbose=0)

                # Apply GAP if spatial (Conv2D output); Dense outputs used directly
                if acts.ndim == 4:
                    acts = acts.mean(axis=(1, 2))  # (N, H, W, C) → (N, C)

                features[name] = acts
                print(f"  Extracted '{name}': shape={acts.shape}")
            except Exception as e:
                print(f"  Warning: could not extract from '{name}': {e}")

        return features


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract and verify IMP winning ticket subnetwork.'
    )
    parser.add_argument(
        '--results', type=str, required=True,
        help='Path to results directory (e.g. explorations/lottery_ticket/results/lenet)',
    )
    parser.add_argument(
        '--arch', type=str, default=None,
        help='Architecture override (vgg16 or lenet). Auto-detected from meta if omitted.',
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Load val data and verify ticket accuracy against logged metrics.',
    )
    parser.add_argument(
        '--plug-into-hybrid', action='store_true',
        help='Extract GAP features from pruned CNN for RF compatibility check.',
    )
    parser.add_argument(
        '--metadata-csv', type=str, default=None,
        help='Path to metadata.csv (auto-resolved from project root if omitted).',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results)

    extractor = WinningTicketExtractor(results_dir)

    # Determine arch
    arch = args.arch
    if arch is None:
        if extractor._winning_meta:
            arch = extractor._winning_meta['arch']
        else:
            raise ValueError(
                "Cannot auto-detect arch. Pass --arch vgg16 or --arch lenet."
            )

    # Find and load winning ticket
    winning_round_info = extractor.find_winning_round()
    winning_round = winning_round_info['round']

    model, mask_manager = extractor.load_winning_ticket(winning_round, arch)
    model = extractor.extract_subnetwork(model, mask_manager)

    # Optional: verify
    if args.verify:
        meta_csv = args.metadata_csv
        if meta_csv is None:
            project_root = _exploration_root.parent.parent
            meta_csv = str(project_root / 'data' / 'metadata.csv')

        print("\nLoading validation data for verification...")
        x_val, y_val = load_split_as_numpy(meta_csv, split='val', img_size=(128, 128))
        extractor.verify_ticket(
            model, x_val, y_val,
            logged_accuracy=winning_round_info['val_accuracy'],
        )

    # Optional: plug into hybrid pipeline
    if args.plug_into_hybrid:
        meta_csv = args.metadata_csv
        if meta_csv is None:
            project_root = _exploration_root.parent.parent
            meta_csv = str(project_root / 'data' / 'metadata.csv')

        print("\nExtracting features for hybrid pipeline compatibility check...")
        x_val, _ = load_split_as_numpy(meta_csv, split='val', img_size=(128, 128))
        features = extractor.plug_into_hybrid_pipeline(model, x_val, arch)
        for layer_name, feat in features.items():
            out_path = results_dir / f'winning_ticket_features_{layer_name}.npz'
            np.savez(str(out_path), features=feat)
            print(f"  Saved features → {out_path.name}")

    print("\nExtraction complete.")


if __name__ == '__main__':
    main()
