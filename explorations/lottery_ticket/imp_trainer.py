"""
imp_trainer.py — Iterative Magnitude Pruning (IMP) trainer.

Implements the Frankle & Carlin (2019) lottery ticket hypothesis algorithm:
  1. Train a dense network (round 0) and save initial weights W_0.
  2. Prune p% of lowest |W| weights → binary mask M.
  3. Reset kept weights to W_0.
  4. Retrain with mask M enforced every batch.
  5. Repeat steps 2-4 until target sparsity is reached.

Usage:
    python imp_trainer.py --config configs/imp_lenet.yaml
    python imp_trainer.py --config configs/imp_vgg16.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# Path setup — import from parent src/ without modifying it
# ---------------------------------------------------------------------------
_exploration_root = Path(__file__).resolve().parent
_src_path = _exploration_root.parent.parent / 'src'
sys.path.insert(0, str(_src_path))

from utils.model_builders import build_vgg16   # build_lenet imported locally below
from utils.data_loaders import load_split_as_numpy
from utils.seeds import set_global_seed, configure_gpu

from mask_utils import MaskManager, MaskEnforcementCallback, get_prunable_layer_names


# ---------------------------------------------------------------------------
# Local LeNet builder (in case src/ version ever regresses)
# ---------------------------------------------------------------------------

def _build_lenet_local(input_shape=(128, 128, 3), num_classes=1):
    """Local LeNet builder — identical to src/ version, defined here for safety."""
    from keras import layers, models

    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(6, (5, 5), activation='relu', name='conv_1')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool_1')(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(16, (5, 5), activation='relu', name='conv_2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool_2')(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(120, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(84, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(num_classes, activation='sigmoid', name='output')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------

def estimate_flops(model, mask_manager: MaskManager, config: dict) -> float:
    """Estimate total FLOPs for a forward pass, accounting for sparsity.

    For prunable layers, scale FLOPs by (1 - layer_sparsity).
    For VGG16, add the constant frozen base FLOPs.

    Args:
        model:        Keras model.
        mask_manager: Provides per-layer sparsity.
        config:       Full experiment config dict.

    Returns:
        Estimated FLOPs (float).
    """
    arch = config['train']['arch'].lower()
    sparsity_report = mask_manager.get_sparsity_report()
    per_layer_sparsity = {
        name: info['sparsity']
        for name, info in sparsity_report['per_layer'].items()
    }

    total_flops = 0.0

    for layer in model.layers:
        name = layer.name
        sparsity = per_layer_sparsity.get(name, 0.0)
        density = 1.0 - sparsity

        if isinstance(layer, tf.keras.layers.Dense):
            # FLOPs = 2 * in_features * out_features
            in_f, out_f = layer.kernel.shape
            flops = 2.0 * int(in_f) * int(out_f) * density
            total_flops += flops

        elif isinstance(layer, tf.keras.layers.Conv2D):
            # FLOPs = 2 * kH * kW * in_C * out_C * out_H * out_W
            try:
                out_shape = layer.output_shape  # (batch, H, W, C)
                out_h, out_w, out_c = out_shape[1], out_shape[2], out_shape[3]
                kh, kw = layer.kernel_size
                in_c = layer.kernel.shape[2]
                flops = 2.0 * kh * kw * int(in_c) * int(out_c) * out_h * out_w * density
                total_flops += flops
            except Exception:
                pass  # Skip if output shape not available (e.g. nested model)

    # Add constant frozen VGG16 conv base if applicable
    flops_cfg = config.get('flops', {})
    if arch == 'vgg16' and flops_cfg.get('include_frozen_base', False):
        base_flops = flops_cfg.get('vgg16_base_flops', 14_714_734_080)
        total_flops += float(base_flops)

    return total_flops


# ---------------------------------------------------------------------------
# IMPTrainer
# ---------------------------------------------------------------------------

class IMPTrainer:
    """Orchestrates the full Iterative Magnitude Pruning experiment.

    Args:
        config: Parsed YAML config dict.
    """

    def __init__(self, config: dict):
        self.config = config
        self.arch = config['train']['arch'].lower()
        self.seed = config['run'].get('seed', 42)
        self.results_dir = Path(config['imp']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self.x_train = self.x_val = self.y_train = self.y_val = None

        # IMP parameters
        self.prune_rate = config['imp']['prune_rate_per_round']
        self.num_rounds = config['imp']['num_rounds']
        self.target_sparsity = config['imp']['target_sparsity']
        self.epochs_per_round = config['train']['epochs_per_round']
        self.batch_size = config['train']['batch_size']
        self.lr = config['train'].get('learning_rate', 0.001)

        # Metadata path resolution (relative to project root)
        self._project_root = _exploration_root.parent.parent
        self._metadata_csv = self._project_root / config['data']['data_dir'] / 'metadata.csv'

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_fresh_model(self) -> tf.keras.Model:
        """Build and compile an unmasked model from scratch."""
        set_global_seed(self.seed)
        img_size = tuple(self.config['data']['image_size'])
        channels = self.config['data'].get('channels', 3)
        input_shape = (*img_size, channels)

        if self.arch == 'vgg16':
            model = build_vgg16(input_shape=input_shape, num_classes=1, freeze_base=True)
        elif self.arch == 'lenet':
            model = _build_lenet_local(input_shape=input_shape, num_classes=1)
        else:
            raise ValueError(f"Unknown arch: '{self.arch}'")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        return model

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Load train and val splits as NumPy arrays (cached in self)."""
        if self.x_train is not None:
            return  # Already loaded

        if not self._metadata_csv.exists():
            print(
                f"metadata.csv not found at {self._metadata_csv}. "
                "Auto-generating via prepare_data.py..."
            )
            from prepare_data import create_metadata_csv_robust
            data_img_dir = self._project_root / self.config['data']['data_dir'] / 'dogs-vs-cats' / 'train'
            create_metadata_csv_robust(
                data_dir=data_img_dir,
                output_path=self._metadata_csv,
                train_split=self.config['data'].get('train_split', 0.70),
                val_split=self.config['data'].get('val_split', 0.20),
                random_state=self.seed,
            )

        img_size = tuple(self.config['data']['image_size'])
        limit = self.config['data'].get('limit_total')
        train_limit = int(limit * self.config['data'].get('train_split', 0.70)) if limit else None
        val_limit   = int(limit * self.config['data'].get('val_split',   0.20)) if limit else None

        print("Loading training data...")
        self.x_train, self.y_train = load_split_as_numpy(
            str(self._metadata_csv), split='train',
            img_size=img_size, limit=train_limit, random_state=self.seed,
        )
        print("Loading validation data...")
        self.x_val, self.y_val = load_split_as_numpy(
            str(self._metadata_csv), split='val',
            img_size=img_size, limit=val_limit, random_state=self.seed,
        )

    # ------------------------------------------------------------------
    # Single training round
    # ------------------------------------------------------------------

    def train_one_round(
        self,
        model: tf.keras.Model,
        mask_manager: MaskManager,
        round_idx: int,
    ) -> dict:
        """Train model for one IMP round and collect metrics.

        Args:
            model:        Keras model (already reset to W_0 * mask if round ≥ 1).
            mask_manager: Active MaskManager.
            round_idx:    0 = dense baseline, ≥1 = pruned rounds.

        Returns:
            Metrics dict for this round.
        """
        callbacks = [MaskEnforcementCallback(mask_manager)]

        if self.config['train'].get('early_stopping', False):
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', patience=3,
                    restore_best_weights=True,
                )
            )

        # Use val set if it has samples; fall back to train set for tiny datasets
        has_val = len(self.x_val) > 0
        eval_x = self.x_val  if has_val else self.x_train
        eval_y = self.y_val  if has_val else self.y_train
        eval_label = 'val' if has_val else 'train (no val split)'

        print(f"\n--- Round {round_idx}: training for {self.epochs_per_round} epochs ---")
        fit_kwargs = dict(
            epochs=self.epochs_per_round,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        if has_val:
            fit_kwargs['validation_data'] = (self.x_val, self.y_val)

        history = model.fit(self.x_train, self.y_train, **fit_kwargs)

        # Classification metrics
        y_pred_proba = model.predict(eval_x, verbose=0).ravel()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        val_acc  = float(np.mean(y_pred == eval_y))
        val_f1   = float(f1_score(eval_y, y_pred, zero_division=0))
        val_prec = float(precision_score(eval_y, y_pred, zero_division=0))
        val_rec  = float(recall_score(eval_y, y_pred, zero_division=0))
        print(f"  Metrics on {eval_label}: acc={val_acc:.4f}, f1={val_f1:.4f}")

        # Inference time (5 warm passes over eval set)
        times = []
        n_eval = len(eval_x)
        for _ in range(5):
            t0 = time.perf_counter()
            model.predict(eval_x, batch_size=self.batch_size, verbose=0)
            t1 = time.perf_counter()
            times.append((t1 - t0) / n_eval * 1000)  # ms per sample
        inference_ms = float(np.mean(times))

        # Sparsity and FLOPs
        sparsity_report = mask_manager.get_sparsity_report()
        flops = estimate_flops(model, mask_manager, self.config)

        # Trainable parameter count (non-zero only)
        total_params = sum(
            int(np.prod(layer.kernel.shape))
            for layer in model.layers
            if hasattr(layer, 'kernel')
            and layer.name in mask_manager.prunable_layers
        )
        active_params = total_params - sparsity_report['global']['zeros']

        metrics = {
            'round': round_idx,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'global_sparsity': sparsity_report['global']['sparsity'],
            'per_layer_sparsity': sparsity_report['per_layer'],
            'estimated_flops': flops,
            'active_params': active_params,
            'total_prunable_params': total_params,
            'inference_ms_per_sample': inference_ms,
            'train_history': {
                'loss': [float(v) for v in history.history['loss']],
                'val_loss': [float(v) for v in history.history.get('val_loss', [])],
                'accuracy': [float(v) for v in history.history['accuracy']],
                'val_accuracy': [float(v) for v in history.history.get('val_accuracy', [])],
            },
        }

        print(
            f"Round {round_idx} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f} "
            f"| global_sparsity={sparsity_report['global']['sparsity']:.4f} "
            f"| FLOPs={flops:.3e} | inf={inference_ms:.3f}ms/sample"
        )
        return metrics

    # ------------------------------------------------------------------
    # Main IMP loop
    # ------------------------------------------------------------------

    def run(self) -> list[dict]:
        """Execute the full IMP experiment.

        Returns:
            List of per-round metric dicts (also saved to results_dir).
        """
        configure_gpu(self.config['run'].get('device', 'auto'))
        self.load_data()

        prunable_layers = self.config['imp']['prunable_layers']
        all_results: list[dict] = []

        # ------- Round 0: dense baseline -----------------------------------
        set_global_seed(self.seed)
        model = self.build_fresh_model()
        mask_manager = MaskManager(model, prunable_layers)
        mask_manager.save_initial_weights(model)   # Snapshot W_0 BEFORE training

        print(f"\n{'='*60}")
        print(f"IMP Experiment: arch={self.arch}, "
              f"prune_rate={self.prune_rate}, target_sparsity={self.target_sparsity}")
        print(f"Prunable layers: {prunable_layers}")
        print(f"{'='*60}")

        round_0_metrics = self.train_one_round(model, mask_manager, round_idx=0)
        all_results.append(round_0_metrics)

        # Save round-0 model and mask
        self._save_round_artifacts(model, mask_manager, round_idx=0)

        # ------- Pruning rounds 1..N -----------------------------------
        for round_idx in range(1, self.num_rounds + 1):
            sparsity_before = all_results[-1]['global_sparsity']
            print(f"\n--- Pruning round {round_idx} "
                  f"(current global sparsity: {sparsity_before:.4f}) ---")

            # Step 1: Update masks (prune lowest |W|)
            mask_manager.assert_masks_binary()
            sparsity_info = mask_manager.update_masks(model, prune_fraction=self.prune_rate)
            new_sparsity = sparsity_info['global']['sparsity']
            print(f"  Global sparsity after pruning: {new_sparsity:.4f}")

            # Step 2: Reset kept weights to W_0
            mask_manager.reset_weights_to_initial(model)
            mask_manager.assert_reset_correct(model)

            # Step 3: Re-compile (fresh optimizer state) + retrain
            set_global_seed(self.seed)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                loss='binary_crossentropy',
                metrics=['accuracy'],
            )

            round_metrics = self.train_one_round(model, mask_manager, round_idx=round_idx)
            all_results.append(round_metrics)

            # Save per-round artifacts
            self._save_round_artifacts(model, mask_manager, round_idx=round_idx)

            # Early exit if target sparsity reached
            if new_sparsity >= self.target_sparsity:
                print(f"\nTarget sparsity {self.target_sparsity:.2%} reached. Stopping.")
                break

        # ------- Save results and identify winning ticket -------------------
        results_path = self.results_dir / 'sparsity_vs_accuracy.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {results_path}")

        self._copy_winning_ticket(all_results, model, mask_manager)

        print("\nIMP experiment complete.")
        self._print_summary(all_results)
        return all_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_round_artifacts(
        self,
        model: tf.keras.Model,
        mask_manager: MaskManager,
        round_idx: int,
    ) -> None:
        """Save model weights and mask for a specific round."""
        weights_path = self.results_dir / f'model_round_{round_idx}.weights.h5'
        mask_path    = self.results_dir / f'mask_round_{round_idx}.npz'

        model.save_weights(str(weights_path))
        mask_manager.save(mask_path)
        print(f"  Saved: {weights_path.name}, {mask_path.name}")

    def _copy_winning_ticket(
        self,
        all_results: list[dict],
        model: tf.keras.Model,
        mask_manager: MaskManager,
    ) -> None:
        """Identify winning round (best val_accuracy at round ≥ 1) and copy artifacts."""
        pruned_rounds = [r for r in all_results if r['round'] >= 1]
        if not pruned_rounds:
            print("No pruned rounds to evaluate for winning ticket.")
            return

        winning = max(pruned_rounds, key=lambda r: r['val_accuracy'])
        winning_round = winning['round']

        print(f"\nWinning ticket: round {winning_round} "
              f"(val_accuracy={winning['val_accuracy']:.4f}, "
              f"global_sparsity={winning['global_sparsity']:.4f})")

        # Reload winning round artifacts
        src_weights = self.results_dir / f'model_round_{winning_round}.weights.h5'
        src_mask    = self.results_dir / f'mask_round_{winning_round}.npz'
        dst_weights = self.results_dir / 'winning_ticket_model.weights.h5'
        dst_mask    = self.results_dir / 'winning_ticket_mask.npz'

        import shutil
        shutil.copy2(src_weights, dst_weights)
        shutil.copy2(src_mask, dst_mask)

        # Also save winning round index for extractor
        meta = {
            'winning_round': winning_round,
            'arch': self.arch,
            'winning_metrics': winning,
        }
        with open(self.results_dir / 'winning_ticket_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"  Winning ticket artifacts → {dst_weights.name}, {dst_mask.name}")

    def _print_summary(self, all_results: list[dict]) -> None:
        """Print a Markdown-style summary table of all rounds."""
        header = (
            f"{'Round':>6} | {'Val Acc':>8} | {'Val F1':>7} | "
            f"{'Sparsity':>9} | {'FLOPs':>12} | {'Inf (ms)':>9}"
        )
        print(f"\n{header}")
        print('-' * len(header))
        for r in all_results:
            print(
                f"{r['round']:>6} | {r['val_accuracy']:>8.4f} | {r['val_f1']:>7.4f} | "
                f"{r['global_sparsity']:>9.4f} | {r['estimated_flops']:>12.3e} | "
                f"{r['inference_ms_per_sample']:>9.3f}"
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Iterative Magnitude Pruning (IMP) trainer.'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config file (e.g. configs/imp_lenet.yaml)',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Resolve relative to CWD
        config_path = Path.cwd() / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    trainer = IMPTrainer(config)
    trainer.run()


if __name__ == '__main__':
    main()
