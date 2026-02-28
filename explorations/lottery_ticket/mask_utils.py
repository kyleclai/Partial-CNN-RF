"""
mask_utils.py — Weight masking utilities for Iterative Magnitude Pruning (IMP).

Implements:
  - get_prunable_layer_names(): resolve which layers to prune per arch
  - MaskManager: tracks binary masks + initial weights; drives prune/reset cycle
  - MaskEnforcementCallback: Keras callback that re-zeros pruned weights every batch
"""

import numpy as np
import tensorflow as tf
from pathlib import Path


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------

def get_prunable_layer_names(model, arch: str) -> list[str]:
    """Return names of layers whose kernels will be pruned.

    VGG16: only the Dense classification head (conv base is frozen/ImageNet).
    LeNet: all Conv2D and Dense layers.

    Args:
        model: Compiled Keras model.
        arch:  'vgg16' or 'lenet'.

    Returns:
        Ordered list of layer names that have a `.kernel` attribute.

    Raises:
        ValueError: if arch is unknown or a named layer has no kernel.
    """
    arch = arch.lower()
    if arch == 'vgg16':
        candidates = ['fc1', 'fc2', 'output']
    elif arch == 'lenet':
        candidates = ['conv_1', 'conv_2', 'dense_1', 'dense_2', 'output']
    else:
        raise ValueError(f"Unknown arch '{arch}'. Expected 'vgg16' or 'lenet'.")

    validated = []
    for name in candidates:
        try:
            layer = model.get_layer(name)
        except ValueError:
            raise ValueError(
                f"Layer '{name}' not found in model. "
                f"Available: {[l.name for l in model.layers]}"
            )
        if not hasattr(layer, 'kernel'):
            raise ValueError(
                f"Layer '{name}' has no `.kernel` — is it a Dense or Conv2D layer?"
            )
        validated.append(name)

    return validated


# ---------------------------------------------------------------------------
# MaskManager
# ---------------------------------------------------------------------------

class MaskManager:
    """Manages binary weight masks and initial weight snapshots for IMP.

    Attributes:
        masks:           {layer_name: np.ndarray} — float32 binary, kernel shape.
        initial_weights: {layer_name: np.ndarray} — W_0 copy, kernel shape.
        prunable_layers: ordered list of layer names.
    """

    def __init__(self, model, prunable_layer_names: list[str]):
        self.prunable_layers = list(prunable_layer_names)
        self.masks: dict[str, np.ndarray] = {}
        self.initial_weights: dict[str, np.ndarray] = {}

        for name in self.prunable_layers:
            layer = model.get_layer(name)
            shape = layer.kernel.shape
            self.masks[name] = np.ones(shape, dtype=np.float32)

    # ------------------------------------------------------------------
    # Initial weight snapshot (call BEFORE first training round)
    # ------------------------------------------------------------------

    def save_initial_weights(self, model) -> None:
        """Deep-copy kernel weights as W_0. Must be called before any training."""
        for name in self.prunable_layers:
            layer = model.get_layer(name)
            self.initial_weights[name] = layer.kernel.numpy().copy()

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def compute_global_prune_threshold(self, model, prune_fraction: float) -> float:
        """Compute the |W| threshold below which weights are pruned (global).

        Considers only currently unmasked (active) weights.

        Args:
            model:          Keras model with current weights.
            prune_fraction: Fraction of *remaining* active weights to prune.

        Returns:
            Scalar threshold value.
        """
        all_abs_weights = []
        for name in self.prunable_layers:
            layer = model.get_layer(name)
            w = layer.kernel.numpy()
            mask = self.masks[name]
            active = np.abs(w[mask == 1])
            if active.size > 0:
                all_abs_weights.append(active)

        if not all_abs_weights:
            return 0.0

        combined = np.concatenate(all_abs_weights)
        threshold = float(np.percentile(combined, prune_fraction * 100))
        return threshold

    def update_masks(self, model, prune_fraction: float) -> dict:
        """Zero-out mask entries where |W| ≤ threshold AND mask was previously 1.

        Args:
            model:          Keras model.
            prune_fraction: Fraction of currently active weights to prune.

        Returns:
            Dict with per-layer and global sparsity after update.
        """
        threshold = self.compute_global_prune_threshold(model, prune_fraction)

        for name in self.prunable_layers:
            layer = model.get_layer(name)
            w = layer.kernel.numpy()
            old_mask = self.masks[name]
            # Prune: set to 0 if |w| <= threshold AND previously active
            new_mask = np.where(
                (np.abs(w) <= threshold) & (old_mask == 1),
                0.0,
                old_mask
            ).astype(np.float32)
            self.masks[name] = new_mask

        return self.get_sparsity_report(model)

    # ------------------------------------------------------------------
    # Weight reset (the core IMP operation)
    # ------------------------------------------------------------------

    def reset_weights_to_initial(self, model) -> None:
        """Reset kept weights to W_0; zero out pruned positions.

        W_new = W_0 * mask   (element-wise)

        This is the non-negotiable IMP step that distinguishes winning tickets
        from simply fine-tuning a smaller network.
        """
        if not self.initial_weights:
            raise RuntimeError(
                "Initial weights not saved. Call save_initial_weights() before training."
            )
        for name in self.prunable_layers:
            layer = model.get_layer(name)
            w0 = self.initial_weights[name]
            mask = self.masks[name]
            layer.kernel.assign(w0 * mask)

    # ------------------------------------------------------------------
    # Mask enforcement (called by callback every batch)
    # ------------------------------------------------------------------

    def apply_masks_to_model(self, model) -> None:
        """Re-zero any weights that crept into pruned positions (e.g. via Adam).

        Stays on GPU — uses tf.Variable.assign rather than host-device round-trips.
        """
        for name in self.prunable_layers:
            layer = model.get_layer(name)
            mask_tensor = tf.constant(self.masks[name], dtype=tf.float32)
            layer.kernel.assign(layer.kernel * mask_tensor)

    # ------------------------------------------------------------------
    # Sparsity reporting
    # ------------------------------------------------------------------

    def get_sparsity_report(self, model=None) -> dict:
        """Return per-layer and global sparsity statistics.

        Args:
            model: Optional — if provided, verifies live kernel zeros match mask.

        Returns:
            {
              'per_layer': {name: {'sparsity': float, 'zeros': int, 'total': int}},
              'global':    {'sparsity': float, 'zeros': int, 'total': int}
            }
        """
        total_zeros = 0
        total_params = 0
        per_layer = {}

        for name in self.prunable_layers:
            mask = self.masks[name]
            zeros = int(np.sum(mask == 0))
            total = int(mask.size)
            sparsity = zeros / total if total > 0 else 0.0
            per_layer[name] = {
                'sparsity': round(sparsity, 6),
                'zeros': zeros,
                'total': total,
            }
            total_zeros += zeros
            total_params += total

        global_sparsity = total_zeros / total_params if total_params > 0 else 0.0
        return {
            'per_layer': per_layer,
            'global': {
                'sparsity': round(global_sparsity, 6),
                'zeros': total_zeros,
                'total': total_params,
            },
        }

    # ------------------------------------------------------------------
    # Assertions / integrity checks
    # ------------------------------------------------------------------

    def assert_masks_binary(self) -> None:
        """Raise AssertionError if any mask value is not exactly 0 or 1."""
        for name, mask in self.masks.items():
            unique = np.unique(mask)
            assert set(unique).issubset({0.0, 1.0}), (
                f"Mask for '{name}' is not binary. Unique values: {unique}"
            )

    def assert_reset_correct(self, model, rtol: float = 1e-5) -> None:
        """Verify W[mask==1] ≈ W_0[mask==1] and W[mask==0] == 0."""
        for name in self.prunable_layers:
            layer = model.get_layer(name)
            w = layer.kernel.numpy()
            mask = self.masks[name]
            w0 = self.initial_weights[name]

            # Pruned positions must be exactly zero
            pruned_vals = w[mask == 0]
            assert np.all(pruned_vals == 0), (
                f"Layer '{name}': pruned weights are not zero. "
                f"Max abs: {np.abs(pruned_vals).max():.6f}"
            )

            # Active positions must match W_0 within tolerance
            if np.any(mask == 1):
                max_diff = np.abs(w[mask == 1] - w0[mask == 1]).max()
                assert max_diff < rtol, (
                    f"Layer '{name}': active weights differ from W_0 by {max_diff:.2e}"
                )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save masks and initial weights to a .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for name in self.prunable_layers:
            arrays[f'mask_{name}'] = self.masks[name]
            if name in self.initial_weights:
                arrays[f'init_{name}'] = self.initial_weights[name]
        np.savez(str(path), **arrays)

    @classmethod
    def load(cls, path: str | Path, model, prunable_layer_names: list[str]) -> 'MaskManager':
        """Load masks and initial weights from a .npz file.

        Args:
            path:                 Path to .npz saved by MaskManager.save().
            model:                Keras model (needed to instantiate MaskManager).
            prunable_layer_names: Layer names to restore.

        Returns:
            Populated MaskManager instance.
        """
        path = Path(path)
        data = np.load(str(path))
        mm = cls(model, prunable_layer_names)
        for name in prunable_layer_names:
            mask_key = f'mask_{name}'
            init_key = f'init_{name}'
            if mask_key in data:
                mm.masks[name] = data[mask_key].astype(np.float32)
            if init_key in data:
                mm.initial_weights[name] = data[init_key]
        return mm


# ---------------------------------------------------------------------------
# Keras Callback
# ---------------------------------------------------------------------------

class MaskEnforcementCallback(tf.keras.callbacks.Callback):
    """Re-zero pruned weights after every optimizer step.

    Adam accumulates momentum for all parameters, including pruned ones.
    This callback enforces mask sparsity on every batch end, preventing
    momentum from slowly restoring pruned weights over time.

    Args:
        mask_manager: MaskManager instance owning the masks.
    """

    def __init__(self, mask_manager: MaskManager):
        super().__init__()
        self.mask_manager = mask_manager

    def on_batch_end(self, batch: int, logs=None):
        self.mask_manager.apply_masks_to_model(self.model)
