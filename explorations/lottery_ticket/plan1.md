# Plan: Implement IMP (Lottery Ticket Pruning) Exploration

## Context

 The explorations/lottery_ticket/ branch currently has only a CLAUDE.md spec — no
 implementation files exist yet. This plan implements the full Iterative Magnitude Pruning
 (IMP) algorithm from Frankle & Carlin (2019) on VGG16 and LeNet baselines, creating the five
 Python files and two YAML configs defined in the spec.

 ---
## Critical Pre-Note: Bug in src/utils/model_builders.py

 build_lenet() (line 77) references undefined variable outputs — should be x. Since we cannot
 modify src/, the exploration will define its own build_lenet locally in imp_trainer.py rather
  than importing the broken one. build_vgg16 is unaffected and will be imported normally.

 ---
## Files to Create

 All files go in explorations/lottery_ticket/ unless noted.

 ┌─────────────────────────────┬───────────────────────────────────────────────────────────┐
 │            File             │                          Purpose                          │
 ├─────────────────────────────┼───────────────────────────────────────────────────────────┤
 │ mask_utils.py               │ MaskManager class + MaskEnforcementCallback Keras         │
 │                             │ callback                                                  │
 ├─────────────────────────────┼───────────────────────────────────────────────────────────┤
 │ imp_trainer.py              │ Main IMP loop orchestrator; CLI entry point               │
 ├─────────────────────────────┼───────────────────────────────────────────────────────────┤
 │ winning_ticket_extractor.py │ Extract winning subnetwork after IMP completes            │
 ├─────────────────────────────┼───────────────────────────────────────────────────────────┤
 │ compare_metrics.py          │ Plotly visualizations: sparsity vs accuracy/FLOPs         │
 ├─────────────────────────────┼───────────────────────────────────────────────────────────┤
 │ configs/imp_vgg16.yaml      │ IMP config for VGG16 (GPU, full 25K dataset)              │
 ├─────────────────────────────┼───────────────────────────────────────────────────────────┤
 │ configs/imp_lenet.yaml      │ IMP config for LeNet (CPU, 2K subset)                     │
 └─────────────────────────────┴───────────────────────────────────────────────────────────┘

 ---
 ### File 1: mask_utils.py

 Key Data Structures

```
 masks: dict[str, np.ndarray]          # {layer_name: float32 binary array, same shape as
 kernel}
 initial_weights: dict[str, np.ndarray] # {layer_name: W_0 copy}
 prunable_layers: list[str]             # ordered layer names

 get_prunable_layer_names(model, arch) (module-level)

 - 'vgg16' → ['fc1', 'fc2', 'output'] (Dense head only; conv base is frozen)
 - 'lenet' → ['conv_1', 'conv_2', 'dense_1', 'dense_2', 'output'] (all layers)
 - Validates each name has a .kernel attribute; raises ValueError if not

 MaskManager class

 - __init__(model, prunable_layer_names) — initializes all masks to all-ones
 - save_initial_weights(model) — deep-copy .kernel.numpy() for each prunable layer; call
 before any training
 - compute_global_prune_threshold(model, prune_fraction) — collect |W| at all
 currently-unpruned positions across all layers → np.percentile(..., prune_fraction * 100)
 - update_masks(model, prune_fraction) — set mask=0 where |W| <= threshold AND old_mask==1;
 return per-layer + global sparsity dict
 - reset_weights_to_initial(model) — layer.kernel.assign(W_0 * mask) for each layer (core IMP
 operation)
 - apply_masks_to_model(model) — layer.kernel.assign(layer.kernel * mask) (used by callback)
 - get_sparsity_report(model) → {'per_layer': {...}, 'global': {...}}
 - save(path) / load(path, ...) — .npz with keys mask_{name} and init_{name}

 MaskEnforcementCallback (Keras Callback)

 - on_batch_end → calls mask_manager.apply_masks_to_model(self.model)
 - Uses layer.kernel.assign() (stays on GPU, no host-device copy)
 - Critical: prevents Adam momentum from restoring pruned weights
```

 ---
 ### File 2: imp_trainer.py

```
 sys.path setup (top of file)

 _src_path = Path(__file__).resolve().parent.parent.parent / 'src'
 sys.path.insert(0, str(_src_path))
 from utils.model_builders import build_vgg16   # build_lenet is broken; define locally
 from utils.data_loaders import load_split_as_numpy
 from utils.seeds import set_global_seed, configure_gpu

 Local build_lenet() (fix for the src/ bug)

 Identical to src/utils/model_builders.py but with outputs=x → model = Model(inputs, x)
 corrected.

 IMPTrainer class

 - build_fresh_model() — calls build_vgg16 or local build_lenet; compiles with Adam +
 binary_crossentropy
 - load_data() — load_split_as_numpy() for train + val splits
 - train_one_round(model, data, mask_manager, round_idx):
   a. Attach MaskEnforcementCallback + optional EarlyStopping
   b. model.fit() for epochs_per_round
   c. Compute val accuracy, F1, precision, recall (scikit-learn)
   d. estimate_flops(model, mask_manager, config)
   e. Measure inference time (5 warm passes over val set, mean ms/sample)
   f. Return round metrics dict

 run() — main IMP loop

 round 0:  train dense baseline (masks = all ones, no pruning)
 round 1+: update_masks → reset_weights_to_W0 → train
           save model_round_N.weights.h5 + mask_round_N.npz per round
           break early if global sparsity > target_sparsity
 save results/sparsity_vs_accuracy.json
 identify winning round (best val_accuracy among rounds ≥ 1)
 copy winning model/mask to canonical results/ names

 estimate_flops(model, mask_manager, config)

 - Dense: 2 * in * out * (1 - layer_sparsity)
 - Conv2D: 2 * kH * kW * in_C * out_C * out_H * out_W * (1 - layer_sparsity)
 - VGG16: adds config['flops']['vgg16_base_flops'] constant for frozen conv base
```

 ---
 ### File 3: winning_ticket_extractor.py

```
 WinningTicketExtractor class

 - find_winning_round(results, metric='val_accuracy') — best result at round ≥ 1
 - load_winning_ticket(winning_round, arch) → (model, mask_manager)
 - extract_subnetwork(model, mask_manager) — applies mask permanently; saves
 results/winning_ticket_model.keras and results/winning_ticket_mask.npz
 - verify_ticket(model, x_val, y_val) — independent eval to confirm saved model matches logged
  metrics
 - plug_into_hybrid_pipeline(model, layer_names, x_data, arch) — extracts GAP features from
 pruned CNN for RF compatibility check (answers experimental question 4)
```

 ---
 ### File 4: compare_metrics.py

 Functions

```
 - load_results(path) — load sparsity_vs_accuracy.json
 - plot_sparsity_curve(results, output_path, arch) — Plotly HTML: sparsity % (x) vs
 val_accuracy + val_F1 (y, dual lines) + inference time (secondary y); dashed horizontal line
 at round-0 dense baseline; Nature-style colors (#E64B35 accuracy, #4DBBD5 F1, #3C5488
 inference)
 - plot_sparsity_vs_param_count(results, output_path, arch) — param count (log x) vs accuracy;
  annotate winning ticket
 - compare_with_hybrid_pipeline(imp_results, hybrid_metrics_path, output_path) — IMP vs
 block3_conv3+RF comparison on accuracy vs FLOPs
 - generate_summary_table(results) — Markdown table string of all rounds
 - main() — CLI: --results, --arch, --output-dir
```

 ---
 ### File 5: configs/imp_vgg16.yaml

```
 run:
   name: imp_vgg16
   seed: 42
   device: auto
   artifacts_dir: artifacts

 data:
   data_dir: data
   image_size: [128, 128]
   channels: 3
   limit_total: null   # full 25K
   train_split: 0.70
   val_split: 0.20

 train:
   arch: vgg16
   epochs_per_round: 10
   batch_size: 32
   learning_rate: 0.001
   early_stopping: false

 imp:
   prunable_layers: [fc1, fc2, output]
   prune_rate_per_round: 0.20
   num_rounds: 10
   target_sparsity: 0.90
   results_dir: explorations/lottery_ticket/results/vgg16

 flops:
   include_frozen_base: true
   vgg16_base_flops: 14_714_734_080   # constant at 128×128 input

 report:
   compare_with_hybrid_pipeline: true
   hybrid_pipeline_metrics_path: artifacts/full_vgg16_gpu/test_metrics.json
```

 ### File 6: configs/imp_lenet.yaml

```
 run:
   name: imp_lenet
   seed: 42
   device: cpu
   artifacts_dir: artifacts

 data:
   data_dir: data
   image_size: [128, 128]
   channels: 3
   limit_total: 2000
   train_split: 0.70
   val_split: 0.20

 train:
   arch: lenet
   epochs_per_round: 20
   batch_size: 32
   learning_rate: 0.001
   early_stopping: false

 imp:
   prunable_layers: [conv_1, conv_2, dense_1, dense_2, output]
   prune_rate_per_round: 0.20
   num_rounds: 10
   target_sparsity: 0.90
   results_dir: explorations/lottery_ticket/results/lenet

 flops:
   include_frozen_base: false

 report:
   compare_with_hybrid_pipeline: false
```

 ---
 ## Implementation Notes

 VGG16 fc1 kernel shape: At 128×128 input through all 5 VGG16 pool blocks (each halves spatial
  res): 128→4. Output of block5_pool is (4, 4, 512) = 8,192. So fc1 kernel is (8192, 256) ≈
 2.1M params (not 25,088×256 as in standard 224×224 VGG16). Use layer.kernel.shape[0]
 dynamically, never hardcode.

 Optimizer state: Do NOT reset Adam between rounds — IMP paper does not. Re-compile model at
 each round to reset optimizer (which is correct behavior), but re-set global seed first for
 reproducibility.

 Adam momentum on pruned weights: on_batch_end mask enforcement handles this. After enough
 batches, Adam's moments for zeroed weights converge to near-zero naturally.

 Sparsity distribution (LeNet): dense_1 kernel is (X, 120) where X = 30×30×16 = 14,400 (two
 MaxPool-by-2 from 128×128). That's 1,728,000 params — ~99% of all prunable LeNet params.
 Global pruning will heavily concentrate sparsity in dense_1. This is an important finding to
 highlight.

 ---
 ## Verification Steps

 1. Mask integrity: After each update_masks(), assert masks are binary and monotonically
 sparser
 2. W_0 reset check: After reset_weights_to_initial(), verify W[mask==1] == W_0[mask==1] and
 W[mask==0] == 0
 3. Sparsity schedule: After N rounds at 20% rate, global sparsity ≈ 1 - 0.8^N
 4. Mask enforcement: After round 1 training, inspect pruned positions → all within float
 epsilon of 0
 5. Smoke test: limit_total: 200, num_rounds: 2, epochs_per_round: 2 → full pipeline in <2 min
  on CPU
 6. Winning ticket verify: verify_ticket() matches logged round accuracy within 0.5%

 ## Run Commands

 ### LeNet smoke test (fast)
 python explorations/lottery_ticket/imp_trainer.py --config
 explorations/lottery_ticket/configs/imp_lenet.yaml

 ### VGG16 full experiment (GPU)
 python explorations/lottery_ticket/imp_trainer.py --config
 explorations/lottery_ticket/configs/imp_vgg16.yaml

 ### Extract and verify winning ticket
 python explorations/lottery_ticket/winning_ticket_extractor.py \
   --results explorations/lottery_ticket/results/lenet

 ### Generate plots
 python explorations/lottery_ticket/compare_metrics.py \
   --results explorations/lottery_ticket/results/lenet/sparsity_vs_accuracy.json \
   --arch lenet --output-dir explorations/lottery_ticket/results/lenet

 ## Key design choices implemented

  - W_0 reset (reset_weights_to_initial): kernel.assign(W_0 * mask) — the non-negotiable IMP
  operation
  - Mask enforcement callback: on_batch_end calls kernel.assign(kernel * mask) staying on GPU,
  preventing Adam momentum from restoring pruned weights
  - Optimizer re-compile per round: fresh Adam state each pruning round (matches IMP paper);
  set_global_seed before each compile for reproducibility
  - Global pruning threshold: np.percentile over all currently-active |W| across all prunable
  layers simultaneously
  - Integrity assertions: assert_masks_binary() and assert_reset_correct() called at round
  boundaries
  - FLOPs estimation: density-scaled per-layer arithmetic, plus the 14.7 GFLOPs VGG16 base
  constant