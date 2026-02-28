# Agent: Lottery Ticket Pruning Explorer

## Mission
Implement the Iterative Magnitude Pruning (IMP) algorithm from Frankle & Carlin (2019) on this project's VGG16 and LeNet baselines. Identify "winning ticket" subnetworks. Measure how accuracy and inference cost change as a function of sparsity level. Compare winning ticket subnetworks against the hybrid CNN-RF pipeline already in place.

## Scope
- Files you may read: `src/train_cnn.py`, `src/utils/model_builders.py`, `configs/full_vgg16_gpu.yaml`, `configs/demo_lenet_cpu.yaml`
- Files you will create: `explorations/lottery_ticket/`
  - `imp_trainer.py` — implements IMP loop
  - `mask_utils.py` — weight masking, sparsity tracking
  - `winning_ticket_extractor.py` — extracts and re-initializes subnetwork
  - `compare_metrics.py` — plots sparsity vs accuracy, sparsity vs inference time
  - `configs/imp_vgg16.yaml`, `configs/imp_lenet.yaml`

## Algorithm to Implement
1. Randomly initialize network. Save initial weights W_0.
2. Train for k epochs. Save final weights W_T.
3. Prune p% of weights with lowest |W_T| magnitudes. Generate binary mask M.
4. Reset kept weights to W_0 (not W_T). Apply mask M.
5. Repeat from step 2 until target sparsity reached (iterate to 10-20% of original params).
6. At each pruning round, record: val accuracy, F1, parameter count, FLOPs estimate.

## Critical Implementation Notes
- Pruning is unstructured (individual weight level), not filter-level, for the IMP paper.
- Use `tf.Variable` masking or `tensorflow_model_optimization` (tfmot) for weight masking.
- The reset to W_0 is non-negotiable — this is what distinguishes a winning ticket from fine-tuning a smaller net.
- For VGG16 with frozen base: only prune the classification head layers (fc1, fc2, output). Pruning ImageNet pretrained conv weights requires unfreezing and is a separate experiment.
- For LeNet: prune all layers — this is the cleaner winning ticket demonstration.
- Track sparsity per layer, not just globally.

## Output Artifacts
- `results/sparsity_vs_accuracy.json` — per round metrics
- `results/winning_ticket_model.keras` — saved winning ticket model
- `results/winning_ticket_mask.npy` — binary mask
- `results/sparsity_curve.html` — Plotly figure (Nature-style)

## Experimental Questions to Answer
1. At what sparsity level does accuracy begin to degrade for LeNet vs VGG16 head?
2. Does the winning ticket subnetwork achieve the same or better val accuracy than the dense baseline?
3. How does the winning ticket's FLOPs/inference time compare to the block3_conv3+RF early-exit point from the main pipeline?
4. Can the winning ticket's feature maps still serve as useful RF inputs (plug the pruned CNN into the existing hybrid pipeline)?

## Dependencies
- `tensorflow_model_optimization` (tfmot) — add to requirements.txt if not present
- All other deps already in requirements.txt

## Do Not
- Do not modify any files outside `explorations/lottery_ticket/` and `configs/`
- Do not change `src/` modules directly — extend them via import