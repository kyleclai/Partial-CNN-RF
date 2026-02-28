# Lottery Ticket Exploration — Notes

## Session Summary (2026-02-28)
Implemented the full IMP (Iterative Magnitude Pruning) exploration from scratch.
Local Mac testing revealed the dataset only has 6 images — real experiments must
run on the Linux GPU lab via SSH.

---

## Files Created

| File | Purpose |
|------|---------|
| `mask_utils.py` | `MaskManager` + `MaskEnforcementCallback` Keras callback |
| `imp_trainer.py` | `IMPTrainer` class + CLI entry point; main IMP loop |
| `winning_ticket_extractor.py` | Extract/verify winning subnetwork; hybrid pipeline plug-in |
| `compare_metrics.py` | Plotly visualizations: sparsity curve, param count, IMP vs hybrid |
| `prepare_data.py` | Robust `metadata.csv` generator (handles any dataset size) |
| `configs/imp_lenet.yaml` | LeNet config: CPU, 2K samples, all-layer pruning |
| `configs/imp_vgg16.yaml` | VGG16 config: GPU, 25K samples, head-only pruning |

Results go to: `explorations/lottery_ticket/results/{lenet,vgg16}/`

---

## Next Step: Run on Linux GPU Lab

### SSH Workflow
```bash
# 1. SSH into the lab machine
ssh <username>@<cssbio-lab-hostname>

# 2. Navigate to project
cd ~/path/to/CSS\ Research/CNN\ Model/Partial-CNN-RF

# 3. Activate conda environment
conda activate <env-name>    # check with: conda env list

# 4. Pull latest code (make sure to push from Mac first)
git pull

# 5. Run LeNet first (faster sanity check)
python explorations/lottery_ticket/imp_trainer.py \
  --config explorations/lottery_ticket/configs/imp_lenet.yaml

# 6. Then run VGG16
python explorations/lottery_ticket/imp_trainer.py \
  --config explorations/lottery_ticket/configs/imp_vgg16.yaml
```

### After Training
```bash
# Extract and verify winning ticket
python explorations/lottery_ticket/winning_ticket_extractor.py \
  --results explorations/lottery_ticket/results/lenet --verify

# Generate plots (lenet)
python explorations/lottery_ticket/compare_metrics.py \
  --results explorations/lottery_ticket/results/lenet/sparsity_vs_accuracy.json \
  --arch lenet \
  --output-dir explorations/lottery_ticket/results/lenet

# Generate plots (vgg16, with hybrid pipeline comparison)
python explorations/lottery_ticket/compare_metrics.py \
  --results explorations/lottery_ticket/results/vgg16/sparsity_vs_accuracy.json \
  --arch vgg16 \
  --output-dir explorations/lottery_ticket/results/vgg16 \
  --hybrid-metrics artifacts/full_vgg16_gpu/test_metrics.json
```

### Checklist Before Running
- [ ] `git pull` succeeded and `explorations/lottery_ticket/` files are present
- [ ] `data/metadata.csv` exists (auto-generated on first run if missing)
- [ ] plotly installed: `pip show plotly` (if not: `pip install plotly`)
- [ ] GPU visible: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

---

## Experimental Questions to Answer
1. At what sparsity does accuracy degrade for LeNet vs VGG16 head?
2. Does the winning ticket match or beat the dense baseline?
3. How does winning ticket FLOPs compare to block3_conv3+RF (~60% of full VGG16)?
4. Can pruned CNN features still feed a useful RF? (use `--plug-into-hybrid` flag)
