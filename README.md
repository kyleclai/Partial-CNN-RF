# 🔬 CNN Early-Exit with Random Forest Feature Classification

**Exploring computational efficiency through hybrid CNN-RF architectures**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📊 Key Results

Our hybrid CNN→RF approach demonstrates that **intermediate convolutional features can rival full network performance** while enabling earlier computational exit points.

### VGG16 Feature Evolution Across Layers

![VGG16 Accuracy Comparison](assets/results/accuracy_comparison_vgg16.png)

**Finding**: Random Forest achieves **competitive accuracy (85-90%)** when trained on features from VGG16's deeper conv blocks (block4/block5), approaching the full CNN baseline while enabling early-exit strategies.

### LeNet Feature Progression

![LeNet Accuracy Comparison](assets/results/accuracy_comparison_lenet.png)

---

## 🎯 Research Question

> **Can intermediate CNN feature representations replace dense layers for classification, enabling computational savings through "early exits"?**

This project evaluates:
1. **Feature Evolution**: How representational power develops across CNN layers
2. **Computational Efficiency**: Where to truncate networks without accuracy loss
3. **Interpretability**: Random Forest feature importance vs. black-box dense layers

---

## 🏗️ Architecture

```
Input Image (128×128×3)
       ↓
┌──────────────────┐
│   VGG16 Base     │
│  (Pretrained)    │
└──────────────────┘
       ↓
    [Extract at Layer N]  ← Early Exit Point
       ↓
┌──────────────────┐
│ Global Avg Pool  │  (H×W×C → C features)
└──────────────────┘
       ↓
┌──────────────────┐
│  Random Forest   │  (300 trees)
│   Classifier     │
└──────────────────┘
       ↓
   Classification
```

**Key Innovation**: Global Average Pooling (GAP) after each conv layer enables fixed-size feature extraction from any depth, making all 13 VGG16 conv layers viable exit points.

---

## 📈 Performance Benchmarks

### Test Set Results (Cats vs Dogs, 2,498 samples)

| Model | Accuracy | F1 Score | Inference Time* |
|-------|----------|----------|-----------------|
| **VGG16 Full Network** | 88.7% | 0.887 | 100% (baseline) |
| **VGG16 block5_conv3 + RF** | 87.2% | 0.869 | ~60% |
| **VGG16 block4_conv3 + RF** | 85.1% | 0.848 | ~45% |
| **Baseline RF (PCA on pixels)** | 58.3% | 0.571 | ~15% |
| **LeNet Full Network** | 63.4% | 0.625 | 20% |

*Relative inference time vs VGG16 full network

### Confusion Matrices

<table>
<tr>
<td width="50%">

**VGG16 Baseline**

![VGG16 Confusion Matrix](assets/results/confusion_matrix_vgg16.png)

</td>
<td width="50%">

**LeNet Baseline**

![LeNet Confusion Matrix](assets/results/confusion_matrix_lenet.png)

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.12 with CUDA-enabled GPU (recommended)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# Download Cats vs Dogs dataset from Kaggle
# Place in data/dogs-vs-cats/train/

# Run complete workflow (preprocess → train → extract → evaluate)
python src/run_pipeline.py --config configs/full_vgg16_gpu.yaml
```

### Run Individual Stages

```bash
# 1. Preprocess data (train/val/test split)
python src/preprocess.py --config configs/full_vgg16_gpu.yaml

# 2. Train CNN baseline
python src/train_cnn.py --config configs/full_vgg16_gpu.yaml

# 3. Extract features from all conv layers (with GAP)
python src/extract_features.py --config configs/full_vgg16_gpu.yaml

# 4. Train Random Forest on extracted features
python src/train_rf.py --config configs/full_vgg16_gpu.yaml

# 5. Evaluate all models on test set
python src/evaluate.py --config configs/full_vgg16_gpu.yaml
```

---

## 📁 Repository Structure

```
.
├── src/
│   ├── utils/
│   │   ├── model_builders.py    # VGG16/LeNet architectures
│   │   ├── pooling.py           # Global Average Pooling utilities
│   │   ├── data_loaders.py      # Efficient data streaming
│   │   └── seeds.py             # Reproducibility utilities
│   ├── preprocess.py            # Dataset splitting
│   ├── train_cnn.py             # CNN baseline training
│   ├── extract_features.py      # Layer-wise feature extraction
│   ├── train_rf.py              # Random Forest training
│   ├── evaluate.py              # Model comparison & metrics
│   └── run_pipeline.py          # Orchestration script
├── configs/
│   ├── full_vgg16_gpu.yaml      # VGG16 configuration (GPU)
│   └── demo_lenet_cpu.yaml      # LeNet configuration (CPU demo)
├── assets/
│   └── results/                 # Generated plots & visualizations
├── artifacts/                   # Trained models & metrics (gitignored)
└── requirements.txt
```

---

## 🔬 Methodology

### 1. Feature Extraction Strategy

**Traditional Approach** (problematic):
- Extract raw spatial features: `block1_conv2` → 128×128×64 = **1M features per image**
- Causes memory overflow, impractical for RF

**Our Approach** (Global Average Pooling):
- Apply GAP after each layer: `block1_conv2` → GAP → **64 features**
- Reduces dimensionality while preserving channel-wise information
- Enables extraction from all 13 VGG16 conv layers

### 2. Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=300,      # 300 decision trees
    max_depth=None,        # Unlimited depth
    random_state=42,
    n_jobs=-1              # Parallel training
)
```

### 3. Baseline Comparisons

- **CNN Baseline**: Full VGG16/LeNet with dense layers
- **RF Baseline**: PCA (200 components) on raw pixels → RF
- **Hybrid Models**: CNN features (with GAP) → RF

---

## 🧪 Experimental Insights

### Finding 1: Feature Maturity Across Depth

Accuracy increases monotonically from block1 → block5, indicating **progressive feature abstraction**:
- `block1_conv2` (64 filters): 72% accuracy
- `block3_conv3` (256 filters): 82% accuracy  
- `block5_conv3` (512 filters): 87% accuracy

### Finding 2: Optimal Early-Exit Point

**block5_conv3** offers the best accuracy/efficiency trade-off:
- Only **1.5% accuracy drop** vs full network
- **~40% reduction** in FLOPs (skips dense layers)
- Enables SHAP/LIME interpretability via RF

### Finding 3: GAP is Essential

Without GAP, early layer extraction is infeasible:
- `block1_conv2` raw: **1M features** → OOM
- `block1_conv2` + GAP: **64 features** → Manageable

---

## 💡 Future Directions

### Interpretability Analysis
- [ ] Apply SHAP to RF models to visualize which CNN features drive classification
- [ ] Compare feature importance across different extraction depths

### Embedded Deployment
- [ ] Model quantization (INT8) for edge devices
- [ ] Benchmark inference on Raspberry Pi / Jetson Nano
- [ ] Adaptive early-exit based on confidence thresholds

### Bioinformatics Application
- [ ] Replace image data with tabular gene expression data
- [ ] Test hypothesis: CNN feature extraction + RF classification for high-dimensional bio-data

---

## 📊 Dataset

**Cats vs Dogs** (Kaggle)
- 25,000 labeled images (12,500 cats, 12,500 dogs)
- Train/Val/Test: 70/20/10 split
- Preprocessing: Resize to 128×128, normalize to [0,1]

---

## 🛠️ Technical Details

### GPU Memory Management
- Uses TensorFlow's `tf.data` streaming to avoid loading full dataset into RAM
- Batch size: 32 for VGG16, 8 for early layer extraction
- Global Average Pooling reduces memory footprint by 1000x

### Reproducibility
- Fixed random seeds: Python, NumPy, TensorFlow
- Deterministic operations enabled
- All configs version-controlled

### Performance Optimizations
- Multi-threaded data loading (`tf.data.AUTOTUNE`)
- Compiled models for faster inference
- Incremental feature extraction to prevent OOM

---

## 📚 References

This work builds on:

1. **Bacterial Spore Segmentation**: Qamar et al. (2023) - Hybrid CNN-RF for TEM images
2. **Burned Area Detection**: Sudiana et al. (2023) - CNN-RF for SAR data fusion  
3. **CAD Diagnosis**: Khozeimeh et al. (2022) - RF-CNN-F for medical imaging

See `Final_Research_Report.pdf` for full literature review.

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional CNN architectures (ResNet, EfficientNet)
- Alternative ensemble methods (XGBoost, LightGBM)
- Deployment optimizations
- Interpretability visualizations

---

## 📄 License

MIT License - see LICENSE file for details

---

## 👥 Authors

**Kyle Lai** & **Ruslan Romanenko**  
*CSS 499 Capstone Research Project*  
*University of Washington Bothell*

---

## 🙏 Acknowledgments

- Dr. Kim (Research Advisor)
- UW Bothell CSSBIO Lab (GPU compute resources: 4× NVIDIA RTX A5000)
- Anthropic Claude (Technical assistance)

---

## 📬 Contact

Questions? Open an issue or reach out via [GitHub](https://github.com/kyleclai)

---

<p align="center">
  <i>Exploring the intersection of deep learning and interpretable machine learning</i>
</p>
