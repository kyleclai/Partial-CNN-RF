# 🔬 CNN Early-Exit with Random Forest Feature Classification

**Exploring computational efficiency through hybrid CNN-RF architectures**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## TL;DR

This project explores a compute-aware alternative to full CNN inference: train a Random Forest on intermediate CNN feature maps (“early-exit”) and compare performance against (1) RF baseline and (2) full CNN baseline. The workflow is orchestrated as an Apache Airflow DAG to ensure reproducible, modular experiments.
- Orchestration: [Apache Airflow](https://github.com/apache/airflow)
- Modeling: TensorFlow/Keras (CNN), scikit-learn (Random Forest)
- Experiment design: layer-wise feature extraction at configurable cut points
- Dataset: Public dataset ([Cats vs Dogs] or your chosen public source)

**[ToDo: Make VGG16 CSV akin to InceptionV3 Inference CSV](https://docs.google.com/spreadsheets/d/1quvbLjKlESu--7Vh5U4s5ZChmjYwV-1sWoylQEz4egU/edit?pli=1&gid=1121146955#gid=1121146955)**

## Why this exists

In applied settings (edge devices, embedded vision, constrained environments), full CNN inference may be too slow or too costly. This repo tests the hypothesis:
> Can intermediate CNN representations serve as a strong feature extractor for a cheaper model (Random Forest), and what accuracy trade-offs appear at different cut layers?

This is a proof-of-concept experiment harness — not a production deployment.

## 📊 Key Results

> Our hybrid CNN→RF approach demonstrates that **intermediate convolutional features can rival full network performance** while enabling earlier computational exit points.

### VGG16 Feature Evolution Across Layers

![VGG16 Accuracy Comparison](assets/results/accuracy_comparison_vgg16.png)

**Finding**: Random Forest achieves **competitive accuracy (75-90%)** when trained on features from VGG16's deeper conv blocks (block3/block4/block5), approaching the full CNN baseline while enabling early-exit strategies.
* *Hybrid CNN→RF (best intermediate cut): block3_conv3 represents the strongest true early-exit point, retaining ~78% accuracy at ~60% of VGG16 compute, while deeper exits primarily trade efficiency gains for interpretability rather than speed.

Interpretation: Intermediate CNN features can improve over a simple RF baseline, but still trail full CNN performance on this dataset. This repo is structured to extend experiments across architectures and cut points.

### LeNet Feature Progression

![LeNet Accuracy Comparison](assets/results/accuracy_comparison_lenet.png)

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

## What the pipeline does

The Airflow DAG runs the experiment end-to-end:
1. Ingest: download and cache a public dataset
2. Preprocess: resize/normalize, train/val split
3. Train CNN baseline: train a small CNN (e.g., LeNet-style)
4. Extract features: cut the CNN at a chosen layer and export embeddings
5. Train RF on embeddings: fit RF on extracted feature vectors
6. Evaluate + report: compare baselines and write a report artifact
Artifacts are saved per run (models, metrics JSON, plots).

---

## 📈 Performance Benchmarks

### Test Set Results (Cats vs Dogs, 2,498 samples)

**Assumption:** VGG16 input size is **128×128**, with standard VGG16 pooling (128→64→32→16→8→4).<br>
**Inference Time*** is **cumulative VGG16 compute up to the exit layer** (conv MACs) divided by **full VGG16 compute** (conv + dense head), expressed as a percent.

| Model                       | Accuracy | F1 Score | Inference Time* |
| --------------------------- | -------- | -------- | --------------- |
| **VGG16 Full Network**      | 90.31%   | 0.9033   | 100% (baseline) |
| **VGG16 block5_conv3 + RF** | 86.47%   | 0.8624   | ~99%            |
| **VGG16 block5_conv2 + RF** | 85.71%   | 0.8466   | ~96%            |
| **VGG16 block5_conv1 + RF** | 86.23%   | 0.8592   | ~93%            |
| **VGG16 block4_conv3 + RF** | 85.35%   | 0.8437   | ~90%            |
| **VGG16 block4_conv2 + RF** | 83.23%   | 0.8124   | ~78%            |
| **VGG16 block4_conv1 + RF** | 80.62%   | 0.7876   | ~66%            |
| **VGG16 block3_conv3 + RF** | 78.30%   | 0.7739   | ~60%            |
| **VGG16 block3_conv2 + RF** | 76.66%   | 0.7501   | ~48%            |
| **VGG16 block3_conv1 + RF** | 75.50%   | 0.7316   | ~36%            |
| **VGG16 block2_conv2 + RF** | 71.34%   | 0.7064   | ~30%            |
| **VGG16 block2_conv1 + RF** | 66.49%   | 0.6667   | ~18%            |
| **VGG16 block1_conv2 + RF** | 64.93%   | 0.6393   | ~12%            |
| **Baseline RF (PCA on pixels)** | 63.73% | 0.632 | ~15% |
| **VGG16 block1_conv1 + RF** | 61.89%   | 0.6052   | ~1%             |

* *Compute proxy (cumulative MACs/FLOPs), normalized to VGG16 full network.
* *Relative inference time (compute proxy) vs VGG16 full network (100%).

<!--
VGG16 Full Network: 180.93198442459106
-->

<!--
| Model | Accuracy | F1 Score | Inference Time* |
|-------|----------|----------|-----------------|
| **VGG16 Full Network** | 88.7% | 0.887 | 100% (baseline) |
| **VGG16 block5_conv3 + RF** | 87.2% | 0.869 | ~60% |
| **VGG16 block4_conv3 + RF** | 85.1% | 0.848 | ~45% |
| **Baseline RF (PCA on pixels)** | 58.3% | 0.571 | ~15% |
| **LeNet Full Network** | 63.4% | 0.625 | 20% |

*Relative inference time vs VGG16 full network
-->

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

## File Structure
```.
├── dags/                    # Airflow DAG definition
├── data/                    # Datasets
│   ├── dogs-vs-cats/        # Dummy data
│       ├── train/
├── src/                     # Pure Python modules (called by Airflow tasks)
│   ├── utils/
│       ├── __pycache__/
│       ├── data_loaders.py
│       ├── model_builders.py
│       ├── pooling.py
│       └── seeds.py
│   ├── run_pipeline.py
│   ├── preprocess.py
│   ├── train_cnn.py
│   ├── extract_features.py
│   ├── train_rf.py
│   └── evaluate.py
├── configs/                 # YAML/JSON configs for models + run params
│   ├── demo_lenet_cpu.yaml
│   ├── full_vgg16_gpu.yaml
│   └── full_vgg16_gpu_gap.yaml
├── reports/                 # Generated reports (optional)
├── artifacts/               # Models/metrics/plots (gitignored)
├── requirements.txt
└── README.md
```

## How to run

### Option A — Run via Airflow (recommended)

1) Create env
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2) Install Airflow
(Use the official constraints file matching your Airflow + Python versions.) [Airflow 2.9.1 - Python 3.12](https://raw.githubusercontent.com/apache/airflow/constraints-2.9.1/constraints-3.12.txt)
3) Start Airflow
```
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow users create --username admin --firstname Admin --lastname User --role Admin --email you@example.com
airflow webserver -p 8080
airflow scheduler
```
4) Trigger the DAG
Open http://localhost:8080
Enable cnn_rf_feature_pipeline
Trigger manually (optionally set params)

### Option B — Run without Airflow (for quick checks)
```
python src/run_pipeline.py
python src/preprocess.py
python src/train_cnn.py --model lenet
python src/extract_features.py --cut_layer conv2
python src/train_rf.py
python src/evaluate.py
```

## Configuration
Experiments are controlled via config/params:
- `MODEL_ARCH`: `lenet | vgg16 | inceptionv3`
- `CUT_LAYER`: layer name or index
- `EPOCHS`, `BATCH_SIZE`, `IMG_SIZE`
- `SEED` for reproducibility

## What to look at (if you’re reviewing quickly)
- Airflow DAG: dags/cnn_rf_feature_pipeline.py
- Feature extraction logic: src/extract_features.py
- Evaluation + report artifacts: src/evaluate.py + artifacts/

## Notes on reproducibility
- Uses fixed random seeds where possible.
- Writes metrics and artifacts per run ID.
- Dataset is public; no clinical/private data is included.

## Roadmap (optional, keep short)
- Add support for additional architectures and standardized cut-layer selection
- Add compute measurement (time per stage, GPU/CPU utilization)
- Add experiment tracking integration (MLflow / W&B) (optional)

## Installation
TODO: ADD INFO

## Contributing
Fork it!<br>
Create your feature branch: git checkout -b my-new-feature<br>
Commit your changes: git commit -am 'Add some feature'<br>
Push to the branch: git push origin my-new-feature<br>
Submit a pull request :D

## History
Version 0.1 (2026-01-23) - adding dataset and processing functionalities

## Credits
Lead Developer - Kyle Lai (@kyleclai)

## License
The MIT License (MIT)

Copyright (c) 2026 Kyle Lai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

https://www.star-history.com/#kyleclai/Partial-CNN-RF&type=date&legend=top-left
