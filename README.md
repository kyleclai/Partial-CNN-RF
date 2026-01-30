# Project: Early-Exit CNN → Random Forest Pipeline (Airflow-Orchestrated)

## TL;DR

This project explores a compute-aware alternative to full CNN inference: train a Random Forest on intermediate CNN feature maps (“early-exit”) and compare performance against (1) RF baseline and (2) full CNN baseline. The workflow is orchestrated as an Apache Airflow DAG to ensure reproducible, modular experiments.
- Orchestration: [Apache Airflow](https://github.com/apache/airflow)
- Modeling: TensorFlow/Keras (CNN), scikit-learn (Random Forest)
- Experiment design: layer-wise feature extraction at configurable cut points
- Dataset: Public dataset ([Cats vs Dogs] or your chosen public source)

## Why this exists

In applied settings (edge devices, embedded vision, constrained environments), full CNN inference may be too slow or too costly. This repo tests the hypothesis:
> Can intermediate CNN representations serve as a strong feature extractor for a cheaper model (Random Forest), and what accuracy trade-offs appear at different cut layers?

This is a proof-of-concept experiment harness — not a production deployment.

## What the pipeline does

The Airflow DAG runs the experiment end-to-end:
1. Ingest: download and cache a public dataset
2. Preprocess: resize/normalize, train/val split
3. Train CNN baseline: train a small CNN (e.g., LeNet-style)
4. Extract features: cut the CNN at a chosen layer and export embeddings
5. Train RF on embeddings: fit RF on extracted feature vectors
6. Evaluate + report: compare baselines and write a report artifact
Artifacts are saved per run (models, metrics JSON, plots).

## Results (example)

> Replace with your real numbers for the public dataset + chosen architecture.
- RF baseline (PCA on raw images): ~63% validation accuracy
- CNN baseline (LeNet-style): ~81% validation accuracy
- Hybrid CNN→RF (best intermediate cut): ~71% validation accuracy
Interpretation: Intermediate CNN features can improve over a simple RF baseline, but still trail full CNN performance on this dataset. This repo is structured to extend experiments across architectures and cut points.

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
python src/ingest.py
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
Fork it!
Create your feature branch: git checkout -b my-new-feature
Commit your changes: git commit -am 'Add some feature'
Push to the branch: git push origin my-new-feature
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
