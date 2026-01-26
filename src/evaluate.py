"""
Evaluate all models on test set and generate comparison report.
"""

import argparse
import yaml
import json
import joblib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix)
import tensorflow as tf
from utils.seeds import set_global_seed, configure_gpu


def evaluate_cnn(model_path, x_test, y_test, batch_size=32):
    """Evaluate trained CNN model."""
    print(f"\nEvaluating CNN: {model_path.name}")
    model = tf.keras.models.load_model(model_path)
    
    y_pred_proba = model.predict(x_test, batch_size=batch_size, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        'model_type': 'cnn_baseline',
        'model_name': model_path.stem,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    return metrics, y_pred


def evaluate_rf(model_path, x_test, y_test):
    """Evaluate trained RF model."""
    print(f"\nEvaluating RF: {model_path.name}")
    model = joblib.load(model_path)
    
    y_pred = model.predict(x_test)
    
    metrics = {
        'model_type': 'rf',
        'model_name': model_path.stem,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    return metrics, y_pred


def plot_accuracy_comparison(all_metrics, output_path):
    """Generate accuracy comparison bar chart."""
    # Sort by accuracy
    sorted_metrics = sorted(all_metrics, key=lambda x: x['accuracy'], reverse=True)
    
    names = [m['model_name'] for m in sorted_metrics]
    accuracies = [m['accuracy'] for m in sorted_metrics]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(names, accuracies, color='steelblue')
    
    # Color code by model type
    for i, m in enumerate(sorted_metrics):
        if m['model_type'] == 'cnn_baseline':
            bars[i].set_color('darkgreen')
        elif 'baseline' in m['model_name']:
            bars[i].set_color('coral')
    
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('Model Accuracy Comparison (Test Set)', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Add value labels
    for i, v in enumerate(accuracies):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy comparison saved to {output_path}")


def plot_confusion_matrix(y_true, y_pred, output_path, title='Confusion Matrix'):
    """Generate confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Cat', 'Dog'], 
                yticklabels=['Cat', 'Dog'],
                ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def generate_markdown_report(all_metrics, output_path):
    """Generate markdown summary report."""
    sorted_metrics = sorted(all_metrics, key=lambda x: x['accuracy'], reverse=True)
    
    report = "# Model Evaluation Report\n\n"
    report += "## Test Set Performance\n\n"
    report += "| Rank | Model | Accuracy | Precision | Recall | F1 Score |\n"
    report += "|------|-------|----------|-----------|--------|----------|\n"
    
    for i, m in enumerate(sorted_metrics, 1):
        report += f"| {i} | {m['model_name']} | {m['accuracy']:.4f} | "
        report += f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} |\n"
    
    report += "\n## Key Findings\n\n"
    
    best_model = sorted_metrics[0]
    report += f"- **Best model**: {best_model['model_name']} "
    report += f"(Accuracy: {best_model['accuracy']:.4f})\n"
    
    cnn_models = [m for m in sorted_metrics if m['model_type'] == 'cnn_baseline']
    if cnn_models:
        report += f"- **CNN baseline**: {cnn_models[0]['model_name']} "
        report += f"(Accuracy: {cnn_models[0]['accuracy']:.4f})\n"
    
    baseline_rf = [m for m in sorted_metrics if 'baseline_rf' in m['model_name']]
    if baseline_rf:
        report += f"- **Baseline RF (PCA)**: {baseline_rf[0]['model_name']} "
        report += f"(Accuracy: {baseline_rf[0]['accuracy']:.4f})\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\nMarkdown report saved to {output_path}")


def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    set_global_seed(config['run']['seed'])
    configure_gpu(config['run']['device'])
    
    # Paths
    run_name = config['run']['name']
    artifacts_dir = Path(config['run']['artifacts_dir']) / run_name
    models_dir = artifacts_dir / 'models'
    features_dir = artifacts_dir / 'features'
    plots_dir = artifacts_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)
    
    all_metrics = []
    
    # Evaluate CNN baseline
    arch = config['train']['arch']
    cnn_model_path = models_dir / f'{arch}_baseline.keras'
    
    if cnn_model_path.exists():
        from utils.data_loaders import load_split_as_numpy
        metadata_path = Path(config['data']['data_dir']) / 'metadata.csv'
        img_size = tuple(config['data']['image_size'])
        
        print("\nLoading test images...")
        x_test, y_test = load_split_as_numpy(
            metadata_path, split='test', img_size=img_size,
            random_state=config['run']['seed']
        )
        
        cnn_metrics, cnn_pred = evaluate_cnn(
            cnn_model_path, x_test, y_test, 
            batch_size=config['train']['batch_size']
        )
        all_metrics.append(cnn_metrics)
        
        # CNN confusion matrix
        plot_confusion_matrix(
            y_test, cnn_pred, 
            plots_dir / f'{arch}_confusion_matrix.png',
            title=f'{arch.upper()} Baseline Confusion Matrix'
        )
    
    # Evaluate baseline RF
    baseline_rf_path = models_dir / 'baseline_rf.pkl'
    baseline_pca_path = models_dir / 'baseline_pca.pkl'
    
    if baseline_rf_path.exists() and baseline_pca_path.exists():
        print("\nPreparing baseline RF test data...")
        pca = joblib.load(baseline_pca_path)
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        x_test_pca = pca.transform(x_test_flat)
        
        baseline_metrics, baseline_pred = evaluate_rf(
            baseline_rf_path, x_test_pca, y_test
        )
        all_metrics.append(baseline_metrics)
        
        # Baseline RF confusion matrix
        plot_confusion_matrix(
            y_test, baseline_pred,
            plots_dir / 'baseline_rf_confusion_matrix.png',
            title='Baseline RF (PCA) Confusion Matrix'
        )
    
    # Evaluate hybrid RF models
    hybrid_rf_files = sorted(models_dir.glob('hybrid_rf_*.pkl'))
    
    for rf_path in hybrid_rf_files:
        layer_name = rf_path.stem.replace('hybrid_rf_', '')
        
        # Load corresponding test features
        test_features_path = features_dir / f'test_{layer_name}_features.npz'
        
        if test_features_path.exists():
            test_data = np.load(test_features_path)
            x_test_features = test_data['features']
            y_test_labels = test_data['labels']
            
            metrics, pred = evaluate_rf(rf_path, x_test_features, y_test_labels)
            all_metrics.append(metrics)
    
    # Save all metrics
    with open(artifacts_dir / 'test_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Generate plots
    plot_accuracy_comparison(all_metrics, plots_dir / 'accuracy_comparison.png')
    
    # Generate report
    if config.get('report', {}).get('save_markdown', False):
        generate_markdown_report(all_metrics, artifacts_dir / 'evaluation_report.md')
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to {artifacts_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate all models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML')
    args = parser.parse_args()
    
    main(args.config)