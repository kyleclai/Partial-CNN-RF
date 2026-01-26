"""
Single-command pipeline runner (no Airflow required for demo).
Executes the full workflow: preprocess -> train CNN -> extract features -> train RF -> evaluate.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_stage(script_name, config_path):
    """Execute a pipeline stage."""
    print(f"\n{'='*70}")
    print(f"STAGE: {script_name}")
    print('='*70)
    
    cmd = [sys.executable, f'src/{script_name}', '--config', config_path]
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"\n❌ Stage '{script_name}' failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"✅ Stage '{script_name}' completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Run the full CNN-RF pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python src/run_pipeline.py --config configs/demo_lenet_cpu.yaml
        """
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML')
    parser.add_argument('--skip-preprocess', action='store_true',
                       help='Skip preprocessing if metadata already exists')
    parser.add_argument('--skip-cnn', action='store_true',
                       help='Skip CNN training if model already exists')
    parser.add_argument('--skip-extract', action='store_true',
                       help='Skip feature extraction if features already exist')
    
    args = parser.parse_args()
    
    config_path = args.config
    
    print("\n" + "="*70)
    print("STARTING CNN-RF PIPELINE")
    print("="*70)
    print(f"Config: {config_path}\n")
    
    # Stage 1: Preprocess
    if not args.skip_preprocess:
        run_stage('preprocess.py', config_path)
    else:
        print("⏭️  Skipping preprocess stage")
    
    # Stage 2: Train CNN
    if not args.skip_cnn:
        run_stage('train_cnn.py', config_path)
    else:
        print("⏭️  Skipping CNN training stage")
    
    # Stage 3: Extract features
    if not args.skip_extract:
        run_stage('extract_features.py', config_path)
    else:
        print("⏭️  Skipping feature extraction stage")
    
    # Stage 4: Train RF
    run_stage('train_rf.py', config_path)
    
    # Stage 5: Evaluate
    run_stage('evaluate.py', config_path)
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print(f"Check artifacts directory for results\n")


if __name__ == '__main__':
    main()