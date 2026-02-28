"""
compare_metrics.py — Plotly visualizations for IMP sparsity experiments.

Generates:
  - sparsity_curve.html          — sparsity % vs val_accuracy + val_F1 + inference time
  - sparsity_vs_params.html      — active param count (log x) vs accuracy
  - imp_vs_hybrid_pipeline.html  — IMP vs block3_conv3+RF on accuracy vs FLOPs

Usage:
    python compare_metrics.py \\
        --results explorations/lottery_ticket/results/lenet/sparsity_vs_accuracy.json \\
        --arch lenet \\
        --output-dir explorations/lottery_ticket/results/lenet

    # With hybrid pipeline comparison:
    python compare_metrics.py \\
        --results explorations/lottery_ticket/results/vgg16/sparsity_vs_accuracy.json \\
        --arch vgg16 \\
        --output-dir explorations/lottery_ticket/results/vgg16 \\
        --hybrid-metrics artifacts/full_vgg16_gpu/test_metrics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Install with: pip install plotly")


# ---------------------------------------------------------------------------
# Nature-style colour palette
# ---------------------------------------------------------------------------
_RED    = '#E64B35'
_CYAN   = '#4DBBD5'
_NAVY   = '#3C5488'
_GREEN  = '#00A087'
_ORANGE = '#F39B7F'

_FONT_FAMILY = 'Arial, Helvetica, sans-serif'
_FONT_SIZE   = 14


def _nature_layout(title: str, xaxis_title: str, yaxis_title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16, family=_FONT_FAMILY)),
        xaxis=dict(
            title=xaxis_title,
            tickfont=dict(size=_FONT_SIZE, family=_FONT_FAMILY),
            showgrid=True, gridcolor='#E5E5E5',
            linecolor='#333333', linewidth=1,
        ),
        yaxis=dict(
            title=yaxis_title,
            tickfont=dict(size=_FONT_SIZE, family=_FONT_FAMILY),
            showgrid=True, gridcolor='#E5E5E5',
            linecolor='#333333', linewidth=1,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            font=dict(size=_FONT_SIZE - 1, family=_FONT_FAMILY),
            bordercolor='#CCCCCC', borderwidth=1,
        ),
        margin=dict(l=80, r=60, t=80, b=60),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str | Path) -> list[dict]:
    """Load sparsity_vs_accuracy.json produced by IMPTrainer."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1: Sparsity curve
# ---------------------------------------------------------------------------

def plot_sparsity_curve(
    results: list[dict],
    output_path: str | Path,
    arch: str,
) -> None:
    """Sparsity % (x) vs val_accuracy + val_F1 (left y) + inference time (right y).

    - Dashed horizontal lines at round-0 dense baseline.
    - Winning ticket round annotated with a star marker.
    - Nature-style colors.

    Args:
        results:     List of round dicts from sparsity_vs_accuracy.json.
        output_path: Output .html file path.
        arch:        Architecture name for title.
    """
    if not _PLOTLY_AVAILABLE:
        print("Skipping plot (plotly not available).")
        return

    sparsity  = [r['global_sparsity'] * 100 for r in results]
    val_acc   = [r['val_accuracy']          for r in results]
    val_f1    = [r['val_f1']                for r in results]
    inf_time  = [r['inference_ms_per_sample'] for r in results]
    rounds    = [r['round']                  for r in results]

    # Round 0 baselines
    baseline_acc = val_acc[0]
    baseline_f1  = val_f1[0]

    # Identify winning ticket (best val_acc among rounds >= 1)
    pruned_idx = [i for i, r in enumerate(results) if r['round'] >= 1]
    if pruned_idx:
        win_i = max(pruned_idx, key=lambda i: val_acc[i])
    else:
        win_i = 0

    fig = make_subplots(specs=[[{'secondary_y': True}]])

    # Accuracy line
    fig.add_trace(go.Scatter(
        x=sparsity, y=val_acc,
        mode='lines+markers',
        name='Val Accuracy',
        line=dict(color=_RED, width=2),
        marker=dict(size=7),
    ), secondary_y=False)

    # F1 line
    fig.add_trace(go.Scatter(
        x=sparsity, y=val_f1,
        mode='lines+markers',
        name='Val F1',
        line=dict(color=_CYAN, width=2),
        marker=dict(size=7),
    ), secondary_y=False)

    # Inference time line (right axis)
    fig.add_trace(go.Scatter(
        x=sparsity, y=inf_time,
        mode='lines+markers',
        name='Inference (ms/sample)',
        line=dict(color=_NAVY, width=2, dash='dot'),
        marker=dict(size=7),
    ), secondary_y=True)

    # Winning ticket star
    if win_i < len(sparsity):
        fig.add_trace(go.Scatter(
            x=[sparsity[win_i]],
            y=[val_acc[win_i]],
            mode='markers',
            name=f'Winning Ticket (R{rounds[win_i]})',
            marker=dict(symbol='star', size=16, color=_ORANGE, line=dict(width=1, color='black')),
        ), secondary_y=False)

    # Baseline dashed lines
    fig.add_hline(
        y=baseline_acc, line_dash='dash', line_color=_RED,
        annotation_text=f'Dense baseline acc={baseline_acc:.4f}',
        annotation_position='bottom right',
    )
    fig.add_hline(
        y=baseline_f1, line_dash='dash', line_color=_CYAN,
        annotation_text=f'Dense baseline F1={baseline_f1:.4f}',
        annotation_position='top right',
    )

    fig.update_layout(
        **_nature_layout(
            title=f'IMP Sparsity Curve — {arch.upper()}',
            xaxis_title='Global Sparsity (%)',
            yaxis_title='Val Accuracy / Val F1',
        )
    )
    fig.update_yaxes(title_text='Inference Time (ms/sample)', secondary_y=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Sparsity vs param count
# ---------------------------------------------------------------------------

def plot_sparsity_vs_param_count(
    results: list[dict],
    output_path: str | Path,
    arch: str,
) -> None:
    """Active param count (log x) vs val_accuracy; annotate winning ticket.

    Args:
        results:     List of round dicts.
        output_path: Output .html path.
        arch:        Architecture name for title.
    """
    if not _PLOTLY_AVAILABLE:
        return

    active_params = [r['active_params']  for r in results]
    val_acc       = [r['val_accuracy']    for r in results]
    rounds        = [r['round']           for r in results]
    sparsities    = [r['global_sparsity'] for r in results]

    pruned_idx = [i for i, r in enumerate(results) if r['round'] >= 1]
    win_i = max(pruned_idx, key=lambda i: val_acc[i]) if pruned_idx else 0

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=active_params, y=val_acc,
        mode='lines+markers',
        name='Val Accuracy',
        line=dict(color=_NAVY, width=2),
        marker=dict(
            size=9,
            color=sparsities,
            colorscale='RdBu',
            colorbar=dict(title='Sparsity', thickness=12),
            showscale=True,
        ),
        text=[f'Round {r}<br>Sparsity {s:.1%}' for r, s in zip(rounds, sparsities)],
        hovertemplate='%{text}<br>Params: %{x:,}<br>Acc: %{y:.4f}<extra></extra>',
    ))

    # Winning ticket annotation
    fig.add_trace(go.Scatter(
        x=[active_params[win_i]],
        y=[val_acc[win_i]],
        mode='markers+text',
        name=f'Winning Ticket',
        marker=dict(symbol='star', size=18, color=_ORANGE, line=dict(width=1, color='black')),
        text=[f'R{rounds[win_i]}'],
        textposition='top center',
    ))

    fig.update_layout(
        **_nature_layout(
            title=f'Active Params vs Val Accuracy — {arch.upper()}',
            xaxis_title='Active Parameters (log scale)',
            yaxis_title='Val Accuracy',
        ),
        xaxis_type='log',
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: IMP vs hybrid pipeline
# ---------------------------------------------------------------------------

def compare_with_hybrid_pipeline(
    imp_results: list[dict],
    hybrid_metrics_path: str | Path,
    output_path: str | Path,
) -> None:
    """Compare IMP winning ticket vs block3_conv3+RF on accuracy vs FLOPs.

    Args:
        imp_results:          List of IMP round dicts.
        hybrid_metrics_path:  Path to artifacts/{run}/test_metrics.json from
                              main pipeline.
        output_path:          Output .html path.
    """
    if not _PLOTLY_AVAILABLE:
        return

    hybrid_path = Path(hybrid_metrics_path)
    if not hybrid_path.exists():
        print(f"Hybrid metrics not found: {hybrid_path}. Skipping comparison plot.")
        return

    with open(hybrid_path) as f:
        hybrid_metrics = json.load(f)

    fig = go.Figure()

    # IMP curve: accuracy vs FLOPs across rounds
    imp_flops = [r['estimated_flops'] for r in imp_results]
    imp_acc   = [r['val_accuracy']    for r in imp_results]
    imp_rounds = [r['round']          for r in imp_results]

    fig.add_trace(go.Scatter(
        x=imp_flops, y=imp_acc,
        mode='lines+markers',
        name='IMP Subnetwork (per round)',
        line=dict(color=_RED, width=2),
        marker=dict(size=8),
        text=[f'Round {r}' for r in imp_rounds],
        hovertemplate='%{text}<br>FLOPs: %{x:.2e}<br>Acc: %{y:.4f}<extra></extra>',
    ))

    # Hybrid pipeline: extract block3_conv3+RF if present
    _plot_hybrid_points(fig, hybrid_metrics)

    fig.update_layout(
        **_nature_layout(
            title='IMP vs Hybrid CNN-RF: Accuracy vs FLOPs',
            xaxis_title='Estimated FLOPs (log scale)',
            yaxis_title='Val / Test Accuracy',
        ),
        xaxis_type='log',
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


def _plot_hybrid_points(fig, hybrid_metrics: dict) -> None:
    """Add hybrid pipeline points to an existing figure."""
    # The hybrid metrics JSON may contain per-layer results.
    # Try to extract block3_conv3+RF and full VGG16 if present.
    layer_results = hybrid_metrics.get('hybrid_rf', {})
    for layer_name, info in layer_results.items():
        acc   = info.get('accuracy') or info.get('val_accuracy')
        flops = info.get('estimated_flops')
        if acc is None or flops is None:
            continue
        fig.add_trace(go.Scatter(
            x=[flops], y=[acc],
            mode='markers+text',
            name=f'Hybrid RF ({layer_name})',
            marker=dict(symbol='diamond', size=12, color=_GREEN),
            text=[layer_name],
            textposition='top center',
        ))

    # Full CNN baseline
    full_acc   = hybrid_metrics.get('full_cnn_accuracy') or hybrid_metrics.get('cnn_accuracy')
    full_flops = hybrid_metrics.get('full_cnn_flops')
    if full_acc and full_flops:
        fig.add_trace(go.Scatter(
            x=[full_flops], y=[full_acc],
            mode='markers+text',
            name='Full VGG16',
            marker=dict(symbol='square', size=12, color=_NAVY),
            text=['Full VGG16'],
            textposition='top center',
        ))


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def generate_summary_table(results: list[dict]) -> str:
    """Return a Markdown table string summarising all IMP rounds.

    Args:
        results: List of round dicts.

    Returns:
        Markdown string.
    """
    header = (
        "| Round | Sparsity | Val Acc | Val F1 | Active Params | FLOPs | Inf (ms/sample) |\n"
        "|------:|:--------:|:-------:|:------:|:-------------:|:-----:|:---------------:|"
    )
    rows = []
    for r in results:
        rows.append(
            f"| {r['round']:>5} "
            f"| {r['global_sparsity']:>7.1%} "
            f"| {r['val_accuracy']:>7.4f} "
            f"| {r['val_f1']:>6.4f} "
            f"| {r['active_params']:>13,} "
            f"| {r['estimated_flops']:>9.3e} "
            f"| {r['inference_ms_per_sample']:>15.3f} |"
        )
    return header + '\n' + '\n'.join(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate IMP sparsity visualizations.'
    )
    parser.add_argument(
        '--results', type=str, required=True,
        help='Path to sparsity_vs_accuracy.json',
    )
    parser.add_argument(
        '--arch', type=str, required=True,
        help='Architecture: vgg16 or lenet',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for HTML plots (defaults to results file directory)',
    )
    parser.add_argument(
        '--hybrid-metrics', type=str, default=None,
        help='Path to hybrid pipeline test_metrics.json for comparison plot',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = load_results(args.results)

    results_path = Path(args.results)
    out_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    arch = args.arch

    plot_sparsity_curve(results, out_dir / 'sparsity_curve.html', arch)
    plot_sparsity_vs_param_count(results, out_dir / 'sparsity_vs_params.html', arch)

    if args.hybrid_metrics:
        compare_with_hybrid_pipeline(
            results,
            args.hybrid_metrics,
            out_dir / 'imp_vs_hybrid_pipeline.html',
        )

    # Print summary table
    table = generate_summary_table(results)
    print(f"\n{'='*80}\nSummary Table\n{'='*80}\n{table}\n")

    # Save as markdown
    md_path = out_dir / 'sparsity_summary.md'
    with open(md_path, 'w') as f:
        f.write(f"# IMP Sparsity Summary — {arch.upper()}\n\n")
        f.write(table)
        f.write('\n')
    print(f"Saved: {md_path}")


if __name__ == '__main__':
    main()
