#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def _series_colors(n):
    import matplotlib.pyplot as plt
    if n <= 20:
        return plt.cm.tab20(np.linspace(0, 1, n))
    return plt.cm.hsv(np.linspace(0, 1, n, endpoint=False))


def _policy_tag_from_filename(path: Path) -> str:
    stem = path.stem
    if "_org_vs_" in stem:
        return stem.split("_org_vs_", 1)[1]
    return stem


def _policy_label_from_runs(policies, default_label="policy") -> str:
    for run in policies:
        flops_summary = run.get("flops_summary") or {}
        policy_name = flops_summary.get("policy_name")
        if policy_name:
            return policy_name
    return default_label


def _series_label_from_cfg(policy_label: str, cfg: dict) -> str:
    # Match eval_org_vs_new.py naming for combined-policy config bars.
    if {
        "layer_freeze_step",
        "layer_threshold",
        "token_freeze_step",
        "token_threshold",
    }.issubset(cfg.keys()):
        return (
            f"{policy_label}_ls{cfg.get('layer_freeze_step')}_lt{cfg.get('layer_threshold')}_"
            f"ts{cfg.get('token_freeze_step')}_tt{cfg.get('token_threshold')}"
        )
    # Match eval_org_vs_new.py naming for token/layer policy bars.
    return f"{policy_label}_s{cfg.get('freeze_step')}_t{cfg.get('threshold')}"


def _plot_bars(
    task_name,
    metric_label,
    series_labels,
    acc_values,
    flops_values,
    output_dir,
    policy_tag,
    file_suffix="",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(2)
    n_series = len(series_labels)
    width = 0.8 / max(n_series, 1)

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
    ax2 = ax1.twinx()

    colors = _series_colors(n_series)
    offsets = [(i - (n_series - 1) / 2) * width for i in range(n_series)]

    acc_plot = [np.nan if v is None else v for v in acc_values]
    flops_plot = [np.nan if v is None else v for v in flops_values]

    for i, label in enumerate(series_labels):
        ax1.bar(x[0] + offsets[i], acc_plot[i], width, label=label, color=colors[i])
        ax2.bar(x[1] + offsets[i], flops_plot[i], width, color=colors[i])

    ax1.set_ylabel(metric_label)
    ax2.set_ylabel("Total FLOPs (TFLOPs)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Accuracy", "Total FLOPs"])

    title = f"{task_name}: {metric_label} vs TFLOPs"
    fig.suptitle(title, y=0.995, fontsize=10)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.985, 1.0),
        ncol=2,
        fontsize=8,
        columnspacing=1.4,
        handlelength=1.4,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    suffix = f"_{file_suffix}" if file_suffix else ""
    out_path = output_dir / f"{task_name}_org_vs_{policy_tag}{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_single_metric_bars(
    task_name,
    metric_label,
    series_labels,
    values,
    output_dir,
    policy_tag,
    file_suffix,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_vals = [np.nan if v is None else v for v in values]
    x = np.arange(len(series_labels))
    colors = _series_colors(len(series_labels))

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(x, plot_vals, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(series_labels, rotation=20, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{task_name}: {metric_label}")
    fig.tight_layout()
    out_path = output_dir / f"{task_name}_org_vs_{policy_tag}_{file_suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _metric_variant(metrics, key_name):
    if metrics is None:
        return None
    return float(metrics[key_name]) if key_name in metrics else None


def plot_from_json(path: Path, output_dir: Path):
    with path.open() as f:
        data = json.load(f)

    task = data.get("task", path.stem.split("_org_vs_")[0])
    original = data.get("original")
    policies = data.get("policies", [])

    policy_tag = _policy_tag_from_filename(path)
    policy_label = _policy_label_from_runs(policies, default_label="policy")
    # Keep bar-prefix consistent with eval output labels for combined policy.
    if any(
        {"layer_freeze_step", "layer_threshold", "token_freeze_step", "token_threshold"}.issubset(
            (run.get("config") or {}).keys()
        )
        for run in policies
    ):
        policy_label = "combined_policy"

    series_labels = []
    acc_values = []
    flops_values = []
    tps_values = []

    flexible_key = "exact_match,flexible-extract"

    if original:
        series_labels.append("orig")
        acc_values.append(_metric_variant(original.get("metrics"), flexible_key))
        flops_values.append(original.get("tflops"))
        tps_values.append(original.get("tokens_per_second"))

    for run in policies:
        cfg = run.get("config", {})
        series_labels.append(_series_label_from_cfg(policy_label, cfg))
        acc_values.append(_metric_variant(run.get("metrics"), flexible_key))
        flops_values.append(run.get("tflops"))
        tps_values.append(run.get("tokens_per_second"))

    if any(v is not None for v in acc_values):
        _plot_bars(
            task,
            "Exact Match (flexible)",
            series_labels,
            acc_values,
            flops_values,
            output_dir,
            policy_tag=policy_tag,
            file_suffix="flexible",
        )

    if any(v is not None for v in tps_values):
        _plot_single_metric_bars(
            task_name=task,
            metric_label="Tokens per second",
            series_labels=series_labels,
            values=tps_values,
            output_dir=output_dir,
            policy_tag=policy_tag,
            file_suffix="tokens_per_second",
        )


def main():
    parser = argparse.ArgumentParser(description="Plot policy results from JSON files.")
    parser.add_argument(
        "--results",
        default="policy/results_use_block_cache",
        help="Path to a JSON file or a directory containing result JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        default="policy/plots_use_block_cache",
        help="Directory to write plots.",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)

    if results_path.is_file():
        plot_from_json(results_path, output_dir)
        return

    json_paths = sorted(results_path.glob("*.json"))
    for path in json_paths:
        plot_from_json(path, output_dir)


if __name__ == "__main__":
    main()
