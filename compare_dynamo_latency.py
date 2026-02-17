#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Compare Dynamo latency logs: hint-aware server vs baseline.

Usage:
    python compare_dynamo_latency.py <baseline.jsonl> <hint_aware.jsonl> [--output-dir ./latency_report]

Reads two JSONL files produced by DynamoModelConfig's latency_log_path,
prints a text report, and writes plots to the output directory.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Sensitivity code -> human label
_SENS_LABELS = {"0": "LOW", "1": "MEDIUM", "2": "HIGH"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_log(path: str | Path) -> pd.DataFrame:
    """Load a JSONL latency log into a DataFrame."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["duration_ms"] = df["duration_ms"].astype(float)
    df["latency_sensitivity"] = df["latency_sensitivity"].astype(str)
    df["sens_label"] = df["latency_sensitivity"].map(_SENS_LABELS).fillna(df["latency_sensitivity"])

    # Extract leaf function name from function_path list
    df["node"] = df["function_path"].apply(lambda p: p[-1] if isinstance(p, list) and p else "unknown")

    return df


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _pct_change(baseline: float, treatment: float) -> str:
    if baseline == 0:
        return "N/A"
    pct = (treatment - baseline) / baseline * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def _stat_block(series: pd.Series) -> dict:
    return {
        "count": int(len(series)),
        "mean": series.mean(),
        "median": series.median(),
        "p90": series.quantile(0.90),
        "p95": series.quantile(0.95),
        "p99": series.quantile(0.99),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
    }


def print_report(baseline: pd.DataFrame, treatment: pd.DataFrame) -> str:
    """Build and return a text comparison report."""
    lines: list[str] = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("DYNAMO LATENCY COMPARISON REPORT")
    lines.append(f"Baseline requests : {len(baseline)}")
    lines.append(f"Hint-aware requests: {len(treatment)}")
    lines.append(sep)

    # -- Overall --
    b_stats = _stat_block(baseline["duration_ms"])
    t_stats = _stat_block(treatment["duration_ms"])

    lines.append("")
    lines.append("1. OVERALL LATENCY (ms)")
    lines.append("-" * 60)
    header = f"{'Metric':<10} {'Baseline':>12} {'Hint-Aware':>12} {'Change':>10}"
    lines.append(header)
    lines.append("-" * 60)
    for key in ["mean", "median", "p90", "p95", "p99", "std", "min", "max"]:
        bv, tv = b_stats[key], t_stats[key]
        lines.append(f"{key:<10} {bv:>12.2f} {tv:>12.2f} {_pct_change(bv, tv):>10}")

    # Mann-Whitney U test on overall
    if len(baseline) >= 5 and len(treatment) >= 5:
        u_stat, p_val = stats.mannwhitneyu(baseline["duration_ms"], treatment["duration_ms"], alternative="two-sided")
        lines.append("")
        lines.append(f"  Mann-Whitney U = {u_stat:.1f},  p-value = {p_val:.4g}")
        if p_val < 0.05:
            lines.append("  -> Difference is statistically significant (p < 0.05)")
        else:
            lines.append("  -> Difference is NOT statistically significant (p >= 0.05)")

    # -- Per sensitivity level --
    lines.append("")
    lines.append(sep)
    lines.append("2. LATENCY BY SENSITIVITY LEVEL (ms)")
    lines.append(sep)

    all_sens = sorted(set(baseline["sens_label"].unique()) | set(treatment["sens_label"].unique()),
                      key=lambda s: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(s, 3))

    for sens in all_sens:
        b_sub = baseline[baseline["sens_label"] == sens]["duration_ms"]
        t_sub = treatment[treatment["sens_label"] == sens]["duration_ms"]
        lines.append("")
        lines.append(f"  [{sens}]  baseline n={len(b_sub)}, hint-aware n={len(t_sub)}")
        if len(b_sub) == 0 or len(t_sub) == 0:
            lines.append("    (skipped — no data in one group)")
            continue

        bs, ts = _stat_block(b_sub), _stat_block(t_sub)
        lines.append(f"    {'Metric':<10} {'Baseline':>12} {'Hint-Aware':>12} {'Change':>10}")
        lines.append(f"    {'-'*50}")
        for key in ["mean", "median", "p90", "p95"]:
            lines.append(f"    {key:<10} {bs[key]:>12.2f} {ts[key]:>12.2f} {_pct_change(bs[key], ts[key]):>10}")

        if len(b_sub) >= 5 and len(t_sub) >= 5:
            _, p = stats.mannwhitneyu(b_sub, t_sub, alternative="two-sided")
            lines.append(f"    Mann-Whitney p = {p:.4g}")

    # -- Per node --
    lines.append("")
    lines.append(sep)
    lines.append("3. LATENCY BY GRAPH NODE (ms)")
    lines.append(sep)

    all_nodes = sorted(set(baseline["node"].unique()) | set(treatment["node"].unique()))
    for node in all_nodes:
        b_sub = baseline[baseline["node"] == node]["duration_ms"]
        t_sub = treatment[treatment["node"] == node]["duration_ms"]
        lines.append("")
        lines.append(f"  [{node}]  baseline n={len(b_sub)}, hint-aware n={len(t_sub)}")
        if len(b_sub) == 0 or len(t_sub) == 0:
            lines.append("    (skipped — no data in one group)")
            continue
        bs, ts = _stat_block(b_sub), _stat_block(t_sub)
        for key in ["mean", "median", "p90"]:
            lines.append(f"    {key:<10} {bs[key]:>12.2f} {ts[key]:>12.2f} {_pct_change(bs[key], ts[key]):>10}")

    # -- Request success rates --
    lines.append("")
    lines.append(sep)
    lines.append("4. HTTP STATUS CODES")
    lines.append(sep)
    if "status_code" in baseline.columns and "status_code" in treatment.columns:
        for label, df in [("Baseline", baseline), ("Hint-Aware", treatment)]:
            counts = df["status_code"].value_counts().sort_index()
            lines.append(f"  {label}:")
            for code, n in counts.items():
                lines.append(f"    {code}: {n} ({n / len(df) * 100:.1f}%)")

    report = "\n".join(lines)
    return report


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _color_pair():
    return "#7f8c8d", "#2ecc71"  # grey baseline, green hint-aware


def plot_overall_histogram(baseline: pd.DataFrame, treatment: pd.DataFrame, out: Path):
    """Side-by-side histogram of all request latencies."""
    fig, ax = plt.subplots(figsize=(10, 5))
    c_base, c_treat = _color_pair()

    all_vals = pd.concat([baseline["duration_ms"], treatment["duration_ms"]])
    bins = np.linspace(all_vals.quantile(0.01), all_vals.quantile(0.99), 50)

    ax.hist(baseline["duration_ms"], bins=bins, alpha=0.6, label="Baseline", color=c_base, edgecolor="white")
    ax.hist(treatment["duration_ms"], bins=bins, alpha=0.6, label="Hint-Aware", color=c_treat, edgecolor="white")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Overall Latency Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_overall_histogram.png", dpi=150)
    plt.close(fig)


def plot_overall_cdf(baseline: pd.DataFrame, treatment: pd.DataFrame, out: Path):
    """CDF of request latencies."""
    fig, ax = plt.subplots(figsize=(10, 5))
    c_base, c_treat = _color_pair()

    for df, label, color in [(baseline, "Baseline", c_base), (treatment, "Hint-Aware", c_treat)]:
        sorted_vals = np.sort(df["duration_ms"].values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, label=label, color=color, linewidth=2)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Latency CDF")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "02_overall_cdf.png", dpi=150)
    plt.close(fig)


def plot_boxplot_by_sensitivity(baseline: pd.DataFrame, treatment: pd.DataFrame, out: Path):
    """Box plot of latency grouped by sensitivity level."""
    order = ["LOW", "MEDIUM", "HIGH"]
    present = [s for s in order if s in baseline["sens_label"].values or s in treatment["sens_label"].values]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    positions_base = np.arange(len(present)) * 3
    positions_treat = positions_base + 1
    c_base, c_treat = _color_pair()

    bp_data_base = [baseline[baseline["sens_label"] == s]["duration_ms"].values for s in present]
    bp_data_treat = [treatment[treatment["sens_label"] == s]["duration_ms"].values for s in present]

    bp1 = ax.boxplot(bp_data_base, positions=positions_base, widths=0.8, patch_artist=True,
                     boxprops=dict(facecolor=c_base, alpha=0.7), medianprops=dict(color="black"),
                     flierprops=dict(markersize=3))
    bp2 = ax.boxplot(bp_data_treat, positions=positions_treat, widths=0.8, patch_artist=True,
                     boxprops=dict(facecolor=c_treat, alpha=0.7), medianprops=dict(color="black"),
                     flierprops=dict(markersize=3))

    ax.set_xticks(positions_base + 0.5)
    ax.set_xticklabels(present)
    ax.set_xlabel("Latency Sensitivity")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency by Sensitivity Level")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["KV", "KV + Hint-Aware"])
    fig.tight_layout()
    fig.savefig(out / "03_boxplot_by_sensitivity.png", dpi=150)
    plt.close(fig)


def plot_bar_by_node(baseline: pd.DataFrame, treatment: pd.DataFrame, out: Path):
    """Grouped bar chart of median latency per graph node."""
    all_nodes = sorted(set(baseline["node"].unique()) | set(treatment["node"].unique()))
    if not all_nodes:
        return

    b_medians = [baseline[baseline["node"] == n]["duration_ms"].median() if n in baseline["node"].values else 0
                 for n in all_nodes]
    t_medians = [treatment[treatment["node"] == n]["duration_ms"].median() if n in treatment["node"].values else 0
                 for n in all_nodes]

    x = np.arange(len(all_nodes))
    width = 0.35
    c_base, c_treat = _color_pair()

    fig, ax = plt.subplots(figsize=(max(10, len(all_nodes) * 1.5), 5))
    ax.bar(x - width / 2, b_medians, width, label="Baseline", color=c_base, alpha=0.8)
    ax.bar(x + width / 2, t_medians, width, label="Hint-Aware", color=c_treat, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(all_nodes, rotation=30, ha="right")
    ax.set_ylabel("Median Latency (ms)")
    ax.set_title("Median Latency by Graph Node")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "04_bar_by_node.png", dpi=150)
    plt.close(fig)


def plot_p90_by_node(baseline: pd.DataFrame, treatment: pd.DataFrame, out: Path):
    """Grouped bar chart of p90 latency per graph node."""
    all_nodes = sorted(set(baseline["node"].unique()) | set(treatment["node"].unique()))
    if not all_nodes:
        return

    b_p90 = [baseline[baseline["node"] == n]["duration_ms"].quantile(0.9) if n in baseline["node"].values else 0
             for n in all_nodes]
    t_p90 = [treatment[treatment["node"] == n]["duration_ms"].quantile(0.9) if n in treatment["node"].values else 0
             for n in all_nodes]

    x = np.arange(len(all_nodes))
    width = 0.35
    c_base, c_treat = _color_pair()

    fig, ax = plt.subplots(figsize=(max(10, len(all_nodes) * 1.5), 5))
    ax.bar(x - width / 2, b_p90, width, label="Baseline", color=c_base, alpha=0.8)
    ax.bar(x + width / 2, t_p90, width, label="Hint-Aware", color=c_treat, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(all_nodes, rotation=30, ha="right")
    ax.set_ylabel("p90 Latency (ms)")
    ax.set_title("p90 Latency by Graph Node")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "05_p90_by_node.png", dpi=150)
    plt.close(fig)


def plot_latency_over_time(baseline: pd.DataFrame, treatment: pd.DataFrame, out: Path):
    """Scatter plot of request latency over time for both runs."""
    fig, ax = plt.subplots(figsize=(12, 5))
    c_base, c_treat = _color_pair()

    ax.scatter(baseline["timestamp"], baseline["duration_ms"], alpha=0.4, s=12, color=c_base, label="Baseline")
    ax.scatter(treatment["timestamp"], treatment["duration_ms"], alpha=0.4, s=12, color=c_treat, label="Hint-Aware")

    ax.set_xlabel("Time")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Request Latency Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "06_latency_over_time.png", dpi=150)
    plt.close(fig)


def plot_improvement_waterfall(baseline: pd.DataFrame, treatment: pd.DataFrame, out: Path):
    """Waterfall chart showing latency improvement (%) per node."""
    all_nodes = sorted(set(baseline["node"].unique()) & set(treatment["node"].unique()))
    if not all_nodes:
        return

    improvements = []
    for node in all_nodes:
        b_med = baseline[baseline["node"] == node]["duration_ms"].median()
        t_med = treatment[treatment["node"] == node]["duration_ms"].median()
        if b_med > 0:
            improvements.append(((b_med - t_med) / b_med) * 100)
        else:
            improvements.append(0)

    # Sort by improvement descending
    order = np.argsort(improvements)[::-1]
    nodes_sorted = [all_nodes[i] for i in order]
    impr_sorted = [improvements[i] for i in order]

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in impr_sorted]

    fig, ax = plt.subplots(figsize=(max(10, len(nodes_sorted) * 1.5), 5))
    ax.bar(range(len(nodes_sorted)), impr_sorted, color=colors, alpha=0.8)
    ax.set_xticks(range(len(nodes_sorted)))
    ax.set_xticklabels(nodes_sorted, rotation=30, ha="right")
    ax.set_ylabel("Median Latency Improvement (%)")
    ax.set_title("Latency Improvement by Node (positive = faster with hints)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "07_improvement_waterfall.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare two Dynamo latency JSONL logs (baseline vs hint-aware server).")
    parser.add_argument("baseline", help="Path to baseline JSONL log (server ignoring hints)")
    parser.add_argument("hint_aware", help="Path to hint-aware JSONL log (server using hints)")
    parser.add_argument("--output-dir", default="./latency_report",
                        help="Directory for plots and report (default: ./latency_report)")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    treatment_path = Path(args.hint_aware)

    if not baseline_path.exists():
        print(f"Error: baseline file not found: {baseline_path}", file=sys.stderr)
        sys.exit(1)
    if not treatment_path.exists():
        print(f"Error: hint-aware file not found: {treatment_path}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline:   {baseline_path}")
    baseline = load_log(baseline_path)
    print(f"Loading hint-aware: {treatment_path}")
    treatment = load_log(treatment_path)

    # --- Text report ---
    report = print_report(baseline, treatment)
    print()
    print(report)
    report_path = out / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {report_path}")

    # --- Plots ---
    print("\nGenerating plots...")
    plot_overall_histogram(baseline, treatment, out)
    plot_overall_cdf(baseline, treatment, out)
    plot_boxplot_by_sensitivity(baseline, treatment, out)
    plot_bar_by_node(baseline, treatment, out)
    plot_p90_by_node(baseline, treatment, out)
    plot_latency_over_time(baseline, treatment, out)
    plot_improvement_waterfall(baseline, treatment, out)

    print(f"Plots saved to {out}/")
    print("  01_overall_histogram.png")
    print("  02_overall_cdf.png")
    print("  03_boxplot_by_sensitivity.png")
    print("  04_bar_by_node.png")
    print("  05_p90_by_node.png")
    print("  06_latency_over_time.png")
    print("  07_improvement_waterfall.png")


if __name__ == "__main__":
    main()
