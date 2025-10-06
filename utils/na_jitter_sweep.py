"""Run a jitter sweep for WaveNA and summarize stabilization behaviour."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from conf.schema import load_config
from evaluation import eval_model
from utils.na_debug_utils import NADebugCallback, save_debug_history


def summarize_tensor_entries(entries: Iterable[Dict[str, torch.Tensor]], key: str) -> Optional[Tuple[float, float, float]]:
    values = []
    for entry in entries:
        value = entry.get(key)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            values.append(value.flatten())
    if not values:
        return None
    stacked = torch.cat(values).float()
    return float(stacked.mean()), float(stacked.median()), float(stacked.std(unbiased=False))


def summarize_early_exit(entries: Iterable[Dict[str, torch.Tensor]]) -> Optional[float]:
    iters = []
    for entry in entries:
        value = entry.get('early_exit_iter')
        if isinstance(value, int):
            iters.append(value)
    if not iters:
        return None
    return float(sum(iters) / len(iters))


def run_sweep(
    config_path: Path,
    jitters: List[float],
    output_dir: Path,
    generate_plots: bool = False,
    log_every_n: int = 1,
    stage: str = 'test',
    save_histories: bool = True,
) -> List[Dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, float]] = []

    for jitter in jitters:
        print(f"\n=== Running jitter={jitter:.4g} ===")
        conf = load_config(str(config_path))

        conf.model = conf.model.model_copy(update={
            'na_seed_strategy': 'jitter',
            'na_seed_jitter_std': float(jitter),
            'na_track_stabilization': True,
        })

        sweep_dir = output_dir / f"jitter_{jitter:.4g}"
        conf.paths = conf.paths.model_copy(update={'results': str(sweep_dir)})

        callback = NADebugCallback(
            log_every_n=log_every_n,
            stage_filter=[stage],
            save_dir=sweep_dir if save_histories else None,
        )

        model = eval_model.run(conf, generate_plots=generate_plots, extra_callbacks=[callback])

        history = model.get_debug_history(stage)
        tensored_entries = []
        for entry in history:
            tensored_entries.append({k: (v if isinstance(v, torch.Tensor) else torch.tensor(v) if isinstance(v, (list, tuple)) else v)
                                     for k, v in entry.items()})

        stabilization_stats = summarize_tensor_entries(tensored_entries, 'stabilization_iters')
        displacement_stats = summarize_tensor_entries(tensored_entries, 'best_displacement')
        final_mse_stats = summarize_tensor_entries(tensored_entries, 'final_target_mse')
        early_exit_avg = summarize_early_exit(tensored_entries)

        row: Dict[str, float] = {
            'jitter': float(jitter),
        }
        if stabilization_stats:
            row.update({
                'stabilization_mean': stabilization_stats[0],
                'stabilization_median': stabilization_stats[1],
                'stabilization_std': stabilization_stats[2],
            })
        if displacement_stats:
            row.update({
                'best_disp_mean': displacement_stats[0],
                'best_disp_median': displacement_stats[1],
                'best_disp_std': displacement_stats[2],
            })
        if final_mse_stats:
            row.update({
                'final_mse_mean': final_mse_stats[0],
                'final_mse_median': final_mse_stats[1],
                'final_mse_std': final_mse_stats[2],
            })
        if early_exit_avg is not None:
            row['early_exit_iter_mean'] = early_exit_avg

        summary_rows.append(row)

        if save_histories:
            history_path = sweep_dir / "na_debug_history.json"
            save_debug_history(model, history_path, stage=stage)
            print(f"Saved debug history to {history_path}")

        summary_path = output_dir / "jitter_summary.json"
        summary_path.write_text(json.dumps(summary_rows, indent=2))
        csv_path = output_dir / "jitter_summary.csv"
        with csv_path.open('w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=sorted({k for row in summary_rows for k in row.keys()}))
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    return summary_rows


def plot_stabilization(summary_rows: List[Dict[str, float]], output_dir: Path) -> Optional[Path]:
    jitters = [row['jitter'] for row in summary_rows if 'stabilization_mean' in row]
    means = [row['stabilization_mean'] for row in summary_rows if 'stabilization_mean' in row]
    if not jitters:
        print("No stabilization statistics available for plotting.")
        return None

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(jitters, means, marker='o')
    ax.set_xlabel('NA seed jitter std')
    ax.set_ylabel('Mean stabilization iteration')
    ax.set_title('NA Stabilization vs. Jitter')
    ax.grid(True, alpha=0.3)
    plot_path = output_dir / "jitter_vs_stabilization.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")
    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WaveNA jitter sensitivity sweep.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--jitters", nargs='+', type=float, required=True, help="List of jitter std values to evaluate")
    parser.add_argument("--output-dir", required=True, help="Directory to store sweep results")
    parser.add_argument("--log-every-n", type=int, default=1, help="How often to log callback metrics")
    parser.add_argument("--stage", default='test', help="Which debug stage to aggregate (train, val, or test)")
    parser.add_argument("--plot", action='store_true', help="Create stabilization vs jitter plot")
    parser.add_argument("--keep-plots", action='store_true', help="Preserve evaluation plots for each jitter")
    parser.add_argument("--no-save-history", action='store_true', help="Skip saving per-jitter debug history")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = run_sweep(
        config_path=Path(args.config),
        jitters=[float(j) for j in args.jitters],
        output_dir=Path(args.output_dir),
        generate_plots=args.keep_plots,
        log_every_n=args.log_every_n,
        stage=args.stage,
        save_histories=not args.no_save_history,
    )

    if args.plot:
        plot_stabilization(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
