"""Command line helper for inspecting WaveNA debugging signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional
import sys

try:
    from near_field_emulator.conf.schema import load_config
    from near_field_emulator.evaluation import eval_model
    from near_field_emulator.utils.na_debug_utils import (
        NADebugCallback,
        extract_debug_history,
        extract_latest_debug_info,
        save_debug_history,
    )
except ModuleNotFoundError:
    try:
        from conf.schema import load_config
        from evaluation import eval_model
        from utils.na_debug_utils import (
            NADebugCallback,
            extract_debug_history,
            extract_latest_debug_info,
            save_debug_history,
        )
    except ModuleNotFoundError:
        from pathlib import Path as _Path

        repo_root = _Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root))

        from conf.schema import load_config
        from evaluation import eval_model
        from utils.na_debug_utils import (
            NADebugCallback,
            extract_debug_history,
            extract_latest_debug_info,
            save_debug_history,
        )


def parse_stage_filter(stage_arg: Optional[str]) -> Optional[List[str]]:
    if stage_arg is None:
        return None
    stages = [stage.strip() for stage in stage_arg.split(',') if stage.strip()]
    return stages or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect WaveNA debug outputs for a given config.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--stage", default=None, help="Optional comma-separated list of stages to record (train,val,test)")
    parser.add_argument("--keys", default=None, help="Optional comma-separated list of debug keys to print")
    parser.add_argument("--save-dir", default=None, help="Optional directory to dump debug history JSON files")
    parser.add_argument("--log-every-n", type=int, default=1, help="How often to log callback stats")
    parser.add_argument("--print-latest", action="store_true", help="Print latest debug info to stdout")
    args = parser.parse_args()

    stage_filter = parse_stage_filter(args.stage)
    keys = None if args.keys is None else [k.strip() for k in args.keys.split(',') if k.strip()]

    config = load_config(args.config)

    callback = NADebugCallback(
        log_every_n=args.log_every_n,
        stage_filter=stage_filter,
        save_dir=Path(args.save_dir) if args.save_dir else None,
    )

    model = eval_model.run(config, generate_plots=False, extra_callbacks=[callback])

    if args.print_latest:
        latest_info = extract_latest_debug_info(model, keys=keys)
        print("Latest debug info:")
        print(json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in latest_info.items()}, indent=2))

    if args.save_dir:
        save_root = Path(args.save_dir)
        save_root.mkdir(parents=True, exist_ok=True)
        history_path = save_root / "na_debug_history.json"
        save_debug_history(model, history_path, stage=stage_filter[0] if stage_filter and len(stage_filter) == 1 else None, keys=keys)
        print(f"Saved debug history to {history_path}")

    if stage_filter:
        history = extract_debug_history(model, stage=stage_filter[0] if len(stage_filter) == 1 else None, keys=keys)
    else:
        history = extract_debug_history(model, keys=keys)

    print("Recorded stages:", list(history.keys()))
    for stage_name, entries in history.items():
        print(f"Stage '{stage_name}' batches captured: {len(entries)}")


if __name__ == "__main__":
    main()
