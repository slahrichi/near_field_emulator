"""Utilities for inspecting Neural Adjoint (NA) debugging signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from pytorch_lightning.callbacks import Callback


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def detach_debug_info(debug_info: Dict[str, Any], keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Return a CPU copy of ``debug_info`` optionally filtered by ``keys``."""
    if not isinstance(debug_info, dict):
        return {}
    if keys is not None:
        allowed = set(keys)
        debug_info = {k: v for k, v in debug_info.items() if k in allowed}
    return {key: _to_cpu(value) for key, value in debug_info.items()}


def extract_latest_debug_info(model: Any, keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Grab the latest debug info from ``model`` and detach it for inspection."""
    info = getattr(model, "latest_debug_info", {})
    return detach_debug_info(info, keys)


def extract_debug_history(model: Any, stage: Optional[str] = None, keys: Optional[Sequence[str]] = None) -> Dict[str, List[Any]]:
    """Return stored debug history from ``model`` for the requested stage(s)."""
    history = getattr(model, "debug_history", {})
    if not isinstance(history, dict):
        return {}

    stages: Iterable[str]
    if stage is None:
        stages = history.keys()
    else:
        if stage not in history:
            raise ValueError(f"Stage '{stage}' not found in debug history. Available: {list(history.keys())}")
        stages = [stage]

    output: Dict[str, List[Any]] = {}
    for stage_name in stages:
        entries = history[stage_name]
        cleaned: List[Any] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cleaned.append(detach_debug_info(entry, keys))
        output[stage_name] = cleaned
    return output


def save_debug_history(model: Any, output_path: Path, stage: Optional[str] = None, keys: Optional[Sequence[str]] = None) -> None:
    """Persist debug history to ``output_path`` as JSON for quick inspection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = extract_debug_history(model, stage=stage, keys=keys)

    serializable = {}
    for stage_name, entries in history.items():
        serializable[stage_name] = _to_serializable(entries)

    output_path.write_text(json.dumps(serializable, indent=2))


def _to_serializable(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for entry in entries:
        serializable_entry = {}
        for key, value in entry.items():
            if isinstance(value, torch.Tensor):
                serializable_entry[key] = value.tolist()
            else:
                serializable_entry[key] = value
        serialized.append(serializable_entry)
    return serialized


class NADebugCallback(Callback):
    """Lightning callback that records NA debug info after each batch."""

    def __init__(
        self,
        log_every_n: int = 1,
        stage_filter: Optional[Sequence[str]] = None,
        save_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.log_every_n = max(1, int(log_every_n))
        self.stage_filter = set(stage_filter or ["train", "val", "test"])
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.records: Dict[str, List[Dict[str, Any]]] = {stage: [] for stage in self.stage_filter}

    def _handle(self, pl_module: Any, stage: str, batch_idx: int) -> None:
        if stage not in self.stage_filter:
            return
        info = extract_latest_debug_info(pl_module)
        if not info:
            return
        self.records[stage].append(info)

        if batch_idx % self.log_every_n == 0:
            displacement = info.get("best_displacement")
            stabilization = info.get("stabilization_iters")
            early_exit_iter = info.get("early_exit_iter")
            msg_parts = [f"stage={stage}", f"batch={batch_idx}"]
            if isinstance(displacement, torch.Tensor) and displacement.numel() > 0:
                msg_parts.append(f"disp_mean={float(displacement.mean()):.4e}")
            if isinstance(stabilization, torch.Tensor) and stabilization.numel() > 0:
                msg_parts.append(f"stab_mean={float(stabilization.float().mean()):.2f}")
            if isinstance(early_exit_iter, int):
                msg_parts.append(f"early_exit_iter={early_exit_iter}")
            print("[NADebugCallback] " + " | ".join(msg_parts))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._handle(pl_module, "train", batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
        self._handle(pl_module, "val", batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
        self._handle(pl_module, "test", batch_idx)

    def on_test_end(self, trainer, pl_module) -> None:
        if self.save_dir is None:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for stage, records in self.records.items():
            if not records:
                continue
            path = self.save_dir / f"na_debug_{stage}.json"
            path.write_text(json.dumps(_to_serializable(records), indent=2))
