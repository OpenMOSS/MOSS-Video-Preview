# Copyright 2025
#
# A Transformers TrainerCallback to record CUDA memory history for a few steps
# and dump a memory snapshot, adapted from a Lightning callback implementation.
#
# Usage options:
# 1) Programmatic: pass an instance of CudaMemorySnapshotCallback to run_exp(callbacks=[...])
# 2) Env var (after patching tuner.py to auto-enable via MEMSNAP_ENABLE):
#    export MEMSNAP_ENABLE=1
#    export MEMSNAP_DUMP_DIR=./snapshots   # optional, default to training_args.output_dir
#    export MEMSNAP_START_STEP=2           # optional
#    export MEMSNAP_DURATION_STEPS=1       # optional
#    export MEMSNAP_MAX_ENTRIES=200000     # optional
#    export MEMSNAP_DUMP_TAG=myrun         # optional

import logging
import os
from datetime import datetime

import torch
from transformers import TrainerCallback

TIME_FORMAT_STR = "%b_%d_%H_%M_%S"
DEFAULT_MAX_MEM_EVENTS = 200000


def _start_record_memory_history(max_entries: int = DEFAULT_MAX_MEM_EVENTS) -> None:
    if not torch.cuda.is_available():
        return
    logging.getLogger(__name__).info(
        f"[MemorySnapshot] Start record_memory_history(max_entries={max_entries})"
    )
    try:
        # Private API; guard with try/except in case of incompatibilities.
        torch.cuda.memory._record_memory_history(max_entries=max_entries)  # type: ignore[attr-defined]
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"[MemorySnapshot] _record_memory_history failed: {e}"
        )


def _stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        return
    logging.getLogger(__name__).info("[MemorySnapshot] Stop record_memory_history")
    try:
        torch.cuda.memory._record_memory_history(enabled=None)  # type: ignore[attr-defined]
    except Exception:
        pass


def _export_memory_snapshot(file_path: str) -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    except Exception:
        pass
    logging.getLogger(__name__).info(f"[MemorySnapshot] Dump snapshot -> {file_path}")
    try:
        torch.cuda.memory._dump_snapshot(file_path)  # type: ignore[attr-defined]
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"[MemorySnapshot] _dump_snapshot failed: {e}"
        )


class CudaMemorySnapshotCallback(TrainerCallback):
    """
    在指定的起始步起，记录 duration_steps 个训练步的 CUDA 内存分配历史，
    并在结束后导出 .pickle 快照。多卡时，文件名包含 rank，避免覆盖。

    注意：这里使用的是 Transformers 的 TrainerCallback，而不是 Lightning 的 Callback。
    """

    def __init__(
        self,
        dump_dir: str,
        enable: bool = True,
        start_step: int = 2,
        duration_steps: int = 1,
        max_entries: int = DEFAULT_MAX_MEM_EVENTS,
        dump_tag: str = "",
    ) -> None:
        super().__init__()
        self.dump_dir = dump_dir
        self.enable = enable
        self.start_step = start_step
        self.duration_steps = duration_steps
        self.max_entries = max_entries
        self.dump_tag = dump_tag
        self.recording = False

    # In HF Trainer, state.global_step is the number of optimizer steps completed.
    # on_step_begin is called before an optimizer step, so equality check matches the intended step.
    def on_step_begin(self, args, state, control, **kwargs):
        if not self.enable or not torch.cuda.is_available():
            return
        step_now = state.global_step
        if (not self.recording) and (step_now == self.start_step):
            _start_record_memory_history(self.max_entries)
            self.recording = True

    # on_step_end sees state.global_step already incremented for the just-finished step.
    def on_step_end(self, args, state, control, **kwargs):
        if not self.enable or not torch.cuda.is_available():
            return
        step_end = state.global_step
        if self.recording and step_end >= self.start_step + self.duration_steps:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            timestamp = datetime.now().strftime(TIME_FORMAT_STR)
            file_name = (
                f"memsnap_rank{rank}_steps{self.start_step}-{step_end-1}_{timestamp}"
            )
            if self.dump_tag:
                file_name += f"_{self.dump_tag}"
            file_path = os.path.join(self.dump_dir, f"{file_name}.pickle")
            _export_memory_snapshot(file_path)
            _stop_record_memory_history()
            self.recording = False

    def on_train_end(self, args, state, control, **kwargs):
        # 容错：若仍在记录，确保导出并停止
        if self.recording and torch.cuda.is_available():
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            timestamp = datetime.now().strftime(TIME_FORMAT_STR)
            file_path = os.path.join(
                self.dump_dir, f"memsnap_rank{rank}_eot_{timestamp}.pickle"
            )
            _export_memory_snapshot(file_path)
            _stop_record_memory_history()
            self.recording = False
