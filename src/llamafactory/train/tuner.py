# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel

from ..data import get_template_and_fix_tokenizer
from ..extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.logging import get_logger
from ..hparams import get_infer_args, get_train_args
from ..model import load_model, load_tokenizer
from .callbacks import LogCallback
# Add optional CUDA memory snapshot callback (enabled via env var)
try:
    from .cuda_mem_snapshot_callback import CudaMemorySnapshotCallback  # noqa: F401
except Exception:
    CudaMemorySnapshotCallback = None  # type: ignore
from .dpo import run_dpo
from .kto import run_kto
from .ppo import run_ppo
from .pt import run_pt
from .rm import run_rm
from .sft import run_sft


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = get_logger(__name__)


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    callbacks.append(LogCallback())
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    # # Optionally enable CUDA memory snapshot via environment variables
    # # MEMSNAP_ENABLE: 1/true to enable
    # # MEMSNAP_DUMP_DIR: directory to save snapshots (default: training_args.output_dir)
    # # MEMSNAP_START_STEP, MEMSNAP_DURATION_STEPS, MEMSNAP_MAX_ENTRIES, MEMSNAP_DUMP_TAG
    # memsnap_enable = os.environ.get("MEMSNAP_ENABLE", "0").lower() in ["1", "true", "yes", "on"]
    # if memsnap_enable and CudaMemorySnapshotCallback is not None:
    #     dump_dir = os.environ.get("MEMSNAP_DUMP_DIR", training_args.output_dir)
    #     try:
    #         start_step = int(os.environ.get("MEMSNAP_START_STEP", "5"))
    #         duration_steps = int(os.environ.get("MEMSNAP_DURATION_STEPS", "5"))
    #         max_entries = int(os.environ.get("MEMSNAP_MAX_ENTRIES", "200000"))
    #     except Exception:
    #         start_step, duration_steps, max_entries = 2, 1, 200000
    #     dump_tag = os.environ.get("MEMSNAP_DUMP_TAG", "")
    #     callbacks.append(
    #         CudaMemorySnapshotCallback(
    #             dump_dir=dump_dir,
    #             enable=True,
    #             start_step=start_step,
    #             duration_steps=duration_steps,
    #             max_entries=max_entries,
    #             dump_tag=dump_tag,
    #         )
    #     )

    if finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "kto":
        run_kto(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError("Unknown task: {}.".format(finetuning_args.stage))


def export_model(args: Optional[Dict[str, Any]] = None) -> None:
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args)  # must after fixing tokenizer to resize vocab

    if getattr(model, "quantization_method", None) is not None and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None) is not None:  # quantized model adopts float16 type
        setattr(model.config, "torch_dtype", torch.float16)
    else:
        if model_args.infer_dtype == "auto":
            output_dtype = getattr(model.config, "torch_dtype", torch.float16)
        else:
            output_dtype = getattr(torch, model_args.infer_dtype)

        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)
        logger.info("Convert model dtype to: {}.".format(output_dtype))

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size="{}GB".format(model_args.export_size),
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size="{}GB".format(model_args.export_size),
            safe_serialization=(not model_args.export_legacy_format),
        )

    if finetuning_args.stage == "rm":
        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        if os.path.exists(os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_SAFE_WEIGHTS_NAME),
            )
            logger.info("Copied valuehead to {}.".format(model_args.export_dir))
        elif os.path.exists(os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_WEIGHTS_NAME),
            )
            logger.info("Copied valuehead to {}.".format(model_args.export_dir))

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

        if processor is not None:
            getattr(processor, "image_processor").save_pretrained(model_args.export_dir)
            if model_args.export_hub_model_id is not None:
                getattr(processor, "image_processor").push_to_hub(
                    model_args.export_hub_model_id, token=model_args.hf_hub_token
                )

    except Exception:
        logger.warning("Cannot save tokenizer, please copy the files manually.")
