import os
import torch
import transformers
from transformers import HfArgumentParser
from transformers.utils import is_torch_bf16_gpu_available, is_torch_npu_available

from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams.parser import _TRAIN_ARGS, _verify_model_args, _check_extra_dependencies, get_infer_args
from llamafactory.extras.misc import check_dependencies, get_current_device



def parse_infer_args(parser_config_path):
    
    parser = HfArgumentParser(_TRAIN_ARGS)
    
    if parser_config_path.endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(parser_config_path))
    elif parser_config_path.endswith(".json"):
        return parser.parse_json_file(os.path.abspath(parser_config_path))
    else:
        NotImplementedError("The current format of the configuration file is not supported for parsing.")
        
        

parser_config_path = "examples/streaming_video_pretrain_1_stage/pretrain_1_stage_32_node_20250312_mllm_pretrain_data_15M_V0_5_inference.yaml"
model_args, data_args, training_args, finetuning_args, generating_args = parse_infer_args(parser_config_path)



from llamafactory.extras.logging import get_logger
logger = get_logger(__name__)



if (not training_args.do_train) and model_args.quantization_bit is not None:
    logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

if training_args.bf16 or finetuning_args.pure_bf16:
    model_args.compute_dtype = torch.bfloat16
elif training_args.fp16:
    model_args.compute_dtype = torch.float16

if finetuning_args.pure_bf16:
        if not (is_torch_bf16_gpu_available() or (is_torch_npu_available() and torch.npu.is_bf16_supported())):
            raise ValueError("This device does not support `pure_bf16`.")

_verify_model_args(model_args, data_args, finetuning_args)
_check_extra_dependencies(model_args, finetuning_args, training_args)

model_args.device_map = {"": get_current_device()}
model_args.model_max_length = data_args.cutoff_len
model_args.block_diag_attn = data_args.neat_packing
data_args.packing = data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"

logger.info("Compute dtype: {}".format(str(model_args.compute_dtype)))

transformers.set_seed(training_args.seed)



from llamafactory.model.loader import load_model, load_tokenizer



tokenizer_module = load_tokenizer(model_args)
tokenizer = tokenizer_module["tokenizer"]



from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer, calculate_length



template = get_template_and_fix_tokenizer(tokenizer, data_args)



model = load_model(tokenizer, model_args, finetuning_args, False)



from llamafactory.extras.constants import IGNORE_INDEX

model_type = getattr(model.config, "model_type", None)
if model_type == "mllama":
    mllama_num_tiles = data_args.mllama_num_tiles
    if mllama_num_tiles is not None:
        assert mllama_num_tiles == 1, "mllama_num_tiles can only be reset to 1!"
else:
    mllama_num_tiles = None

data_collator = SFTDataCollatorWith4DAttentionMask(
    template=template,
    mllama_num_tiles=mllama_num_tiles, # for mllama
    model_type=model_type,
    add_image_token_in_input_ids=model_args.add_image_token_in_input_ids,
    add_video_position_encoding=model_args.mllama_add_video_position_encoding,
    pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
    label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    block_diag_attn=model_args.block_diag_attn,
    attn_implementation=getattr(model.config, "_attn_implementation", None),
    compute_dtype=model_args.compute_dtype,
    **tokenizer_module,
)



dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", disable_train_shuffling=finetuning_args.disable_train_shuffling, **tokenizer_module)



dataset = dataset_module["train_dataset"]
sample = [dataset[0]]




features = data_collator(sample)




from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from collections.abc import Mapping

def _prepare_input(data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": model.device}
        return data.to(**kwargs)
    return data



features = _prepare_input(features)



output = model.generate(**features)