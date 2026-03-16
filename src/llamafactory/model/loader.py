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

from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict
from types import MethodType

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from transformers.utils import is_torch_npu_available, is_torch_cuda_available, is_flash_attn_2_available
from transformers.cache_utils import Cache
from trl import AutoModelForCausalLMWithValueHead

from ..extras.logging import get_logger
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_ms
from .adapter import init_adapter
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .model_utils.visual import get_image_seqlen
from .patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model

from transformers.models.mllama.modeling_mllama import MllamaVisionEncoder


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_ms(model_args)
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")

    patch_tokenizer(tokenizer)

    try:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        setattr(processor, "tokenizer", tokenizer)
        setattr(processor, "image_seqlen", get_image_seqlen(config))
        setattr(processor, "image_resolution", model_args.image_resolution)
        setattr(processor, "video_resolution", model_args.video_resolution)
        setattr(processor, "video_fps", model_args.video_fps)
        setattr(processor, "video_minlen", model_args.video_minlen)
        setattr(processor, "video_maxlen", model_args.video_maxlen)
        setattr(processor, "extract_frame_func", model_args.extract_frame_func)
        setattr(processor, "frame_extract_num_threads", model_args.frame_extract_num_threads)
    except Exception:
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if "Processor" not in processor.__class__.__name__:
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)

    model_type = getattr(config, "model_type", None)
    # 为了兼容mllama的transformers实现没有兼容flash-attention
    flash_attn = model_args.flash_attn
    if model_type == "mllama" and flash_attn != "auto" and flash_attn != "sdpa":
        model_args.flash_attn = "auto"

    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if model_type == "mllama" and model_args.mllama_add_video_position_encoding:
                from modeling_video_mllama import VideoMllamaForConditionalGeneration
                load_class = VideoMllamaForConditionalGeneration
                init_kwargs["pool_mode"] = model_args.mllama_pool_mode
                init_kwargs["stride"] = model_args.mllama_pool_stride
                silence_token_id = -100 if finetuning_args.silence_token is None else tokenizer.convert_tokens_to_ids(tokenizer.tokenize(finetuning_args.silence_token))[0]
                init_kwargs["silence_token_id"] = silence_token_id
                init_kwargs["silence_token_weight"] = finetuning_args.silence_token_weight
            elif type(config) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
                load_class = AutoModelForVision2Seq
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config)
            else:
                model = load_class.from_pretrained(**init_kwargs)

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if model_type == "mllama" and flash_attn != "auto" and flash_attn != "sdpa":
        ## we use `replace_with_flash_attention` function to replace the forward() of Attention
        #  in mllama.py, only attention_mask of MllamaTextModel need the "_attn_implementation"
        #  specifically, model.language_model.model.config._attn_implementation
        #  this config parameter is used to calculate causal_mask for each text_self_attention layer
        ## so, we only need to accurately set model.language_model.model.config._attn_implementation
        if flash_attn == "fa2" and is_torch_npu_available():
            from NPUAtten import replace_with_flash_attention
            replace_with_flash_attention(model, model_args.mllama_add_video_position_encoding)
            logger.info("*************************************")
            logger.info("NPU Flash Attention Added")
            logger.info("*************************************")
            # set "_attn_implementation" a correct name: "npu_flash_attention_2"
            setattr(model.config, "_attn_implementation", "npu_flash_attention_2")
            setattr(model.vision_model.config, "_attn_implementation", "npu_flash_attention_2")
            setattr(model.language_model.config, "_attn_implementation", "npu_flash_attention_2")
            setattr(model.language_model.model.config, "_attn_implementation", "npu_flash_attention_2")
            # Note: NPUMllamaTextSelfAttention
            # Note: NPUMllamaVisionAttention
            # Note: NPUMllamaTextCrossAttention
            # Note: NPUMllamaTextRMSNorm
        
        elif flash_attn == "fa2" and is_torch_cuda_available():
            if not is_flash_attn_2_available():
                raise RuntimeError("flash attention 2 is not available")
            from FAatten import replace_with_flash_attention
            replace_with_flash_attention(model, model_args.mllama_use_full_attn)
            logger.info("*************************************")
            logger.info("CUDA Flash Attention Added")
            logger.info("*************************************")
            # set "_attn_implementation" a correct name: "flash_attention_2"
            setattr(model.config, "_attn_implementation", "flash_attention_2")
            setattr(model.vision_model.config, "_attn_implementation", "flash_attention_2")
            setattr(model.language_model.config, "_attn_implementation", "flash_attention_2")
            setattr(model.language_model.model.config, "_attn_implementation", "flash_attention_2")
            # Note: FAMllamaTextSelfAttention
            # Note: FAMllamaVisionAttention, MllamaVisionSdpaAttention
            # Note: MllamaTextCrossSdpaAttention
        
        else:
            raise NotImplementedError("'{}' attention has not been implemented for mllama.".format(flash_attn))

    if model_type == "mllama":
        from MllamaVideoModel import replace_vision_encoder
        replace_vision_encoder(model, model_args.mllama_use_full_attn)

    # if we use video position_encoding
    # we should add this replace() function behind attention-block replacement
    if model_type == "mllama" and model_args.mllama_add_video_position_encoding:
        from modeling_video_mllama import replace_with_VideoMllama
        replace_with_VideoMllama(model) #, model_args.mllama_pool_mode, model_args.mllama_pool_stride)
        # Note: VideoMllamaTextCrossSdpaAttention / VideoMllamaTextCrossAttention
    
    ## transfer attention-mask to torch.bool for npu_flash_attention_2
    if model_type == "mllama" and getattr(model.config, "_attn_implementation", None) == "npu_flash_attention_2":
        text_model = model.language_model.model
        
        ## covert causal_mask to bool() for self-attention-layers
        text_model._update_causal_mask_non_bool = text_model._update_causal_mask
        
        def _update_causal_mask_bool(self, *args, **kwargs):
            causal_mask = self._update_causal_mask_non_bool(*args, **kwargs)
            return causal_mask.bool() if causal_mask is not None else None
        
        text_model._update_causal_mask = MethodType(_update_causal_mask_bool, text_model)

        ## convert cross_attention_mask to bool() for cross-attention-layers
        text_model.forward_non_bool = text_model.forward
        
        def forward_bool(self, *args, **kwargs):
            if "cross_attention_mask" in kwargs:
                kwargs["cross_attention_mask"] = kwargs["cross_attention_mask"].bool() \
                      if kwargs["cross_attention_mask"] is not None else None
            return self.forward_non_bool(*args, **kwargs)
        
        text_model.forward = MethodType(forward_bool, text_model)

        ## convert attention_mask to bool() for vision-attention-layers
        transformer = model.vision_model.transformer
        transformer.vision_encoder_forward_non_bool = transformer.forward

        def transformer_vision_encoder_forward_bool(self, *args, **kwargs):
            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = kwargs["attention_mask"].bool() if kwargs["attention_mask"] is not None else None
            return self.vision_encoder_forward_non_bool(*args, **kwargs)
        
        transformer.forward = MethodType(transformer_vision_encoder_forward_bool, transformer)
        
        global_transformer = model.vision_model.global_transformer
        global_transformer.vision_encoder_forward_non_bool = global_transformer.forward

        def global_transformer_vision_encoder_forward_bool(self, *args, **kwargs):
            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = kwargs["attention_mask"].bool() if kwargs["attention_mask"] is not None else None
            return self.vision_encoder_forward_non_bool(*args, **kwargs)
        
        global_transformer.forward = MethodType(global_transformer_vision_encoder_forward_bool, global_transformer)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:,}".format(all_param)

    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model


def expand_lm_head_for_mllama(model, tokenizer, training_args, data_args):
    """
    Expand the vocabulary size of the language model head for mllama model
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        training_args: Training arguments
        data_args: Data arguments
    
    Returns:
        None (modifies model in-place)
    """
    model_type = getattr(model.config, "model_type", None)
    
    # Check if expansion operation is needed
    if not (training_args.do_predict and 
            data_args.predict_dataset is not None and 
            model_type == "mllama"):
        return
    
    # Get current language model head weight and bias
    lm_head_weight = model.language_model.lm_head.weight
    lm_head_size, lm_head_dim = lm_head_weight.shape
    lm_head_bias = model.language_model.lm_head.bias

    # Get embedding matrix information
    embedding_vocab_size, embedding_dim = model.language_model.model.embed_tokens.weight.shape
    expand_vocab_size = embedding_vocab_size

    # Create expanded language model head weight matrix
    expand_lm_head_weight = torch.zeros(
        expand_vocab_size, lm_head_dim, 
        device=lm_head_weight.device, 
        dtype=lm_head_weight.dtype
    )
    expand_lm_head_weight[:lm_head_size, :] = lm_head_weight

    # Get video token vector and use it to fill newly added vocabulary
    video_id = tokenizer.convert_tokens_to_ids("<|video|>")
    vector_for_video_token = lm_head_weight[video_id].clone()
    expand_lm_head_weight[lm_head_size:, :] = vector_for_video_token.unsqueeze(0).repeat(
        expand_vocab_size - lm_head_size, 1
    )

    # Create new linear layer and replace the original language model head
    expand_lm_head = torch.nn.Linear(lm_head_dim, expand_vocab_size, bias=lm_head_bias is not None)
    expand_lm_head.weight = torch.nn.Parameter(expand_lm_head_weight)

    if lm_head_bias is not None:
        expand_lm_head.bias = torch.nn.Parameter(lm_head_bias)
    
    model.language_model.lm_head = expand_lm_head