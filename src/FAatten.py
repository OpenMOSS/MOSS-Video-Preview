import importlib
import math
import sys
from packaging import version
from types import MethodType
from typing import Optional, Tuple
from typing_extensions import Unpack
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.mllama.modeling_mllama import (
    MllamaTextCrossAttention,
    MllamaTextSelfAttention,
    MllamaTextConfig,
    MllamaForCausalLM,
    MllamaForConditionalGeneration,
    MllamaVisionAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    MLLAMA_TEXT_CROSS_ATTENTION_CLASSES,
    MLLAMA_TEXT_ATTENTION_CLASSES,
    MllamaVisionSdpaAttention,
    MllamaTextCrossSdpaAttention
)

from transformers.utils import logging, is_torch_sdpa_available
from transformers.utils.import_utils import _is_package_available
from transformers.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)


def is_flash_attn_greater_or_equal_2_10():
    if not _is_package_available("flash_attn"):
        return False
    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")


class FAMllamaTextSelfAttention(MllamaTextSelfAttention):
    def __init__(self, config: MllamaTextConfig,layer_idx:int):
        super().__init__(config,layer_idx)
        self.setup()

    def setup(self):
        self._softmax_scale = 1 / math.sqrt(self.head_dim)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_value=None,
        cache_position=None,
        **kwargs
    ):
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        ## LlamaFlashAttention2.forward() also not do this
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        is_causal = True if q_len > 1 else False
        dropout_rate = self.dropout if self.training else 0.0
        
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=is_causal,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class FAMllamaVisionAttention(MllamaVisionAttention):

    # def __init__(self, config: MllamaVisionConfig):
    def __init__(self, config):
        super().__init__(config)
        self.setup()

    def setup(self):
        self._softmax_scale = 1 / math.sqrt(self.head_dim)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = None,
        **kwargs
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise NotImplementedError("The Flash-attention-2 implementation of MllamaVisionModel can only be used when attention_mask is None.")
        
        output_attentions = False

        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        batch_size, q_seq_len, _ = query.shape
        _, kv_seq_len, _ = key.shape

        query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        query_states = query.transpose(1, 2)
        key_states = key.transpose(1, 2)
        value_states = value.transpose(1, 2)

        is_causal = False

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_seq_len,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=is_causal,
        )
        # attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_seq_len, -1).contiguous()
        output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return output, attn_weights


def replace_with_flash_attention(model: MllamaForConditionalGeneration, use_full_attn: bool = False) -> None:
    for name, module in model.named_modules():
        if isinstance(module, MllamaTextSelfAttention):
            module.__class__ = FAMllamaTextSelfAttention
            module.setup()
        if isinstance(module, MllamaVisionAttention):
            if use_full_attn:
                module.__class__ = FAMllamaVisionAttention
                module.setup()
            elif is_torch_sdpa_available():
                logger.info("*" * 64)
                logger.info(
                    "There is no flash-attn implementation for MllamaVisionAttention, "
                    "so we use sdpa implementation here to speed up the process."
                )
                logger.info("*" * 64)
                module.__class__ = MllamaVisionSdpaAttention
        if isinstance(module, MllamaTextCrossAttention):
            if is_torch_sdpa_available():
                logger.info("*" * 64)
                logger.info(
                        "There is no flash-attn implementation for MllamaTextCrossAttention, "
                        "so we use sdpa implementation here to speed up the process."
                    )
                logger.info("*" * 64)
                module.__class__ = MllamaTextCrossSdpaAttention
