"""PyTorch VideoMllama model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.models.mllama.modeling_mllama import (
    MllamaConfig,
    MllamaVisionModel,
    MllamaForConditionalGeneration,
    MllamaForCausalLM,
    MllamaTextModel,
    MllamaTextCrossAttention,
    MllamaTextCrossSdpaAttention,
    MllamaCrossAttentionDecoderLayer,
    MllamaSelfAttentionDecoderLayer,
    _prepare_cross_attention_mask,
    rotate_half,
    repeat_kv,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, CausalLMOutputWithPast

from transformers.utils import logging
logger = logging.get_logger(__name__)


def weighted_cross_entropy(
    source, 
    target, 
    silence_token_id, 
    silence_token_weight: float = 0.1, 
    num_items_in_batch: int = None, 
    ignore_index: int = -100, 
    **kwargs
):
    """
    Calculate weighted cross entropy loss with 0.1 weight for silence token positions and 1.0 for others
    
    Args:
        source: predicted logits [batch_size * seq_len, vocab_size]
        target: ground truth labels [batch_size * seq_len]
        silence_token_id: the id of silence token
        num_items_in_batch: number of items in batch
        ignore_index: index to ignore
    """
    # Create weight tensor with default weight 1.0
    weight = torch.ones_like(target, dtype=torch.float)
    
    # Set weight to 0.1 for silence token positions
    weight[target == silence_token_id] = silence_token_weight
    
    # Set weight to 0.0 for ignore_index positions (so they won't contribute to loss)
    weight[target == ignore_index] = 0.0
    
    reduction = "none"  # Don't reduce first, we need to apply weights
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    
    # Apply weights
    weighted_loss = loss * weight
    
    # Aggregate loss based on whether num_items_in_batch is provided
    if num_items_in_batch is not None:
        # Sum and divide by batch size
        total_loss = weighted_loss.sum() / num_items_in_batch
    else:
        # Calculate weighted average
        # Only average over non-ignore positions
        valid_mask = (target != ignore_index)
        if valid_mask.sum() > 0:
            total_loss = weighted_loss.sum() / valid_mask.sum()
        else:
            total_loss = weighted_loss.sum() * 0
    
    return total_loss


def ForCausalLMLossWithSilence(
    logits, 
    labels, 
    vocab_size: int, 
    silence_token_id: int, 
    silence_token_weight: float = 0.1, 
    num_items_in_batch: int = None, 
    ignore_index: int = -100, 
    **kwargs
):
    """
    Loss function for causal language model with silence token weight adjustment
    
    Args:
        logits: model output logits [batch_size, seq_len, vocab_size]
        labels: ground truth labels [batch_size, seq_len]
        vocab_size: vocabulary size
        silence_token_id: the id of silence token
        num_items_in_batch: number of items in batch
        ignore_index: index to ignore
    """
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    
    loss = weighted_cross_entropy(
        shift_logits, shift_labels, silence_token_id, silence_token_weight,
        num_items_in_batch, ignore_index, **kwargs
    )
    return loss


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def _apply_rotary_pos_emb(states, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    states_embed = (states * cos) + (rotate_half(states) * sin)
    return states_embed


class VideoMllamaTextCrossAttention(MllamaTextCrossAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = None,
        position_ids: Optional[torch.LongTensor] = None,  # vision_position_ids
        cache_position: Optional[torch.LongTensor] = None,  # vision_cache_position
        position_embeddings: Optional[torch.Tensor] = None,  # vision_position_embeddings
        query_position_embeddings: Optional[torch.Tensor] = None,  # position_embeddings
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)

        if cross_attention_states is not None:
            key_states = self.k_proj(cross_attention_states)
            value_states = self.v_proj(cross_attention_states)
            key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            
            # add rope embedding for query and key
            # Note that position embedding is not the same for query and key
            cos, sin = query_position_embeddings
            query_states= _apply_rotary_pos_emb(query_states, cos, sin)
            vision_cos, vision_sin = position_embeddings
            # vision_cos.shape is (batch_size, num_concurrent_media * num_tiles * num_patches, head_dim)
            # key_states.shape is (batch_size, num_key_value_heads, num_concurrent_media * num_tiles * num_patches, head_dim)
            key_states = _apply_rotary_pos_emb(key_states, vision_cos, vision_sin)
            
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            key_states = self.k_norm(key_states)

            if past_key_value is not None:
                # if we have a new image + new tokens, we only computed key_states on that new image
                # we still update the cross key states, past_image, new_image. And use it!
                head_num = key_states.shape[1]
                num_concurrent_media = len(cache_position)
                
                key_states = key_states.view(bsz, head_num, num_concurrent_media, -1, self.head_dim).transpose(2, 3)
                # key_states.shape is (batch_size, head_num, num_tiles * num_patches, num_concurrent_media, head_dim)
                key_states = key_states.reshape(bsz, -1, num_concurrent_media, self.head_dim)
                # key_states.shape is (batch_size, head_num * num_tiles * num_patches, num_concurrent_media, head_dim)
                value_states = value_states.view(bsz, head_num, num_concurrent_media, -1, self.head_dim).transpose(2, 3)
                # value_states.shape is (batch_size, head_num, num_tiles * num_patches, num_concurrent_media, head_dim)
                value_states = value_states.reshape(bsz, -1, num_concurrent_media, self.head_dim)
                # value_states.shape is (batch_size, head_num * num_tiles * num_patches, num_concurrent_media, head_dim)

                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

                # Restore the shape of key_states and value_states
                num_concurrent_media = key_states.shape[-2]
                key_states = key_states.reshape(bsz, head_num, -1, num_concurrent_media, self.head_dim)
                # key_states.shape is (batch_size, head_num, num_tiles * num_patches, num_concurrent_media, head_dim)
                key_states = key_states.transpose(2, 3).reshape(bsz, head_num, -1, self.head_dim)
                value_states = value_states.reshape(bsz, head_num, -1, num_concurrent_media, self.head_dim)
                # value_states.shape is (batch_size, head_num, num_tiles * num_patches, num_concurrent_media, head_dim)
                value_states = value_states.transpose(2, 3).reshape(bsz, head_num, -1, self.head_dim)
                # key_states.shape is (batch_size, head_num, num_concurrent_media * num_tiles * num_patches, head_dim)
        elif past_key_value is not None:
            key_states, value_states = (
                past_key_value.key_cache[self.layer_idx],
                past_key_value.value_cache[self.layer_idx],
            )
            # Restore the shape of key_states and value_states
            num_concurrent_media = key_states.shape[-2]
            head_num = self.num_heads   # num_key_value_heads * num_key_value_groups
            key_states = key_states.reshape(bsz, head_num, -1, num_concurrent_media, self.head_dim)
            # key_states.shape is (batch_size, head_num, num_tiles * num_patches, num_concurrent_media, head_dim)
            key_states = key_states.transpose(2, 3).reshape(bsz, head_num, -1, self.head_dim)
            value_states = value_states.reshape(bsz, head_num, -1, num_concurrent_media, self.head_dim)
            # value_states.shape is (batch_size, head_num, num_tiles * num_patches, num_concurrent_media, head_dim)
            value_states = value_states.transpose(2, 3).reshape(bsz, head_num, -1, self.head_dim)
            # key_states.shape is (batch_size, head_num, num_concurrent_media * num_tiles * num_patches, head_dim)
        else:
            raise ValueError(
                "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
            )
        
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)
        # key_states = self.k_norm(key_states)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class VideoMllamaTextCrossSdpaAttention(MllamaTextCrossSdpaAttention):
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = None,
        position_ids: Optional[torch.LongTensor] = None,  # vision_position_ids
        cache_position: Optional[torch.LongTensor] = None,  # vision_cache_position
        position_embeddings: Optional[torch.Tensor] = None,  # vision_position_embeddings
        query_position_embeddings: Optional[torch.Tensor] = None,  # position_embeddings
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MllamaModel is using MllamaTextCrossSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                cross_attention_states=cross_attention_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)

        if cross_attention_states is not None:
            key_states = self.k_proj(cross_attention_states)
            value_states = self.v_proj(cross_attention_states)
            key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # add rope embedding for query and key
            # Note that position embedding is not the same for query and key
            cos, sin = query_position_embeddings
            query_states= _apply_rotary_pos_emb(query_states, cos, sin)
            vision_cos, vision_sin = position_embeddings
            key_states = _apply_rotary_pos_emb(key_states, vision_cos, vision_sin)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            # key_states.shape is (batch_size, head_num, num_concurrent_media * num_tiles * num_patches, head_dim)
            key_states = self.k_norm(key_states)

            if past_key_value is not None:
                # if we have a new image + new tokens, we only computed key_states on that new image
                # we still update the cross key states, past_image, new_image. And use it!
                head_num = key_states.shape[1]
                num_concurrent_media = len(cache_position)
                
                key_states = key_states.view(bsz, head_num, num_concurrent_media, -1, self.head_dim).transpose(2, 3)
                # key_states.shape is (batch_size, head_num, num_tiles * num_patches, num_concurrent_media, head_dim)
                key_states = key_states.reshape(bsz, -1, num_concurrent_media, self.head_dim)
                # key_states.shape is (batch_size, head_num * num_tiles * num_patches, num_concurrent_media, head_dim)
                value_states = value_states.view(bsz, head_num, num_concurrent_media, -1, self.head_dim).transpose(2, 3)
                # value_states.shape is (batch_size, head_num, num_tiles * num_patches, num_concurrent_media, head_dim)
                value_states = value_states.reshape(bsz, -1, num_concurrent_media, self.head_dim)
                # value_states.shape is (batch_size, head_num * num_tiles * num_patches, num_concurrent_media, head_dim)

                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

                # Restore the shape of key_states and value_states
                num_concurrent_media = key_states.shape[-2]
                key_states = key_states.reshape(bsz, head_num, -1, num_concurrent_media, self.head_dim)
                # key_states.shape is (batch_size, head_num, num_tiles * num_patches, num_concurrent_media, head_dim)
                key_states = key_states.transpose(2, 3).reshape(bsz, head_num, -1, self.head_dim)
                value_states = value_states.reshape(bsz, head_num, -1, num_concurrent_media, self.head_dim)
                # value_states.shape is (batch_size, head_num, num_tiles * num_patches, num_concurrent_media, head_dim)
                value_states = value_states.transpose(2, 3).reshape(bsz, head_num, -1, self.head_dim)
                # key_states.shape is (batch_size, head_num, num_concurrent_media * num_tiles * num_patches, head_dim)
        elif past_key_value is not None:
            key_states, value_states = (
                past_key_value.key_cache[self.layer_idx],
                past_key_value.value_cache[self.layer_idx],
            )
            # Restore the shape of key_states and value_states
            num_concurrent_media = key_states.shape[-2]
            head_num = self.num_heads   # num_key_value_heads * num_key_value_groups
            key_states = key_states.reshape(bsz, head_num, -1, num_concurrent_media, self.head_dim)
            # key_states.shape is (batch_size, head_num, num_tiles * num_patches, num_concurrent_media, head_dim)
            key_states = key_states.transpose(2, 3).reshape(bsz, head_num, -1, self.head_dim)
            value_states = value_states.reshape(bsz, head_num, -1, num_concurrent_media, self.head_dim)
            # value_states.shape is (batch_size, head_num, num_tiles * num_patches, num_concurrent_media, head_dim)
            value_states = value_states.transpose(2, 3).reshape(bsz, head_num, -1, self.head_dim)
            # key_states.shape is (batch_size, head_num, num_concurrent_media * num_tiles * num_patches, head_dim)
        else:
            raise ValueError(
                "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
            )
        
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)
        # key_states = self.k_norm(key_states)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if attention_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# Modified from transformers.models.llama.modeling_llama.LlamaDecoderLayer
class VideoMllamaSelfAttentionDecoderLayer(MllamaSelfAttentionDecoderLayer):

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        vision_position_ids: Optional[torch.LongTensor] = None, # vision_position_ids
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        vision_cache_position: Optional[torch.LongTensor] = None, # vision_cache_position
        position_embeddings: Optional[torch.Tensor] = None,
        vision_position_embeddings: Optional[torch.Tensor] = None, # vision_position_embeddings
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        outputs = super().forward(
            hidden_states=hidden_states,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            attention_mask=attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings
        )
        
        return outputs


class VideoMllamaCrossAttentionDecoderLayer(MllamaCrossAttentionDecoderLayer):

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        vision_position_ids: Optional[torch.LongTensor] = None, # vision_position_ids
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        vision_cache_position: Optional[torch.LongTensor] = None, # vision_cache_position
        position_embeddings: Optional[torch.Tensor] = None,
        vision_position_embeddings: Optional[torch.Tensor] = None, # vision_position_embeddings
    ) -> Tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attn_weights, past_key_value = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_ids=vision_position_ids,
            cache_position=vision_cache_position,
            position_embeddings=vision_position_embeddings,
            query_position_embeddings=position_embeddings,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if full_text_row_masked_out_mask is not None:
            hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states  # type: ignore
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


class VideoMllamaTextModel(MllamaTextModel):
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        vision_position_ids: Optional[torch.LongTensor] = None, # vision_position_ids
        cross_attention_states: Optional[torch.FloatTensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vision_cache_position: Optional[torch.LongTensor] = None,   # vision_cache_position
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """

        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, MllamaTextModel

        >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        >>> model = MllamaTextModel.from_pretrained(checkpoint)
        >>> processor = AutoProcessor.from_pretrained(checkpoint)

        >>> text = "<|image|>If I had to write a haiku for this one"
        >>> inputs = processor(text=text, return_tensors="pt")

        >>> output = model(**inputs)

        >>> print(output.last_hidden_state.shape)
        torch.Size([1, 13, 4096])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if cross_attention_states is not None:
            # cross_attention_states.shape is (batch_size, num_concurrent_media, num_tiles, num_patches, dim)
            batch_size, num_concurrent_media, num_tiles, num_patches, hidden_size = cross_attention_states.shape

            ## delay this reshape operation from VideoMllamaForConditionalGeneration.forward()
            cross_attention_states = cross_attention_states.reshape(
                -1, cross_attention_states.shape[-2], hidden_size
            )
            # cross_attention_states.shape is (batch_size * num_concurrent_media * num_tiles, num_patches, dim)

            # Vision Position: calculate vision_cache_position and vision_position_ids
            if vision_cache_position is None:
                # cross_attention_states.shape is (batch_size * num_concurrent_media * num_tiles, num_patches, dim)
                past_seen_media = past_key_values.get_seq_length(self.cross_attention_layers[0]) if past_key_values is not None else 0
                vision_cache_position = torch.arange(
                    past_seen_media, past_seen_media + num_concurrent_media, device=cross_attention_states.device
                )
                # vision_cache_position.shape be: (num_concurrent_media, )
            if vision_position_ids is None:
                vision_position_ids = vision_cache_position.unsqueeze(0)
                # vision_position_ids.shape be: (1, num_concurrent_media)
            assert num_concurrent_media == len(vision_cache_position), "num_concurrent_media is not equal to len(vision_cache_position)"

            # Vision Position: calculate vision_position_embeddings
            # create vision position embeddings to be shared across the decoder layers
            vision_position_embeddings = self.rotary_emb(cross_attention_states, vision_position_ids)
            vision_cos, vision_sin = vision_position_embeddings
            # vision_cos.shape is (batch_size, num_concurrent_media, head_dim)
            # vision_cos.shape should be (batch_size, num_concurrent_media * num_tiles * num_patches, head_dim)
            vision_cos = torch.repeat_interleave(vision_cos, repeats=num_tiles * num_patches, dim=1)
            vision_sin = torch.repeat_interleave(vision_sin, repeats=num_tiles * num_patches, dim=1)
            vision_position_embeddings = (vision_cos, vision_sin)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # For text-only path we should skip cross attention layers.
            # Let's check if the layer is cross attention layer and if we have cross attention states
            # or cached cross attention states.
            is_cross_attention_layer = idx in self.cross_attention_layers
            is_cross_attention_cache_empty = past_key_values is None or (
                past_key_values is not None and past_key_values.get_seq_length(idx) == 0
            )

            if is_cross_attention_layer and cross_attention_states is None and is_cross_attention_cache_empty:
                continue

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    cross_attention_states,
                    cross_attention_mask,
                    causal_mask,
                    full_text_row_masked_out_mask,
                    position_ids,
                    vision_position_ids,    # vision_position_ids
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    vision_cache_position, # vision_cache_position
                    position_embeddings,
                    vision_position_embeddings if cross_attention_states is not None else None, # vision_position_embeddings
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    cross_attention_states=cross_attention_states,
                    cross_attention_mask=cross_attention_mask,
                    attention_mask=causal_mask,
                    full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                    position_ids=position_ids,
                    vision_position_ids=vision_position_ids,    # vision_position_ids
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    vision_cache_position=vision_cache_position, # vision_cache_position
                    position_embeddings=position_embeddings,
                    vision_position_embeddings=vision_position_embeddings if cross_attention_states is not None else None, # vision_position_embeddings
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class VideoMllamaForCausalLM(MllamaForCausalLM):

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        vision_position_ids: Optional[torch.LongTensor] = None, # vision_position_ids
        cross_attention_states: Optional[torch.LongTensor] = None,
        cross_attention_mask: Optional[torch.LongTensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vision_cache_position: Optional[torch.LongTensor] = None,   # vision_cache_position
        num_logits_to_keep: int = 0,
        silence_token_id: int = -100, 
        silence_token_weight: float = 1.0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            vision_position_ids=vision_position_ids,    # vision_position_ids
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            vision_cache_position=vision_cache_position,    # vision_cache_position
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            if silence_token_id >= 0:
                loss = ForCausalLMLossWithSilence(logits, labels, self.vocab_size, silence_token_id=silence_token_id, silence_token_weight=silence_token_weight, **loss_kwargs)
            else:
                loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VideoMllamaForConditionalGeneration(MllamaForConditionalGeneration):

    def __init__(self, config: MllamaConfig, pool_mode=None, stride=None, silence_token_id=-100, silence_token_weight=1.0):
        """copied from transformers.mllama.MllamaForConditionalGeneration.__init__()"""
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.max_num_tiles = config.vision_config.max_num_tiles
        self.vision_output_dim = config.vision_config.vision_output_dim
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        self.vision_model = MllamaVisionModel._from_config(config.vision_config)
        self.language_model = MllamaForCausalLM._from_config(config.text_config)
        self.multi_modal_projector = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=True,
        )
        
        self.silence_token_id = silence_token_id
        self.silence_token_weight = silence_token_weight

        self.pool_mode = pool_mode
        self.stride = stride
        if pool_mode is not None and stride is None:
            raise ValueError("stride must be a positive number when pool_mode is not None")
        
        if pool_mode == "channelFusion":
            self.patch_merger = nn.Conv2d(
                in_channels=self.vision_output_dim,
                out_channels=self.vision_output_dim,
                kernel_size=self.stride,
                stride=self.stride,
                bias=False,
            )
        self.num_vision_tokens = 0  # A variable to record the number of vision tokens in each frame, primarily used in the generate() function.
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        vision_position_ids: Optional[torch.LongTensor] = None,    # vision_position_ids is used to specify video position_id
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vision_cache_position: Optional[torch.LongTensor] = None, # vision_cache_position is used to specify video cache_position_id
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            cross_attention_states = vision_outputs[0]
            if self.pool_mode is not None:
                cross_attention_states = self.get_2dPool(cross_attention_states)
            if self.num_vision_tokens == 0:
                self.num_vision_tokens = cross_attention_states.shape[-2]
            # cross_attention_states.shape is (batch_size, num_concurrent_media, num_tiles, num_patches, dim)
            cross_attention_states = self.multi_modal_projector(cross_attention_states)
            
            ## delay this reshape operation to VideoMllamaTextModel.forward()
            # cross_attention_states = cross_attention_states.reshape(
            #     -1, cross_attention_states.shape[-2], self.hidden_size
            # )
            # After reshape(), cross_attention_states.shape is (batch_size * num_concurrent_media * num_tiles, num_patches, dim)

        if cross_attention_mask is not None:
            assert self.num_vision_tokens != 0, "Please correctly calculate or pass the value of 'num_vision_tokens'."
            cross_attention_mask, full_text_row_masked_out_mask = _prepare_cross_attention_mask(
                cross_attention_mask,
                num_vision_tokens=self.num_vision_tokens,
                dtype=self.dtype,
            )
        else:
            full_text_row_masked_out_mask = None

        if cross_attention_mask is not None and cache_position is not None:
            # cross_attention_mask.shape is (batch_size, 1, seq_len, num_concurrent_media * num_tiles * num_patches)
            cross_attention_mask = cross_attention_mask[:, :, cache_position]
            full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            vision_position_ids=vision_position_ids,  # vision_position_ids
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            cache_position=cache_position,
            vision_cache_position=vision_cache_position,    # vision_cache_position
            num_logits_to_keep=num_logits_to_keep,
            silence_token_id = self.silence_token_id,
            silence_token_weight = self.silence_token_weight,
        )

        return outputs
    
    def get_2dPool(self, hidden_state):
        """
        Args:
            hidden_state: (batch_size, num_concurrent_media, num_tiles, num_patches, dim)
        """
        height = width = self.vision_model.image_size // self.vision_model.patch_size
        batch_size, num_concurrent_media, num_tiles, num_patches, dim = hidden_state.shape
        # num_patches = 1 + height * width
        cls_hidden_state = hidden_state[:, :, :, :1, :]
        img_hidden_state = hidden_state[:, :, :, 1:, :]
        
        image_feature = img_hidden_state.reshape(batch_size * num_concurrent_media * num_tiles, height, width, dim)

        # these codes are referenced from llava-next.llava.model.llava_arch.py.LlavaMetaForCausalLM.get_2dPool()
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        if self.pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, self.stride)
        elif self.pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, self.stride)
        elif self.pool_mode == "bilinear":
            scaled_shape = [math.ceil(height / self.stride), math.ceil(width / self.stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
        elif self.pool_mode == "channelFusion":
            image_feature = self.patch_merger(image_feature)
        else:
            raise ValueError(f"Unexpected pool_mode: {self.pool_mode}")
        
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(batch_size * num_concurrent_media * num_tiles, -1, dim)

        img_hidden_state = image_feature.reshape(batch_size, num_concurrent_media, num_tiles, -1, dim)
        hidden_state = torch.cat([cls_hidden_state, img_hidden_state], dim=-2)

        return hidden_state
    
    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        vision_position_ids=None,    # vision_position_ids is used to specify video position_id
        pixel_values=None,
        aspect_ratio_ids=None,
        aspect_ratio_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        vision_cache_position=None, # vision_cache_position is used to specify video cache_position_id
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        # TODO: we have no attention_mask so this won't work, check if we really won't need attention mask and find another way
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            
        if past_key_values and position_ids is not None:
            position_ids = position_ids[:, -input_ids.shape[1] :]
            # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
            position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "vision_position_ids": vision_position_ids,
                "cache_position": cache_position,
                "vision_cache_position": vision_cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cross_attention_mask": cross_attention_mask,
            }
        )

        # If we're in pre-fill or cacheless decoding step, then we need pixel_values and aspect ratios
        # to compute image hidden states, otherwise they are cached within each cross attn layer
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["aspect_ratio_ids"] = aspect_ratio_ids
            model_inputs["aspect_ratio_mask"] = aspect_ratio_mask

        return model_inputs
        # raise NotImplementedError("`prepare_inputs_for_generation` has not been implemented.")

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        # TODO Currently, our generation code only supports OFFLINE visual understanding, 
        # TODO meaning that all multimodal content and questions must be provided before generating the answer.
        # TODO NEED to implement new inference code to support real-time streaming visual understanding.
        # Notice: `vision_position_ids` and `vision_cache_position` will remain unchanged in **offline** scenarios.
        
        ### Notice ###
        # The following code was directly copied from transformers.modeling_mllama.MllamaForConditionalGeneration._update_model_kwargs_for_generation()
        cross_attention_mask_prev = model_kwargs.get("cross_attention_mask", None)
        position_ids_prev = model_kwargs.get("position_ids", None)
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
        # add cross-attn mask for new token
        if cross_attention_mask_prev is not None:
            # each image or video frame has its own <image> token in text token list
            # we just need to copy last cross_attention_mask for the current newly generated token.
            model_kwargs["cross_attention_mask"] = torch.cat(
                [cross_attention_mask_prev, cross_attention_mask_prev[:, -1:, ...]], dim=1
            )
        # add position_id for new token
        if position_ids_prev is not None:
            last_position_ids = position_ids_prev[:, -1:]
            cur_position_ids = last_position_ids + 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids_prev, cur_position_ids], dim=1
            )
        return model_kwargs
        # raise NotImplementedError("`_update_model_kwargs_for_generation` has not been implemented.")



# def replace_with_VideoMllama(model: MllamaForConditionalGeneration, pool_mode: Optional[str] = None, stride: Optional[int] = None) -> None:
def replace_with_VideoMllama(model: MllamaForConditionalGeneration) -> None:
    attn_implementation = getattr(model.config, "_attn_implementation", None)

    for name, module in model.named_modules():
        # if isinstance(module, MllamaForConditionalGeneration):
        #     module.__class__ = VideoMllamaForConditionalGeneration
        #     module.setup(pool_mode, stride)

        if isinstance(module, MllamaForCausalLM):
            module.__class__ = VideoMllamaForCausalLM
        
        if isinstance(module, MllamaTextModel):
            module.__class__ = VideoMllamaTextModel

        if isinstance(module, MllamaCrossAttentionDecoderLayer):
            module.__class__ = VideoMllamaCrossAttentionDecoderLayer
        if isinstance(module, MllamaSelfAttentionDecoderLayer):
            module.__class__ = VideoMllamaSelfAttentionDecoderLayer

        # MllamaTextCrossSdpaAttention is a sub-class of MllamaTextCrossAttention
        # so, if we need replace MllamaTextCrossSdpaAttention first
        # and if we replace, we should not check if this class is an instance of MllamaTextCrossAttention
        if isinstance(module, MllamaTextCrossSdpaAttention):
            module.__class__ = VideoMllamaTextCrossSdpaAttention
        elif isinstance(module, MllamaTextCrossAttention) and attn_implementation != "npu_flash_attention_2":
            module.__class__ = VideoMllamaTextCrossAttention
        
    
    logger.info("VideoMllama has replaced Mllama.")
    logger.info("The most significant change is the introduction of unified position encoding for both vision and text.")