import math
from types import MethodType
from typing import Optional, Tuple
import torch
from transformers.cache_utils import Cache
from transformers.models.mllama.modeling_mllama import (
    MllamaTextCrossAttention,
    MllamaTextSelfAttention,
    MllamaConfig,
    MllamaVisionConfig,
    MllamaTextConfig,
    MllamaForConditionalGeneration,
    MllamaForCausalLM,
    MllamaVisionAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    MLLAMA_TEXT_CROSS_ATTENTION_CLASSES,
    MLLAMA_TEXT_ATTENTION_CLASSES,
    MllamaTextRMSNorm,
    MllamaVisionModel, MllamaPrecomputedAspectRatioEmbedding, MllamaPrecomputedPositionEmbedding, _prepare_aspect_ratio_attention_mask
)
from transformers.modeling_outputs import BaseModelOutput

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from llamafactory.extras import logging
logger = logging.get_logger(__name__)


class MllamaVideoPrecomputedAspectRatioEmbedding(MllamaPrecomputedAspectRatioEmbedding):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = True):
        super().__init__(config, is_gated)

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: (batch_size * num_concurrent_media, num_tiles(video_num_tiles), num_patches, hidden_size)
            aspect_ratio_ids: (batch_size * num_concurrent_media, 1)
        """
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

        num_tiles = hidden_state.shape[1]
        embeddings = embeddings[:, :num_tiles, :, :]

        if self.is_gated:
            embeddings = embeddings * self.gate.tanh()

        hidden_state = hidden_state + embeddings
        return hidden_state


class MllamaVideoPrecomputedPositionEmbedding(MllamaPrecomputedPositionEmbedding):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__(config)

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: hidden_state.shape is (batch_size * num_concurrent_media, num_tiles(video_num_tiles), num_patches, hidden_size)
            aspect_ratio_ids: (batch_size * num_concurrent_media, 1)
        """
        # position embeddings
        gated_position_embedding = (1 - self.gate.tanh()) * self.embedding
        hidden_state = hidden_state + gated_position_embedding.view(1, 1, self.num_patches, self.hidden_size)

        # precomputed tile position embeddings
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(
            batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
        )

        num_tiles = hidden_state.shape[1]
        tile_position_embedding = tile_position_embedding[:, :num_tiles, :, :]

        gated_tile_position_embedding = self.gate.tanh() * tile_position_embedding
        hidden_state = hidden_state + gated_tile_position_embedding

        return hidden_state
    

# detailed dimension infomation comments are in MllamaVisionModel
# MllamaVideoModel is the same as MllamaVisionModel
class MllamaVideoModel(MllamaVisionModel):
    """This class is primarily based on the [MllamaVisionModel] from `transformers.mllama.modeling_mllama`"""

    def __init__(self, config: MllamaVisionConfig):
        super().__init__(config)
        self.setup()

    def setup(self, use_full_attn=False):
        self._use_full_attn = use_full_attn
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        aspect_ratio_ids: torch.Tensor,
        aspect_ratio_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        r"""

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MllamaVisionModel

        >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        >>> model = MllamaVisionModel.from_pretrained(checkpoint)
        >>> processor = AutoProcessor.from_pretrained(checkpoint)

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")

        >>> output = model(**inputs)

        >>> print(output.last_hidden_state.shape)
        torch.Size([1, 1, 4, 1025, 7680])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape
        # pixel_values.shape is (batch_size, num_concurrent_media, num_tiles(video_num_tiles), num_channels, height, width)
        # aspect_ratio_mask.shape is (batch_size, num_concurrent_media, num_tiles(video_num_tiles))

        # DONE
        pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
        # pixel_values.shape is (batch_size * num_concurrent_media * num_tiles(video_num_tiles), num_channels, height, width)
        # aspect_ratio_ids.shape is (batch_size, num_concurrent_media)
        aspect_ratio_ids = aspect_ratio_ids.reshape(batch_size * num_concurrent_media, -1)
        # aspect_ratio_ids.shape is (batch_size * num_concurrent_media, 1)

        # DONE
        # Patch embedding
        patch_embeds = self.patch_embedding(pixel_values.to(self.dtype).to(self.device))
        # patch_embeds.shape is (batch_size * num_concurrent_media * num_tiles(video_num_tiles), hidden_size, image_size // patch_size, image_size // patch_size)
        hidden_state = patch_embeds.flatten(2).transpose(1, 2)
        # hidden_state.shape is (batch_size * num_concurrent_media * num_tiles(video_num_tiles), num_patches, hidden_size)
        # num_patches = (image_size // patch_size) ** 2

        # DONE
        # Tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, -1, dim)
        # hidden_state.shape is (batch_size * num_concurrent_media, num_tiles(video_num_tiles), num_patches, hidden_size)
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        # hidden_state.shape is kept the same

        # DONE
        # Add cls token
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        # hidden_state.shape is (batch_size * num_concurrent_media * num_tiles(video_num_tiles), num_patches, hidden_size)
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1
        # !!! NOTICE !!! num_patches is now increased by 1

        # DONE
        # Position embeddings
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches, dim)
        # hidden_state.shape is (batch_size * num_concurrent_media, num_tiles(video_num_tiles), num_patches, hidden_size)
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)

        # DONE
        hidden_state = self.layernorm_pre(hidden_state)

        # DONE: Whether or not to padd the hidden state ?????????
        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (0, 0, 0, num_padding_patches)  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = F.pad(hidden_state, padding, mode="constant", value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        # DONE
        # Prepare attention mask
        if self._use_full_attn:
            # when using full attention, we need to create attn mask
            attention_mask = None
        else:
            # aspect_ratio_mask.shape is (batch_size, num_concurrent_media, num_tiles(video_num_tiles))
            attention_mask = aspect_ratio_mask.reshape(batch_size * num_concurrent_media, -1)
            # aspect_ratio_mask.shape is (batch_size * num_concurrent_media, num_tiles(video_num_tiles))
            attention_mask = _prepare_aspect_ratio_attention_mask(
                aspect_ratio_mask=attention_mask,
                num_patches=self.num_patches,
                target_length=hidden_state.shape[2],
                dtype=self.dtype,
            )
            # attention_mask.shape is (batch_size * num_concurrent_media, num_tiles(video_num_tiles) * num_patches, num_tiles(video_num_tiles) * num_patches)
            # NOTICE: the num_patches here is the padded hidden_state's num_patches, not the original one

        # DONE
        # Apply encoder
        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
        # hidden_state.shape is (batch_size * num_concurrent_media, num_tiles(video_num_tiles) * num_patches, dim)
        output = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )
        hidden_state = output[0]

        hidden_state = self.layernorm_post(hidden_state)

        # DONE
        # Apply global encoder
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        # hidden_state.shape is (batch_size * num_concurrent_media, num_tiles(video_num_tiles), num_patches, dim)
        hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles * (num_patches + num_padding_patches), dim
        )
        global_output = self.global_transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_state = global_output[0]

        # DONE
        # Remove padding form hidden state
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_state = hidden_state[:, :, :slice_index]
        hidden_state = hidden_state.reshape(batch_size, num_concurrent_media, num_tiles, num_patches, dim)

        # DONE
        # Collect intermediate layer outputs from encoder output
        # all_intermediate_hidden_states = output[1]
        # intermediate_hidden_states = torch.stack(all_intermediate_hidden_states, dim=-1)
        # intermediate_hidden_states = intermediate_hidden_states[..., self.intermediate_layers_indices]
        # Collect intermediate layer outputs from encoder output
        all_intermediate_hidden_states = [output[1][i] for i in self.intermediate_layers_indices]
        intermediate_hidden_states = torch.stack(all_intermediate_hidden_states, dim=-1)

        # DONE
        # Remove padding from intermediate hidden states
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, -1
        )
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, -1
        )

        # DONE
        # Concatenate final hidden state and intermediate hidden states
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states], dim=-1)

        # DONE
        if output_hidden_states:
            hidden_states = tuple(all_intermediate_hidden_states) + tuple(global_output[1])
        else:
            hidden_states = None

        if output_attentions:
            # global transformer in contrast to `self.transformer` doesn't always return hidden states so we might go index out-of-range
            global_attn = tuple(global_output[2]) if output_hidden_states else tuple(global_output[1])
            attentions = tuple(output[2]) + global_attn
        else:
            attentions = None

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states, attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
        )


def replace_vision_encoder(model: MllamaForConditionalGeneration, use_full_attn: bool = False) -> None:
    for name, module in model.named_modules():
        if isinstance(module, MllamaVisionModel):
            module.__class__ = MllamaVideoModel
            module.setup(use_full_attn)
        if isinstance(module, MllamaPrecomputedAspectRatioEmbedding):
            module.__class__ = MllamaVideoPrecomputedAspectRatioEmbedding
        if isinstance(module, MllamaPrecomputedPositionEmbedding):
            module.__class__ = MllamaVideoPrecomputedPositionEmbedding