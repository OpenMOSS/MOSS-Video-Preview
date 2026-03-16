# Copyright 2024 OpenAccess AI Collective and the LlamaFactory team.
#
# This code is inspired by the OpenAccess AI Collective's axolotl library.
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence

from typing import List, Optional, Union
import numpy as np
import torch
from transformers import DataCollatorForSeq2Seq
from transformers.utils import PaddingStrategy

from .mm_plugin import REAL_TIME_QA_MODE


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from .template import Template


def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""
    Expands the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    while handles packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    e.g.
    ```python
    # input
    [[1, 1, 2, 2, 2, 0]]
    # output
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    bsz, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
    # Create a binary mask from the original mask where zeros remain zeros and all other values are set to one
    padding_mask = torch.where(expanded_mask != 0, 1, 0)
    # Create a block-diagonal mask.
    attention_mask_4d = torch.eq(expanded_mask, expanded_mask.transpose(-1, -2)).int() * padding_mask
    # Use the lower triangular mask to zero out the upper triangular part
    attention_mask_4d *= torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long))
    # Invert the attention mask.
    attention_mask_4d = torch.where(attention_mask_4d != 0, torch.tensor(0, dtype=dtype), min_dtype)
    return attention_mask_4d


def get_media_order_numpy(input_ids, image_token_id, video_token_id):
    """
    High-performance version using numpy to extract media order from input_ids
    
    Args:
        input_ids: Token sequence
        image_token_id: Token ID for images
        video_token_id: Token ID for videos
    
    Returns:
        list: Media type sequence, 0=image, 1=video
    """
    input_ids = np.array(input_ids)
    
    # Find positions of all image and video tokens
    image_mask = input_ids == image_token_id
    video_mask = input_ids == video_token_id
    
    # Get position indices
    image_indices = np.where(image_mask)[0]
    video_indices = np.where(video_mask)[0]
    
    # Combine and sort by position
    all_indices = np.concatenate([image_indices, video_indices])
    media_types = np.concatenate([np.zeros(len(image_indices), dtype=int), 
                                  np.ones(len(video_indices), dtype=int)])
    
    # Sort by position order
    sort_order = np.argsort(all_indices)
    
    return media_types[sort_order].tolist()


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""
    Data collator that supports VLMs.

    Features should contain input_ids, attention_mask, labels and images.
    """

    is_training: bool = False
    template: Optional["Template"] = None
    mllama_num_tiles: int = None
    model_type: Optional["str"] = None
    pad_position_id: int = 0
    cross_attention_token_mask_pad_token_id: int = -100
    processor: Optional["ProcessorMixin"] = None
    add_video_position_encoding: bool = False
    add_image_token_in_input_ids: bool = False

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        batch_images, batch_videos, batch_imglens, batch_vidlens, batch_seqlens = [], [], [], [], []
        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            if self.model_type == "mllama":
                # create list of [list of images]: (batch_size, image_num, )
                batch_images.append(images)
                batch_videos.append(videos)
                continue
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_seqlens.append(len(feature["input_ids"]))

        if self.model_type == "mllama":
            # get image_token_id
            image_token = self.template.mm_plugin.image_token
            image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
            # get video_token_id
            video_token = self.template.mm_plugin.video_token
            video_token_id = self.tokenizer.convert_tokens_to_ids(video_token)
            assert video_token_id is not None, "You should set video_token_id for mllama! Default to 128255! Modify tokenizer.json and tokenizer_config.json."
            
            batch_media_orders = [get_media_order_numpy(feature['input_ids'], image_token_id, video_token_id) for feature in features]
            
            mm_inputs, frame_num_per_video = self.template.mm_plugin.get_mm_inputs(
                batch_media_orders, batch_images, batch_videos, batch_imglens, batch_vidlens, batch_seqlens, self.processor, self.mllama_num_tiles, self.is_training
            )
        else:
            mm_inputs = self.template.mm_plugin.get_mm_inputs(
                batch_images, batch_videos, batch_imglens, batch_vidlens, batch_seqlens, self.processor
            )
            
        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        ####### DataCollator for [Llama3-V] ####### 
        if self.model_type == "mllama":
            # get image_token_id
            image_token = self.template.mm_plugin.image_token
            image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
            # get video_token_id
            video_token = self.template.mm_plugin.video_token
            video_token_id = self.tokenizer.convert_tokens_to_ids(video_token)
            assert video_token_id is not None, "You should set video_token_id for mllama! Default to 128255! Modify tokenizer.json and tokenizer_config.json."

            # whether the input of this batch all pure text data
            is_text_only_inputs = not mm_inputs
            
            if self.is_training:
                assert is_text_only_inputs is not True, "When a batch of data consists purely of text and no images are input into the vision encoder, an NCCL timeout error occurs during gradient synchronization."
            
            # get cross_attention_token_masks from the input_id
            # input_ids, labels, attention_masks： convert <video> to N * <image>
            cross_attention_token_masks = [
                self._get_cross_attention_token_mask(
                    feature, image_token_id, video_token_id, f_num_per_vid, vids
                ) for feature, f_num_per_vid, vids in zip(features, frame_num_per_video, batch_videos)
            ] if not is_text_only_inputs else None

            position_ids, vision_position_ids = [], []
            
            # actual image num after cutting off
            context_image_num = 0
            for idx, feature in enumerate(features):
                context_image_num = max(context_image_num, np.sum(np.array(feature['input_ids'], dtype=np.int64) == image_token_id))
            # Avoid cases where the <image> token count becomes 0 due to truncation (`cut_off_len`) in multimodal input, which could lead to subsequent errors. 
            # Generally, this mainly applies to interleaved inputs.
            if not is_text_only_inputs:
                context_image_num = max(1, context_image_num)
                # update image_num
                num_tiles = mm_inputs.pop("num_tiles")
                mm_inputs = {k: v[:, :context_image_num] for k, v in mm_inputs.items()}
                for idx, n_tile in enumerate(num_tiles):
                    num_tiles[idx] = n_tile[:context_image_num]
                mm_inputs["num_tiles"] = num_tiles

            # delete all <image>_id from input_ids, attention_mask, labels, cross_attention_token_mask
            for idx, feature in enumerate(features):
                ## calculate the current mask for non-image_token_id
                non_image_mask = np.ones(len(feature['input_ids']), dtype=bool) \
                    if self.add_image_token_in_input_ids else (np.array(feature['input_ids'], dtype=np.int64) != image_token_id) 
                image_mask = (np.array(feature['input_ids'], dtype=np.int64) == image_token_id)
                ## set position_ids and vision_position_ids
                if self.add_video_position_encoding and not is_text_only_inputs:
                    position_id = np.arange(len(feature['input_ids']), dtype=np.int64)
                    position_ids.append(position_id[non_image_mask])
                    ### set vision_position_id
                    vision_position_id = position_id[image_mask]
                    if context_image_num == 1 and len(vision_position_id) == 0:
                        vision_position_id = [self.pad_position_id]
                    vision_position_ids.append(vision_position_id)
                ## delete for input_id
                input_ids = np.array(feature['input_ids'], dtype=np.int64)[non_image_mask]
                feature['input_ids'] = input_ids.tolist()
                ## delete for attention_mask
                attention_mask_name = "attention_mask" if "attention_mask" in feature.keys() else "attention_masks"
                attention_mask = np.array(feature[attention_mask_name], dtype=np.int64)[non_image_mask] if attention_mask_name in feature.keys() else None
                if attention_mask is not None:
                    feature[attention_mask_name] = attention_mask.tolist()
                ## delete for labels
                label_name = "label" if "label" in feature.keys() else "labels"
                labels = np.array(feature[label_name], dtype=np.int64) if label_name in feature.keys() else None
                if labels is not None:
                    # convert image_token_id to -100, because we need not to predict image, image is only our input
                    labels[image_mask] = self.label_pad_token_id
                    labels = labels[non_image_mask]
                    feature[label_name] = labels.tolist()
                ## delete for cross_attention_token_mask
                if is_text_only_inputs:
                    continue
                cross_attention_token_mask = np.array(cross_attention_token_masks[idx], dtype=np.int64)[non_image_mask]
                cross_attention_token_masks[idx] = cross_attention_token_mask
            
            # padding input_ids, attention_mask, labels
            features: Dict[str, "torch.Tensor"] = super().__call__(features)

            if not is_text_only_inputs:
                # padding cross_attention_token_masks
                cross_attention_token_masks = self._pad_cross_attention_token_masks(cross_attention_token_masks)["cross_attention_token_masks"]

                # Convert cross_attention_token_masks to cross_attention_mask
                num_tiles = mm_inputs.pop("num_tiles")
                max_num_tiles = mm_inputs['aspect_ratio_mask'].shape[-1]
                cross_attention_mask = self._convert_sparse_cross_attention_mask_to_dense(cross_attention_token_masks, num_tiles, max_num_tiles)

                # add cross_attention_mask to features
                features.update({"cross_attention_mask": cross_attention_mask})

            # add position_ids and vision_position_ids to features
            if self.add_video_position_encoding and not is_text_only_inputs:
                features.update({"position_ids": self._pad_position_ids(position_ids, self.pad_position_id)})
                features.update({"vision_position_ids": self._pad_position_ids(vision_position_ids, self.pad_position_id, pad_to_multiple=False)})
        else:
            features: Dict[str, "torch.Tensor"] = super().__call__(features)
        
        features.update(mm_inputs)
        return features
    
    def _get_cross_attention_token_mask(self, feature, image_token_id, video_token_id, f_num_per_vid, vids):
        """
        Generate a cross-attention-token-mask for each input_tokens in the input sequence.

        [Process Logic]:
        ## 1 ## Assume there are two data points in features
        # features[0]['input_ids']: [122, 122, 122, 128256, 122, 122, 128256, 122, 122]
        # features[1]['input_ids']: [123, 128256, 123]

        ## 2 ## Calculate cross_attention_token_mask for each datapoint according to input_ids
        ##      each <token> can only see the <image> before it, and can see all the <image> before it
        ##      if -100(self.cross_attention_token_mask_pad_token_id) is related to a token, this token can see no <image>
        ##      if N(N >= 0) is related to a token, this token can see <image_0>, <image_1> ... <image_N>
        # features[0]['cross_attention_token_mask']: [-100, -100, -100, 0, 0, 0, 1, 1, 1]
        # features[1]['cross_attention_token_mask']: [-100, 0, 0]
        
        ## 3 ## convert video_token_id to image_token_id, video_frame_num is in f_num_per_vid
        ## image_token_id = 128256
        ## video_token_id = 128255
        ## features[0]['input_ids']: [128255, 122, 128255, 122]
        ## f_num_per_vid[0]: [2, 4]
        ## convert_input_ids[0]: [128256, 128256, 122, 128256, 128256, 128256, 128256, 122]
        """
        input_ids = np.array(feature['input_ids'], dtype=np.int64)
        attention_mask_name = "attention_mask" if "attention_mask" in feature.keys() else "attention_masks"
        attention_mask = np.array(feature[attention_mask_name], dtype=np.int64) if attention_mask_name in feature.keys() else None
        labels_name = "label" if "label" in feature.keys() else "labels"
        labels = np.array(feature[labels_name], dtype=np.int64) if labels_name in feature.keys() else None

        if vids and vids[0].startswith(REAL_TIME_QA_MODE):
            silence_token_id = self.tokenizer.convert_tokens_to_ids("<|silence|>")
            assert silence_token_id is not None, "You should add `<|silence|>` to the vocabulary! Default to 128011! Modify tokenizer.json and tokenizer_config.json."
            ellipsis_token_id = self.tokenizer.convert_tokens_to_ids("<|...|>")
            assert ellipsis_token_id is not None, "You should add `<|...|>` to the vocabulary! Default to 128012! Modify tokenizer.json and tokenizer_config.json."
            end_header_id = self.tokenizer.convert_tokens_to_ids("<|end_header_id|>")
            assert end_header_id is not None, "You should add `<|end_header_id|>` to the vocabulary! Default to 128007! Modify tokenizer.json and tokenizer_config.json."

            # swap video token
            video_token_indices = np.where(input_ids == video_token_id)[0]
            for i in video_token_indices:
                if i < 2:
                    continue

                if (
                    input_ids[i - 1] != silence_token_id
                    and input_ids[i - 2] != end_header_id
                    and input_ids[i - 2] != video_token_id
                ):
                    if np.random.rand() < 0.5:
                        input_ids[i - 1], input_ids[i] = input_ids[i], input_ids[i - 1]
                        if attention_mask is not None:
                            attention_mask[i - 1], attention_mask[i] = attention_mask[i], attention_mask[i - 1]
                        if labels is not None:
                            labels[i - 1], labels[i] = labels[i], labels[i - 1]
        
        # Failed to fetch images, dropping this sample.
        if f_num_per_vid and f_num_per_vid[0] == -2:
            f_num_per_vid = f_num_per_vid[1:]
            labels = [self.label_pad_token_id] * len(labels) if labels_name in feature.keys() else None
        
        total_vid_num = sum(np.array(input_ids) == video_token_id)
        f_num_per_vid = f_num_per_vid[:total_vid_num]
        
        # Failed to fetch video, dropping this sample.
        if f_num_per_vid and f_num_per_vid[0] == -1:
            f_num_per_vid = [1] * len(f_num_per_vid)
            labels = [self.label_pad_token_id] * len(labels) if labels_name in feature.keys() else None
        
        convert_input_ids = np.zeros(
            len(input_ids) + sum(f_num_per_vid) - len(f_num_per_vid), dtype=np.int64
        )
        convert_attention_mask = np.zeros(
            len(attention_mask) + sum(f_num_per_vid) - len(f_num_per_vid), dtype=np.int64
        ) if attention_mask is not None else None
        convert_labels = np.zeros(
            len(labels) + sum(f_num_per_vid) - len(f_num_per_vid), dtype=np.int64
        ) if labels is not None else None

        vid_idx = 0
        convert_idx = 0
        for idx, token_id in enumerate(input_ids):
            if token_id == video_token_id:
                vid_len = f_num_per_vid[vid_idx]
                vid_idx += 1
                convert_input_ids[convert_idx : convert_idx+vid_len] = image_token_id
                if convert_attention_mask is not None:
                    convert_attention_mask[convert_idx : convert_idx+vid_len] = attention_mask[idx]
                if convert_labels is not None:
                    convert_labels[convert_idx : convert_idx+vid_len] = image_token_id
                convert_idx += vid_len
            else:
                convert_input_ids[convert_idx] = token_id
                if convert_attention_mask is not None:
                    convert_attention_mask[convert_idx] = attention_mask[idx]
                if convert_labels is not None:
                    convert_labels[convert_idx] = labels[idx]
                convert_idx += 1
        
        feature["input_ids"] = convert_input_ids
        if convert_attention_mask is not None:
            feature[attention_mask_name] = convert_attention_mask
        if convert_labels is not None:
            if vids and vids[0].startswith(REAL_TIME_QA_MODE):
                assert self.tokenizer.eos_token == "<|eot_id|>", "The eos_token should be '<|eot_id|>' for this template."
                eot_token_id = self.tokenizer.eos_token_id
                convert_labels[convert_labels == eot_token_id] = self.label_pad_token_id
            feature[labels_name] = convert_labels

        image_token_locations = np.where(convert_input_ids == image_token_id)[0]

        cross_attention_token_mask_pad_token_id = self.cross_attention_token_mask_pad_token_id
        vision_masks = np.array([cross_attention_token_mask_pad_token_id] * len(convert_input_ids), dtype=np.int64)

        for i in range(len(image_token_locations)):
            start_idx = image_token_locations[i]
            if i + 1 < len(image_token_locations):
                end_idx = image_token_locations[i+1]
            else:
                end_idx = len(vision_masks)
            vision_masks[start_idx : end_idx] = i

        return vision_masks

    # This function replicates the functionality of [__call__()] from [transformers.DataCollatorForSeq2Seq].
    def _pad_cross_attention_token_masks(self, cross_attention_token_masks, return_tensors=None):
        """
        This function is a direct adaptation of the [__call__()] method from the [transformers.DataCollatorForSeq2Seq] class.
        
        It has been adapted to pad the cross-attention-token-masks 
        following the same logic as `__call__()` does for `input_ids`, `labels`, and `attention_masks`.
        
        Source: [transformers.DataCollatorForSeq2Seq].[__call__()]
        """
        if return_tensors is None:
            return_tensors = self.return_tensors

        batch = {}

        # we have to pad the cross_attention_token_masks manually as we cannot rely on `tokenizer.pad` 
        # and we need them to be of the same length to return tensors
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        
        if no_padding:
            if isinstance(cross_attention_token_masks[0], list):
                batch["cross_attention_token_masks"] = list(cross_attention_token_masks)
            else:
                batch["cross_attention_token_masks"] = [np.concatenate([cross_attn_token_mask, []]) for cross_attn_token_mask in cross_attention_token_masks]
        else:
            max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
            max_label_length = max(len(cross_attn_token_mask) for cross_attn_token_mask in cross_attention_token_masks) if not max_padding else self.max_length
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            cross_attention_token_mask_pad_token_id = self.cross_attention_token_mask_pad_token_id
            if isinstance(cross_attention_token_masks[0], list):
                batch["cross_attention_token_masks"] = [
                    cross_attn_token_mask + [cross_attention_token_mask_pad_token_id] * (max_label_length - len(cross_attn_token_mask))
                    if padding_side == "right"
                    else [cross_attention_token_mask_pad_token_id] * (max_label_length - len(cross_attn_token_mask)) + cross_attn_token_mask
                    for cross_attn_token_mask in cross_attention_token_masks
                ]
            else:
                batch["cross_attention_token_masks"] = [
                    np.concatenate(
                        [
                            cross_attn_token_mask,
                            np.array([cross_attention_token_mask_pad_token_id] * (max_label_length - len(cross_attn_token_mask)), dtype=np.int64),
                        ]
                    )
                    if padding_side == "right"
                    else np.concatenate(
                        [
                            np.array([cross_attention_token_mask_pad_token_id] * (max_label_length - len(cross_attn_token_mask)), dtype=np.int64),
                            cross_attn_token_mask,
                        ]
                    )
                    for cross_attn_token_mask in cross_attention_token_masks
                ]
                batch["cross_attention_token_masks"] = np.array(batch["cross_attention_token_masks"], dtype=np.int64)

        if batch.get("cross_attention_token_masks", None) is not None:
            if return_tensors == "pt":
                import torch

                batch["cross_attention_token_masks"] = torch.tensor(batch["cross_attention_token_masks"], dtype=torch.int64)
            elif return_tensors == "tf":
                import tensorflow as tf

                batch["cross_attention_token_masks"] = tf.constant(batch["cross_attention_token_masks"], dtype=tf.int64)
            else:
                batch["cross_attention_token_masks"] = np.array(batch["cross_attention_token_masks"], dtype=np.int64)
        else:
            batch["cross_attention_token_masks"] = None

        return batch

    # This function replicates the functionality of [convert_sparse_cross_attention_mask_to_dense()] from [transformers.models.mllama.processing_mllama].
    def _convert_sparse_cross_attention_mask_to_dense(
        self,
        cross_attention_token_masks: torch.Tensor,
        num_tiles: List[List[int]],
        max_num_tiles: int,
        return_tensors: Optional[str] = None
    ) -> torch.Tensor:
        """
        Convert the cross attention mask indices to a cross attention mask 4D array.

        This function takes a sparse representation of cross attention masks and converts it to a dense 4D numpy array.
        The sparse representation is a torch.Tensor that defines [the range of images that can be seen] for [each input token].

        Args:
            cross_attention_token_mask (torch.Tensor): A tensor that:
                - defines [the range of images that can be seen] for [each input token].
                - if -100(self.cross_attention_token_mask_pad_token_id) is related to a token, this token can see no <image>.
                - if N(N >= 0) is related to a token, this token can see <image_0>, <image_1> ... <image_N>
            num_tiles (List[List[int]]): A nested list structure specifying the number of tiles for each image in each batch item.
            max_num_tiles (int): The maximum possible number of tiles.

        Returns:
            torch.Tensor: A 4D tensor of shape (batch_size, seq_len, max_num_images, max_num_tiles)
                The tensor contains `1` where attention is allowed and `0` where it is not.
        """

        if return_tensors is None:
            return_tensors = self.return_tensors

        batch_size, seq_len = cross_attention_token_masks.shape
        max_num_images = max([len(n_tiles) for n_tiles in num_tiles])

        import torch
        
        cross_attention_masks = torch.zeros(
            (batch_size, seq_len, max_num_images, max_num_tiles),
            dtype=torch.int64,
        )

        for batch_idx, (cross_attention_token_mask, n_tiles) in enumerate(zip(cross_attention_token_masks, num_tiles)):
            for image_idx, mask_n_tiles in enumerate(n_tiles):
                token_ids = (cross_attention_token_mask >= image_idx) & \
                            (cross_attention_token_mask != self.cross_attention_token_mask_pad_token_id)
                cross_attention_masks[batch_idx, token_ids, image_idx, :mask_n_tiles] = 1

        if return_tensors == "pt":
            import torch
            cross_attention_masks = cross_attention_masks
        elif return_tensors == "tf":
            import tensorflow as tf
            cross_attention_masks = tf.constant(cross_attention_masks.numpy(), dtype=tf.int64)
        else:
            cross_attention_masks = cross_attention_masks.numpy()

        return cross_attention_masks

    # This function replicates the functionality of [__call__()] from [transformers.DataCollatorForSeq2Seq].
    def _pad_position_ids(self, position_ids, pad_token_id, pad_to_multiple=True, return_tensors=None):
        """
        This function is a direct adaptation of the [__call__()] method from the [transformers.DataCollatorForSeq2Seq] class.
        
        It has been adapted to pad the cross-attention-token-masks 
        following the same logic as `__call__()` does for `input_ids`, `labels`, and `attention_masks`.
        
        Source: [transformers.DataCollatorForSeq2Seq].[__call__()]
        """
        if return_tensors is None:
            return_tensors = self.return_tensors

        batch = {}

        # we have to pad the position_ids manually as we cannot rely on `tokenizer.pad` 
        # and we need them to be of the same length to return tensors
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        
        if no_padding:
            if isinstance(position_ids[0], list):
                batch["position_ids"] = list(position_ids)
            else:
                batch["position_ids"] = [np.concatenate([pos_ids, []]) for pos_ids in position_ids]
        else:
            max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
            max_label_length = max(len(pos_ids) for pos_ids in position_ids) if not max_padding else self.max_length
            if pad_to_multiple and self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            if isinstance(position_ids[0], list):
                batch["position_ids"] = [
                    pos_ids + [pad_token_id] * (max_label_length - len(pos_ids))
                    if padding_side == "right"
                    else [pad_token_id] * (max_label_length - len(pos_ids)) + pos_ids
                    for pos_ids in position_ids
                ]
            else:
                batch["position_ids"] = [
                    np.concatenate(
                        [
                            pos_ids,
                            np.array([pad_token_id] * (max_label_length - len(pos_ids)), dtype=np.int64),
                        ]
                    )
                    if padding_side == "right"
                    else np.concatenate(
                        [
                            np.array([pad_token_id] * (max_label_length - len(pos_ids)), dtype=np.int64),
                            pos_ids,
                        ]
                    )
                    for pos_ids in position_ids
                ]
                batch["position_ids"] = np.array(batch["position_ids"], dtype=np.int64)

        if batch.get("position_ids", None) is not None:
            if return_tensors == "pt":
                import torch

                batch["position_ids"] = torch.tensor(batch["position_ids"], dtype=torch.int64)
            elif return_tensors == "tf":
                import tensorflow as tf

                batch["position_ids"] = tf.constant(batch["position_ids"], dtype=tf.int64)
            else:
                batch["position_ids"] = np.array(batch["position_ids"], dtype=np.int64)
        else:
            batch["position_ids"] = None

        return batch["position_ids"]


@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for 4d attention mask.
    """

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)

        return features


@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature["{}_input_ids".format(key)],
                    "attention_mask": feature["{}_attention_mask".format(key)],
                    "labels": feature["{}_labels".format(key)],
                    "images": feature["images"],
                    "videos": feature["videos"],
                }
                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)


@dataclass
class KTODataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for KTO data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        target_features = []
        kl_features = []
        kto_tags = []
        for feature in features:
            target_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"],
                "images": feature["images"],
                "videos": feature["videos"],
            }
            kl_feature = {
                "input_ids": feature["kl_input_ids"],
                "attention_mask": feature["kl_attention_mask"],
                "labels": feature["kl_labels"],
                "images": feature["images"],
                "videos": feature["videos"],
            }
            target_features.append(target_feature)
            kl_features.append(kl_feature)
            kto_tags.append(feature["kto_tags"])

        batch = super().__call__(target_features)
        kl_batch = super().__call__(kl_features)
        batch["kl_input_ids"] = kl_batch["input_ids"]
        batch["kl_attention_mask"] = kl_batch["attention_mask"]
        batch["kl_labels"] = kl_batch["labels"]
        if "token_type_ids" in kl_batch:
            batch["kl_token_type_ids"] = kl_batch["token_type_ids"]

        batch["kto_tags"] = torch.tensor(kto_tags)
        return batch
