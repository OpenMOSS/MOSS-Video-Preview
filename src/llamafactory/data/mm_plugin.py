import math
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import cv2
import concurrent.futures

import json

import numpy as np
import torch
from typing_extensions import override
import base64
import io
from ..extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ..extras.packages import is_pillow_available, is_pyav_available

from ..extras.logging import get_logger
logger = get_logger(__name__)


PURE_TEXT_BATCH = "PURE_TEXT_BATCH"

if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject
WHITE_IMAGE = Image.new('RGB', (224, 224), 'white')


if is_pyav_available():
    import av


if TYPE_CHECKING:
    from av.stream import Stream
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, EncodedImage, ImageObject]
    VideoInput = str


def _get_paligemma_token_type_ids(
    imglens: Sequence[int], seqlens: Sequence[int], processor: "ProcessorMixin"
) -> List[List[int]]:
    r"""
    Gets paligemma token type ids for computing loss.

    Returns:
        batch_token_type_ids: shape (batch_size, sequence_length)
    """
    batch_token_type_ids = []
    for imglen, seqlen in zip(imglens, seqlens):
        image_seqlen = imglen * getattr(processor, "image_seqlen")
        batch_token_type_ids.append([0] * image_seqlen + [1] * (seqlen - image_seqlen))

    return batch_token_type_ids


class BasePlugin:
    def __init__(self, image_token: Optional[str], video_token: Optional[str]) -> None:
        self.image_token = image_token
        self.video_token = video_token

    def _validate_input(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
    ) -> None:
        r"""
        Validates if this model accepts the input modalities.
        """
        if len(images) != 0 and self.image_token is None:
            raise ValueError("This model does not support image input.")

        if len(videos) != 0 and self.video_token is None:
            raise ValueError("This model does not support video input.")

    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        r"""
        Pre-processes a single image.
        """
        image_resolution: int = kwargs.get("image_resolution")
        if max(image.width, image.height) > image_resolution:
            resize_factor = image_resolution / max(image.width, image.height)
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_frames(self, video_stream: "Stream", total_frames: int = 0, **kwargs):
        r"""
        video_stream is a [PyAV stream].
        Computes video sample frames according to fps.
        """
        video_fps: float = kwargs.get("video_fps")
        video_maxlen: int = kwargs.get("video_maxlen")
        video_minlen: int = kwargs.get("video_minlen")
        
        # [Step 1.1]: obtain av_total_frames directly
        obtained_total_frames = int(video_stream.frames)
        
        # [Step 1.2]: calculate av_total_frames using duration_time && frame_rate
        duration = float(video_stream.duration * video_stream.time_base)
        frame_rate = float(video_stream.average_rate)
        calculated_total_frames = round(duration * frame_rate)
        assert video_fps <= frame_rate, f"Sampling frequency ({video_fps}) must be less than or equal to video frame rate ({frame_rate})"
        
        # [Step 1.3]: calculate the final total_frames
        total_frames_num = [x for x in [total_frames, obtained_total_frames, calculated_total_frames] if x > 0]
        total_frames = min(total_frames_num) if total_frames_num else 0
        if total_frames == 0:
            raise AttributeError("Unable to obtain or calculate the total number of frames in the video.")
        
        # [Step 1.4]: calculate the target total_frames
        # Use ceil to avoid losing the last frame, and subtract a small epsilon for float precision robustness at boundaries.
        target_total_frames = int(math.ceil(duration * video_fps - 1e-6))
        sample_frames = max(target_total_frames, video_minlen)
        sample_frames = min(sample_frames, video_maxlen, total_frames)
        
        # [Step 1.5.1]: Fixed frame rate sampling
        if target_total_frames == sample_frames:
            # sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
            sample_indices = np.arange(target_total_frames, dtype=np.int32)
            sample_indices = (sample_indices * frame_rate / video_fps).astype(np.int32)
            # sample_frame_times = sample_indices / video_fps
        # [Step 1.5.2]: Uniform sampling
        else:
            sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)

        return sample_indices
    
    def _get_cv2_video_sample_frames(self, video: "VideoInput", total_frames: int = 0, **kwargs):
        r"""
        Computes video sample frames according to fps.
        """
        container = av.open(video, "r")
        video_stream = next(stream for stream in container.streams if stream.type == "video")
        sample_frames = self._get_video_sample_frames(video_stream, total_frames=total_frames, **kwargs)
        
        return sample_frames

    def _regularize_images(self, images: Sequence["ImageInput"], **kwargs) -> List["ImageObject"]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        """
        results = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError("Expect input is a list of Images, but got {}.".format(type(image)))

            results.append(self._preprocess_image(image, **kwargs))

        return results

    def _base64_to_image(self,base64_string):
        # Decode the Base64 string into binary data
        img_data = base64.b64decode(base64_string)
        # Use BytesIO to create a file-like object, 
        # and then use PIL's Image.open to read this file-like object.
        img = Image.open(io.BytesIO(img_data))
        return img
    
    def _regularize_videos(self, videos: Sequence["VideoInput"], **kwargs) -> List[List["ImageObject"]]:
        r"""
        Regularizes videos to avoid error. Including reading, resizing and converting.
        """
        results = []
        for video in videos:
            container = av.open(video, "r")
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            total_frames = video_stream.frames
            sample_frames = self._get_video_sample_frames(video_stream, **kwargs)
            sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
            frames: List["ImageObject"] = []
            container.seek(0)
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                if frame_idx in sample_indices:
                    frames.append(frame.to_image())

            frames = self._regularize_images(frames, **kwargs)
            results.append(frames)

        return results

    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 512),
            )
            input_dict["images"] = images

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_resolution=getattr(processor, "video_resolution", 128),
                video_fps=getattr(processor, "video_fps", 1.0),
                video_maxlen=getattr(processor, "video_maxlen", 64),
            )
            input_dict["videos"] = videos

        if input_dict.get("images", None) is not None or input_dict.get("videos", None) is not None:
            return image_processor(**input_dict, return_tensors="pt")
        else:
            return {}

    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        r"""
        Pre-processes input messages before tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return messages

    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        r"""
        Pre-processes token ids after tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        r"""
        Builds batched multimodal inputs for VLMs.
        """
        self._validate_input(images, videos)
        return {}


class LlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        image_seqlen = getattr(processor, "image_seqlen")
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                num_image_tokens += 1
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}", 1)

            message["content"] = content.replace("{{image}}", self.image_token * image_seqlen)

        if len(images) != num_image_tokens:
            raise ValueError("The number of images does not match the number of {} tokens".format(IMAGE_PLACEHOLDER))

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


class PaliGemmaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                num_image_tokens += 1
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}", 1)

            message["content"] = content.replace("{{image}}", "")

        if len(images) != num_image_tokens:
            raise ValueError("The number of images does not match the number of {} tokens".format(IMAGE_PLACEHOLDER))

        return messages

    @override
    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        self._validate_input(images, videos)
        num_images = len(images)
        image_seqlen = num_images * getattr(processor, "image_seqlen")
        image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        input_ids = [image_token_id] * image_seqlen + input_ids
        if labels is not None:
            labels = [IGNORE_INDEX] * image_seqlen + labels

        return input_ids, labels

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        mm_inputs["token_type_ids"] = _get_paligemma_token_type_ids(imglens, seqlens, processor)
        return mm_inputs


class Qwen2vlPlugin(BasePlugin):
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    @override
    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        sample_frames = super()._get_video_sample_frames(video_stream, **kwargs)
        sample_frames = sample_frames // 2 * 2
        return sample_frames

    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError("`len(images)` is less than the number of {} tokens.".format(IMAGE_PLACEHOLDER))

                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.image_token * (image_grid_thw[num_image_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError("`len(videos)` is less than the number of {} tokens.".format(VIDEO_PLACEHOLDER))

                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.video_token * (video_grid_thw[num_video_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError("The number of images does not match the number of {} tokens".format(IMAGE_PLACEHOLDER))

        if len(videos) != num_video_tokens:
            raise ValueError("The number of videos does not match the number of {} tokens".format(VIDEO_PLACEHOLDER))

        return messages

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


def validate_frame_sampling(sample_indices, frames, max_missing_frames=2, max_missing_ratio=0.1):
    """
    Validate the completeness of sampled frames
    
    Args:
        sample_indices: Expected frame indices to sample
        frames: Actually obtained frames
        max_missing_frames: Maximum allowed missing frames (default: 2)
        max_missing_ratio: Maximum allowed missing ratio (default: 0.1)
        
    Raises:
        ValueError: When too many frames are missing
    """
    expected_count = len(sample_indices)
    actual_count = len(frames)
    missing_count = expected_count - actual_count
    
    # No missing frames, validation passed
    if missing_count <= 0:
        return
    
    missing_ratio = missing_count / expected_count
    
    # Check if exceeds threshold: more than max_missing_frames AND ratio > max_missing_ratio
    if missing_count > max_missing_frames and missing_ratio > max_missing_ratio:
        raise ValueError(
            f"Too many frames missing: {missing_count}/{expected_count} "
            f"({missing_ratio:.1%}) frames missing, exceeding "
            f"{max_missing_ratio:.0%} threshold."
        )


STREAMING_EVENT_CAPTION_MODE = "[Streaming Event Caption]"
REAL_TIME_QA_MODE = "[RealTime QA]"

class MllamaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        messages = deepcopy(messages)

        num_image_tokens = 0
        num_video_tokens = 0
        
        for message in messages:
            content = message["content"]
            message["content"] = content.replace(IMAGE_PLACEHOLDER, self.image_token)
            num_image_tokens += message["content"].count(self.image_token)
            content = message["content"]
            message["content"] = content.replace(VIDEO_PLACEHOLDER, self.video_token)
            num_video_tokens += message["content"].count(self.video_token)
        
        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")
        if len(videos) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens.")
        if len(images) != 0 and len(videos) != 0:
            # You need to modify self.get_mm_inputs()
            # Currently, we can not determine the order of <video> and <image>.
            # raise ValueError("Currently, a sample cannot contain both images and videos.")
            pass
        return messages

    def get_video_sample_frames(self, video: "VideoInput", **kwargs):
        container = av.open(video, "r")
        video_stream = next(stream for stream in container.streams if stream.type == "video")
        
        sample_indices = self._get_video_sample_frames(video_stream, **kwargs)
        sample_indices_set = set(sample_indices)
        
        frames: List["ImageObject"] = []
        
        container.seek(0)
        for frame_idx, frame in enumerate(container.decode(video_stream)):
            if frame_idx in sample_indices_set:
                frames.append(frame.to_image())
            if len(frames) == len(sample_indices):
                break
        
        validate_frame_sampling(sample_indices, frames)
        
        return frames, sample_indices
    
    def get_cv2_video_sample_frames_multithread(self, video: "VideoInput", **kwargs):
        # num_threads
        num_threads = kwargs.get("frame_extract_num_threads")
        num_threads = int(num_threads)
        
        # Open the video file and get the total number of frames
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video}")
        
        # Calculate the frame indices to sample
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))        
        cap.release()
        
        frame_indices = self._get_cv2_video_sample_frames(video, total_frames=total_frames, **kwargs)
        
        # Initialize global storage and index mapping
        unique_frames = [None] * len(frame_indices)  # Storage for sampled frames
        index_map = {idx: pos for pos, idx in enumerate(frame_indices)}  # Map frame indices to storage positions
        
        # Split frame indices into chunks for multi-threaded processing
        chunks = np.array_split(frame_indices, min(num_threads, len(frame_indices)))

        # Define the worker function for each thread
        def worker(chunk_indices, chunk_id):
            # Each thread opens its own video reader
            local_cap = cv2.VideoCapture(video)
            
            # Set the starting position for the thread's assigned frames
            if chunk_indices[0] > 0:
                local_cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_indices[0])
            
            frame_idx_cursor = chunk_indices[0]  # Current position in the video frames
            chunk_cursor = 0  # Current position within the chunk
            
            while chunk_cursor < len(chunk_indices):
                target_idx = chunk_indices[chunk_cursor]
                ok = local_cap.grab()
                if not ok:
                    break
                
                if frame_idx_cursor == target_idx:
                    ret, frame = local_cap.retrieve()
                    if ret:
                        unique_pos = index_map[target_idx]
                        unique_frames[unique_pos] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format
                    chunk_cursor += 1
                frame_idx_cursor += 1
            
            local_cap.release()
        
        # Start multi-threaded processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                futures.append(executor.submit(worker, chunk, i))
            
            # Wait for all threads to complete
            for future in futures:
                future.result()
        
        # Construct the final result
        frames = [unique_frames[index_map[idx]] for idx in frame_indices]
        
        ## Raise frames with value of None:
        #  Case 1: When total_frames > actual_frames, the index may exceed the actual range.
        #  Case 2: When the video is corrupted.
        try:
            idx = next(i for i, item in enumerate(frames) if item is None)
            frames = frames[:idx]
        except StopIteration:
            pass
        
        validate_frame_sampling(frame_indices, frames)
        
        if not frames:
            return self.get_video_sample_frames(video)
        
        return frames, frame_indices
    
    def get_mm_inputs(
        self,
        batch_media_orders: List[List[int]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
        mllama_num_tiles: int,
        is_training: bool = False,
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        
        max_image_num = max([len(imgs) for imgs in images])
        max_video_num = max([len(vds) for vds in videos])
        
        batch_size = len(images)
        frame_num_per_video = [[] for _ in range(batch_size)]

        if max_image_num == 0 and max_video_num == 0:
            # When max_image_num == 0 and max_video_num == 0, it indicates that all the data in the batch consists purely of text data.
            # However, during training on VLLM, pure text data can cause NCCL timeout issues due to gradient synchronization problems in the vision encoder.
            # Therefore, we need to place a special image at the very beginning.
            if is_training:
                images[0] = [PURE_TEXT_BATCH]
                batch_media_orders[0] = [0]
            else:
                return {}, frame_num_per_video

        loaded_images = []
        loaded_images_idxs = []
        
        get_video_sample_frames_fun = self.get_video_sample_frames
        if getattr(processor, "extract_frame_func", "av") == "cv2":
            get_video_sample_frames_fun = self.get_cv2_video_sample_frames_multithread
        
        for idx, media_orders in enumerate(batch_media_orders):
            media_orders = np.array(media_orders)
            
            imgs = images[idx]
            vids = videos[idx]
            
            cur_loaded_images = []
            cur_loaded_images_from_image = []   # A temporary buffer used for storing image data
            cur_loaded_images_from_video = []   # A temporary buffer used for storing video frame data
            
            # Local variables used for video frame extraction.
            cur_loaded_video_frames = []
            cur_frame_num_per_video = []
            last_frame_indices = []
            
            if imgs:
                if len(imgs) == 1 and imgs[0] == PURE_TEXT_BATCH:
                    imgs = [WHITE_IMAGE]
                else:
                    try:
                        imgs = [Image.open(image) if isinstance(image, str) else self._base64_to_image(image["base64"]) for image in imgs]
                        for image in imgs:
                            image.load()
                    except Exception as e:
                        imgs = [WHITE_IMAGE] * len(imgs)
                        # We use frame_num_per_video to indicate whether there are any image loading failures in the current sample.
                        # 1. Specifically, we use -2 to denote that there are corrupted images in the current sample.
                        # 2. Since the processing of images occurs before videos, -2 can only appear at the very beginning of frame_num_per_video.
                        # 3. NOTE: If interleaved images and videos are supported in the future, the code here will need to be modified.
                        frame_num_per_video[idx].append(-2)
                        logger.info(f"Failed to fetch images, dropping this sample. Exception: {e}")
                            
                # imgs = [Image.open(image) if isinstance(image, str) else image for image in imgs]
                cur_loaded_images_from_image.extend(imgs)
            if vids:
                try:
                    process_vids = vids
                
                    # [Streaming Event Caption] mode
                    if vids[0].startswith(STREAMING_EVENT_CAPTION_MODE):
                        video_info = vids[0].replace(STREAMING_EVENT_CAPTION_MODE + "[INFO]", "").strip()
                        video_info = json.loads(video_info)
                        
                        video_path = video_info["video_path"]
                        time_info = video_info["time_info"]
                        
                        process_vids = [video_path]
                        
                        for vid in vids[1:]:
                            assert vid == STREAMING_EVENT_CAPTION_MODE + "[PLACEHOLDER]", "Invalid streaming event caption mode"
                    
                    # [RealTime QA] mode
                    elif vids[0].startswith(REAL_TIME_QA_MODE):
                        video_info = vids[0].replace(REAL_TIME_QA_MODE + "[INFO]", "").strip()
                        video_info = json.loads(video_info)
                        
                        video_path = video_info["video_path"]
                        time_info = video_info["time_info"]
                        
                        process_vids = [video_path]
                        
                        for vid in vids[1:]:
                            assert vid == REAL_TIME_QA_MODE + "[PLACEHOLDER]", "Invalid real-time QA mode"
                        
                    for video in process_vids:
                        video_frames, frame_indices = get_video_sample_frames_fun(
                            video,
                            video_fps=getattr(processor, "video_fps", 1.0),
                            video_minlen=getattr(processor, "video_minlen", 32),
                            video_maxlen=getattr(processor, "video_maxlen", 32),
                            frame_extract_num_threads=getattr(processor, "frame_extract_num_threads", 4),
                        )
                        cur_frame_num = len(video_frames)
                        last_frame_indices = frame_indices[:cur_frame_num]
                        
                        if len(video_frames) == 0:
                            raise ValueError(f"0 valid video frame is extracted from the video URL: {video}")
                        
                        cur_loaded_video_frames.extend(video_frames)
                        cur_frame_num_per_video.append(len(video_frames))
                    
                    # [Streaming Event Caption] mode
                    if vids[0].startswith(STREAMING_EVENT_CAPTION_MODE):
                        event_end_times = [self.time_to_seconds(time_range['end']) for time_range in time_info]
                        event_end_times = self.increment_adjacent(event_end_times)
                        
                        event_end_time_idxs = self.convert_seconds_to_frame_indices(process_vids[0], event_end_times)
                        cur_frame_num_per_video = self.split_list_by_boundaries(last_frame_indices, event_end_time_idxs)
                        assert len(cur_frame_num_per_video) == len(vids), "[Streaming Event Caption]: len(cur_frame_num_per_video) != len(vids)"
                        
                        total_frames_before_last_event_caption = sum(cur_frame_num_per_video)
                        cur_loaded_video_frames = cur_loaded_video_frames[:total_frames_before_last_event_caption]
                    
                    # [RealTime QA] mode
                    elif vids[0].startswith(REAL_TIME_QA_MODE):
                        event_end_times = [self.time_to_seconds(t) for t in time_info]
                        assert event_end_times, "RealTime QA error: event_end_times should not be empty."
                        assert event_end_times[0] == 0, "RealTime QA error: The first event end time must be 0."
                        assert all(event_end_times[i+1] - event_end_times[i] == 1 for i in range(len(event_end_times)-1)), "RealTime QA error: event end times must be strictly increasing by 1."
                        
                        valid_frame_num = len(event_end_times)
                        last_frame_indices = last_frame_indices[:valid_frame_num]
                        
                        event_end_time_idxs = self.convert_seconds_to_frame_indices(process_vids[0], event_end_times)
                        cur_frame_num_per_video = self.split_list_by_boundaries(last_frame_indices, event_end_time_idxs)
                        assert len(cur_frame_num_per_video) == len(vids), "[RealTime QA]: len(cur_frame_num_per_video) != len(vids)"
                        assert all(n == 1 for n in cur_frame_num_per_video), "[RealTime QA]: Each position must have exactly one frame."
                        
                        total_frames_before_last_event_caption = sum(cur_frame_num_per_video)
                        cur_loaded_video_frames = cur_loaded_video_frames[:total_frames_before_last_event_caption]
                        
                except Exception as e:
                    cur_loaded_video_frames = [WHITE_IMAGE] * len(vids)
                    cur_frame_num_per_video = [-1] * len(vids)
                    logger.info(f"Failed to fetch video, dropping this sample. Exception: {e}")                
                
                cur_loaded_images_from_video.extend(cur_loaded_video_frames)
                frame_num_per_video[idx].extend(cur_frame_num_per_video)
            
            # Assemble `cur_loaded_images` according to the order of appearance of images and videos.
            cur_image_idx = 0
            cur_video_idx = 0
            cur_video_end_idx_list = np.cumsum(np.abs(cur_frame_num_per_video)).tolist()
            
            if cur_video_end_idx_list:
                assert cur_video_end_idx_list[-1] == len(cur_loaded_images_from_video), "`cur_frame_num_per_video` NOT match `cur_loaded_images_from_video`"
            
            for media_tag in media_orders:
                if media_tag == 0:
                    cur_loaded_images.append(cur_loaded_images_from_image[cur_image_idx])
                    cur_image_idx += 1
                elif media_tag == 1:
                    start_idx = cur_video_end_idx_list[cur_video_idx - 1] if cur_video_idx > 0 else 0
                    end_idx = cur_video_end_idx_list[cur_video_idx]
                    cur_loaded_images.extend(cur_loaded_images_from_video[start_idx:end_idx])
                    cur_video_idx += 1
                else:
                    raise NotImplementedError(f"Invalid media tag: {media_tag}")
                    
            if cur_loaded_images:
                loaded_images.append(cur_loaded_images)
                loaded_images_idxs.append(idx)
        
        image_features = processor.image_processor(loaded_images, max_image_tiles=mllama_num_tiles)
        num_tiles = image_features.pop("num_tiles")
        
        image_features = {k: v if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in image_features.items()}
        image_features["num_tiles"] = num_tiles
        if batch_size == image_features['pixel_values'].shape[0]:
            return image_features, frame_num_per_video

        _, max_num_images, max_image_tiles, channels, tile_height, tile_width = image_features['pixel_values'].shape
        # recover images
        stacked_images = torch.zeros(
            (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width),
            dtype=torch.float32,
        )
        stacked_images[loaded_images_idxs] = image_features['pixel_values']
        image_features['pixel_values'] = stacked_images
        # recover aspect_ratio_ids
        aspect_ratio_ids = torch.zeros((batch_size, max_num_images), dtype=torch.int64)
        aspect_ratio_ids[loaded_images_idxs] = image_features['aspect_ratio_ids']
        image_features['aspect_ratio_ids'] = aspect_ratio_ids
        # recover aspect_ratio_mask
        aspect_ratio_mask = torch.zeros((batch_size, max_num_images, max_image_tiles), dtype=torch.int64)
        ## copied from transformers.mllama.build_aspect_ratio_mask()
        # Set the first tile to 1 for all aspect ratios
        # because in original implementation aspect ratios are padded with (1, 1),
        # but original code examples are not built to handle batches, so we might remove it later
        aspect_ratio_mask[:, :, 0] = 1
        aspect_ratio_mask[loaded_images_idxs] = image_features['aspect_ratio_mask']
        image_features['aspect_ratio_mask'] = aspect_ratio_mask
        # recover num_tiles
        stacked_num_tiles = [[] for _ in range(batch_size)]
        for idx, n_tiles in zip(loaded_images_idxs, num_tiles):
            stacked_num_tiles[idx] = n_tiles
        image_features["num_tiles"] = stacked_num_tiles

        return image_features, frame_num_per_video
    
    def convert_seconds_to_frame_indices(self, video_path: "VideoInput", timestamp_seconds: list[float]) -> list[int]:
        container = av.open(video_path, "r")
        video_stream = next(stream for stream in container.streams if stream.type == "video")
        frame_rate = float(video_stream.average_rate)
        frame_indices = [int(timestamp * frame_rate) for timestamp in timestamp_seconds]
        return frame_indices
    
    def time_to_seconds(self, time_str: str) -> float:
        """ Convert time format to seconds """
        if isinstance(time_str, (int, float)):
            return float(time_str)
        
        if ':' in str(time_str):
            parts = str(time_str).split(':')
            if len(parts) == 3:  # HH:MM:SS
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:  # MM:SS
                return int(parts[0]) * 60 + float(parts[1])
        
        return float(time_str)
    
    def increment_adjacent(self, second_times, increment=1):
        """
        Increment the next element by `increment` if two adjacent elements are equal.
        Continue comparing with the incremented value.
        Args:
            second_times (list of float): Input list to process.
            increment (float, optional): The value to increment by. Default is 1.
        Returns:
            list of float: The processed list.
        """
        i = len(second_times) - 1
        while i > 0:
            if second_times[i] == second_times[i - 1]:
                second_times[i] += increment
            i -= 1
        return second_times
    
    def split_list_by_boundaries(self, data_list, boundary_list):
        """
        Split the first list based on boundary values and return the lengths of each segment.
        
        Args:
            data_list: List to be split, monotonically increasing
            boundary_list: Boundary list, monotonically increasing
        
        Returns:
            List of integers representing the length of each split segment
        """
        result = []
        start_idx = 0
        
        for boundary in boundary_list:
            # Find the position of the first element greater than boundary
            end_idx = start_idx
            while end_idx < len(data_list) and data_list[end_idx] <= boundary:
                end_idx += 1
            
            # Add the length of current segment
            segment_length = end_idx - start_idx
            result.append(segment_length)
            
            start_idx = end_idx
        
        return result


PLUGINS = {
    "base": BasePlugin,
    "llava": LlavaPlugin,
    "paligemma": PaliGemmaPlugin,
    "qwen2_vl": Qwen2vlPlugin,
    "mllama": MllamaPlugin,
}


def get_mm_plugin(
    name: str,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
) -> "BasePlugin":
    plugin_class = PLUGINS.get(name, None)
    if plugin_class is None:
        raise ValueError("Multimodal plugin `{}` not found.".format(name))

    return plugin_class(image_token, video_token)
