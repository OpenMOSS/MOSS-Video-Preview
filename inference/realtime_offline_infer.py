"""
Offline (non-streaming) inference for video understanding using the
`video_mllama_real_time` checkpoint.

This script mirrors the structure and CLI style of `realtime_streaming_infer.py`,
but calls the model's `offline_generate` entrypoint instead of `real_time_generate`.

Core behavior is adapted from:
- `/resources/video_hf/models/video_mllama_real_time/test_offline.py`
"""

import argparse
import os
import queue
import threading
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from transformers import AutoProcessor, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(checkpoint: str):
    """Load model and processor (same pattern as other inference scripts)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {checkpoint} on {device}...", flush=True)
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        frame_extract_num_threads=1,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    return model, processor


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Offline inference for video understanding.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Model checkpoint path. "
            "Env: CHECKPOINT. "
            "No default: MUST be provided either via CLI or env."
        ),
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help=(
            "Video file path. "
            "Env: VIDEO_PATH. "
        ),
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help=(
            "Image file path. "
            "Env: IMAGE_PATH. "
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text. Env: PROMPT. Default: 'Describe the video.' (or 'Describe the image.' if image is provided)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Default: 1.0",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling. Default: 50",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling. Default: 1.0",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Max new tokens. Default: 1024",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty. Default: 1.0",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 4. Main offline-generate flow
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()

    checkpoint = args.checkpoint or os.environ.get("CHECKPOINT")
    video_path = args.video_path or os.environ.get("VIDEO_PATH")
    image_path = args.image_path or os.environ.get("IMAGE_PATH")

    if checkpoint is None:
        raise ValueError("Checkpoint must be provided via --checkpoint or CHECKPOINT env.")

    if video_path and image_path:
        raise ValueError("Only one of --video_path or --image_path can be provided.")
    if not video_path and not image_path:
        raise ValueError("Either --video_path or --image_path must be provided.")

    if video_path:
        media_path = video_path
        media_type = "video"
    else:
        media_path = image_path
        media_type = "text_image"

    default_prompt = f"Describe the {'image' if media_type == 'text_image' else 'video'}."
    prompt = args.prompt or os.environ.get("PROMPT") or default_prompt

    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    max_new_tokens = args.max_new_tokens
    repetition_penalty = args.repetition_penalty

    if not os.path.isfile(media_path):
        raise FileNotFoundError(f"{media_type.capitalize()} file not found: {media_path}")

    print("Configuration:", flush=True)
    print(f"- device: {device}")
    print(f"- checkpoint: {checkpoint}")
    if video_path:
        print(f"- video_path: {video_path}")
    else:
        print(f"- image_path: {image_path}")
    print(f"- prompt: {prompt}")
    print(
        f"- temperature={temperature}, top_k={top_k}, "
        f"top_p={top_p}, max_new_tokens={max_new_tokens}, "
        f"repetition_penalty={repetition_penalty}",
        flush=True,
    )

    print(
        "[Reminder] This script is ONLY for realtime-SFT checkpoints' offline inference "
        "(e.g. models with realtime/streaming fine-tuning). "
        "Do NOT use it for base models or plain SFT (non-realtime) checkpoints.",
        flush=True,
    )

    # 1. Load model & processor
    model, processor = load_model_and_processor(checkpoint)

    # 2. Prepare queues (follow test_offline.py semantics)
    new_queries: "queue.Queue[dict]" = queue.Queue()
    output_text_queue: "queue.Queue[str]" = queue.Queue()

    media_kwargs = {
        "video_fps": 1.0,
        "video_minlen": 8,
        "video_maxlen": 256,
    }

    generate_kwargs = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }

    # 添加推理任务
    query = {
        "prompt": f"\n{prompt}",
        "images": [],
        "videos": [video_path] if video_path else [],
        "media_kwargs": media_kwargs,
        "thinking_mode": "no_thinking",
        "system_prompt_type": media_type,
        "generate_kwargs": generate_kwargs,
        "stop_offline_generate": False,
    }
    if image_path:
        from PIL import Image
        query["images"] = [Image.open(image_path).convert("RGB")]

    new_queries.put(query)

    # 发送停止信号，让模型在完成当前任务后退出循环
    new_queries.put({"stop_offline_generate": True})

    # 3. 启动输出打印线程
    def print_output():
        print("\n--- 模型输出开始 ---", flush=True)
        while True:
            try:
                token = output_text_queue.get()
                if token == "<|round_end|>":
                    print("\n--- 模型输出结束 ---", flush=True)
                    break
                print(token, end="", flush=True)
            except Exception as e:
                print(f"\nError in output thread: {e}", flush=True)
                break

    output_thread = threading.Thread(target=print_output, daemon=True)
    output_thread.start()

    print("Running offline_generate...", flush=True)
    try:
        model.offline_generate(
            processor,
            new_queries,
            output_text_queue,
            vision_chunked_length=64,
        )
    except Exception as e:
        print(f"\nError during generation: {e}", flush=True)
    finally:
        # 等待打印线程结束
        output_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()

