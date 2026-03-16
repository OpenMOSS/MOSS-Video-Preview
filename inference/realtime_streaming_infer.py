"""
Streaming inference for video understanding.

Integrates the real-time (streaming) model flow from videollm-backend_new:
- Uses real_time_generate() with image_queue, prompt_queue, token_queue.
- Feeds local video frames into the image queue and a text prompt into the prompt queue.
- Consumes generated tokens from the token queue and prints them in real time.

IMPORTANT:
- This script requires a **realtime-SFT** checkpoint (a model trained with streaming/realtime capability),
  e.g. `models/moss-video-realtime-sft`. Offline/SFT-only checkpoints may not provide `real_time_generate()`.

Format and flow follow infer_offline.py where applicable.
"""
import os
import argparse
import os
import queue
import threading
import time

import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(checkpoint: str):
    """Load model and processor (same pattern as infer_offline.py)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {checkpoint}...")
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
# 3. Video frame feeder (simulate streaming input from local video)
# ---------------------------------------------------------------------------

def video_feeder(video_path: str, image_queue: queue.Queue, fps: float = 1.0):
    """
    Read a local video file, extract frames at `fps` rate, convert to PIL Images,
    and put them into `image_queue`. Runs in a separate thread.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(video_fps / fps)))
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                # print(f"Feeding frame {frame_idx} to model...", flush=True)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                image_queue.put(pil_image)
                # Control feeding speed to match the desired FPS
                time.sleep(1.0 / fps)
            frame_idx += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# 4. Output listener (consume token queue and print)
# ---------------------------------------------------------------------------

def output_listener(
    token_queue: queue.Queue,
    idle_timeout_sec: float,
    stop_event: threading.Event = None,
):
    """
    Read from `token_queue` and print tokens in real time.
    Stops when receiving a known end marker or after `idle_timeout_sec` with no new token
    (after at least one token has been received), or when `stop_event` is set.
    """
    received_any = False
    last_get_time = time.time()
    end_markers = {"[DONE]", "[ERROR]", "<|round_end|>"}
    in_silence = False
    pending_token = None

    def _process_token(t: str):
        nonlocal in_silence
        if t == "<|silence|>":
            if not in_silence:
                print("\n" + "-"*30 + " [Silence / Observing] " + "-"*30, flush=True)
                in_silence = True
            return
            
        if in_silence:
            in_silence = False
            
        print(t, end="", flush=True)

    while True:
        if stop_event and stop_event.is_set():
            if pending_token is not None:
                _process_token(pending_token)
            break
            
        try:
            token = token_queue.get(timeout=0.1)
        except queue.Empty:
            if pending_token is not None:
                _process_token(pending_token)
                pending_token = None
            if received_any and (time.time() - last_get_time) > idle_timeout_sec:
                break
            continue
        
        # 只要收到任何 token（包括静默符），就更新计时器，表示模型还在运行
        last_get_time = time.time()
        received_any = True

        if token == "<|round_start|>":
            # 关键修复：如果收到新的 round_start，说明之前的 token（如由于空 prompt 产生的 "How" 或 "I"）被用户输入打断了，直接丢弃
            pending_token = None
            continue
            
        if token in end_markers:
            if pending_token is not None:
                _process_token(pending_token)
                pending_token = None
            break
            
        if pending_token is not None:
            _process_token(pending_token)
            
        pending_token = token

    print(flush=True)


# ---------------------------------------------------------------------------
# 5. Main: wire queues, start generation and listener
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Streaming inference for video understanding.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/moss-video-realtime-sft",
        help="Model checkpoint path (MUST be a realtime-SFT checkpoint).",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="data/example_video.mp4",
        help="Video file path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the video.",
        help="Prompt text.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=86400,
        help="Max new tokens per turn.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Video feed FPS.",
    )
    parser.add_argument(
        "--idle_timeout",
        type=float,
        default=5.0,
        help="Idle timeout in seconds for the output listener.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = args.checkpoint
    video_path = args.video_path
    prompt = args.prompt
    max_new_tokens = args.max_new_tokens
    video_feed_fps = args.fps
    idle_timeout = args.idle_timeout

    print(
        "[Reminder] Streaming inference requires a realtime-SFT checkpoint (e.g. `models/moss-video-realtime-sft`).",
        flush=True,
    )

    # 2. Load model and processor
    model, processor = load_model_and_processor(checkpoint)

    # 3. Prepare queues (same roles as in videollm-backend_new model_pool.py)
    image_queue = queue.Queue()
    prompt_queue = queue.Queue()
    token_queue = queue.Queue()

    # 4. Prepare inputs: prompt + video path
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 5. Start video feeder thread (fill image_queue from local video)
    feeder_thread = threading.Thread(
        target=video_feeder,
        args=(video_path, image_queue),
        kwargs={"fps": video_feed_fps},
        daemon=True,
    )
    feeder_thread.start()

    # Wait 1 second before feeding the prompt
    print("Waiting 1 second before sending prompt...", flush=True)
    time.sleep(1.0)
    print(f"Sending prompt: {prompt}", flush=True)
    prompt_queue.put(prompt)

    # 6. Start real_time_generate in a background thread
    gen_exception = []

    def run_generation():
        try:
            if hasattr(model, "real_time_generate"):
                model.real_time_generate(
                    image_queue,
                    prompt_queue,
                    token_queue,
                    processor,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    # repetition_penalty=1.2,
                )
            else:
                gen_exception.append(RuntimeError("Model has no real_time_generate"))
        except Exception as e:
            gen_exception.append(e)

    gen_thread = threading.Thread(target=run_generation, daemon=True)
    gen_thread.start()

    # 7. Consume token queue and print (streaming output)
    print("Generating (streaming)...")
    output_listener(token_queue, idle_timeout_sec=idle_timeout)
    
    # Check if video feeder is still alive
    if not feeder_thread.is_alive():
        print("\n[Video processing completed: reached end of file]")
    else:
        print("\n[Inference stopped: idle timeout or end marker received]")
        
    print("=" * 50)

    # Stop generation so the background thread can exit
    if hasattr(model, "stop_real_time_generate"):
        model.stop_real_time_generate()
    gen_thread.join(timeout=3.0)

    if gen_exception:
        raise gen_exception[0]


if __name__ == "__main__":
    main()
