import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import time
import torch
import numpy as np
import argparse
from typing import List, Dict, Any
from transformers import AutoProcessor, AutoModelForCausalLM


def print_video_basic_info(video_path: str) -> None:
    """Print basic video information (resolution, bitrate, duration) without external binaries."""
    if not video_path:
        return
    if not os.path.exists(video_path):
        print(f"[Video Info] File not found: {video_path}")
        return

    file_size_bytes = os.path.getsize(video_path)
    file_size_mb = file_size_bytes / (1024 * 1024) if file_size_bytes else 0.0

    width = height = None
    duration = None

    # Try to use OpenCV if available for resolution & duration
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps > 0 and frame_count > 0:
                duration = frame_count / fps
        cap.release()
    except Exception as e:
        print(f"[Video Info] OpenCV probing failed ({type(e).__name__}): {e}")

    # Approximate bitrate from file size and duration if possible
    bitrate_kbps = None
    if duration and duration > 0:
        bitrate_kbps = (file_size_bytes * 8 / duration) / 1000

    print("\n[Video Info]")
    print(f"  Path      : {video_path}")
    print(f"  File Size: {file_size_mb:.2f} MB")
    if width and height:
        print(f"  Resolution: {width}x{height}")
    if duration:
        print(f"  Duration  : {duration:.2f} s")
    if bitrate_kbps:
        print(f"  Bitrate   : {bitrate_kbps:.2f} kbps (approx)")
    print("")


class MossVideoBenchmarker:
    def __init__(self, model_path: str, device: str = "cuda:0", dtype=torch.bfloat16):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        
        print(f"Loading MOSS-Video model and processor from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True,
            frame_extract_num_threads=1
        )
        
        # MOSS-Video typically uses AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=self.device,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2"
        ).eval()

    def run_benchmark(self, prompt: str, video_path: str = None, num_runs: int = 5, 
                      max_new_tokens: int = 512, video_fps: float = 1.0, 
                      video_minlen: int = 8, video_maxlen: int = 16):
        
        print(f"\nStarting benchmark for {self.model_path}")
        print(f"Runs: {num_runs}, Max New Tokens: {max_new_tokens}")
        if video_path:
            print(f"Video: {video_path}")
            print(f"Video Args: fps={video_fps}, minlen={video_minlen}, maxlen={video_maxlen}")
            print_video_basic_info(video_path)
        
        # Prepare messages in MOSS-Video format
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful AI assistant specializing in video analysis. Respond to the user's request based on the provided video content."
            },
            {
                "role": "user", 
                "content": [
                    {"type": "video"} if video_path else None,
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        # Filter out None if no video
        messages[1]["content"] = [c for c in messages[1]["content"] if c is not None]

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Video processing args
        video_args = {"video_fps": video_fps, "video_minlen": video_minlen, "video_maxlen": video_maxlen} if video_path else {}
        
        print("Processing inputs...")
        inputs = self.processor(
            text=input_text,
            videos=[video_path] if video_path else None,
            **video_args,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)

        ttfts = []
        tpss = []
        total_latencies = []
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            self.model.generate(**inputs, max_new_tokens=10, do_sample=False)

        print("Running benchmark...")
        for i in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                # 1. Prefill / Time to First Token (TTFT)
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=1, 
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                torch.cuda.synchronize()
                ttft = time.perf_counter() - start_time
                ttfts.append(ttft)
                
                # 2. Full generation
                gen_start = time.perf_counter()
                full_output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                torch.cuda.synchronize()
                gen_end = time.perf_counter()
                
                # Decode and print model output for the first run or every run
                generated_ids = full_output[:, inputs.input_ids.shape[1]:]
                output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                total_latency = gen_end - start_time
                total_latencies.append(total_latency)
                
                # Calculate TPS (Tokens Per Second)
                num_gen_tokens = full_output.shape[1] - inputs.input_ids.shape[1]
                # TPS = (Total Tokens - 1) / (Total Gen Time - TTFT)
                if num_gen_tokens > 1:
                    tps = (num_gen_tokens - 1) / (gen_end - gen_start - ttft) if (gen_end - gen_start - ttft) > 0 else 0
                    tpss.append(tps)
                else:
                    tps = 0
                
            print(f"Run {i+1}: TTFT={ttft:.4f}s, TPS={tps:.2f}, Total={total_latency:.4f}s, Tokens={num_gen_tokens}")
            if i == 0: # Print output only for the first run to save space
                print(f"\nSample Output:\n{'-'*20}\n{output_text}\n{'-'*20}\n")

        print("\n" + "="*50)
        print(f"MOSS-VIDEO BENCHMARK RESULTS: {self.model_path}")
        print("="*50)
        print(f"Average TTFT: {np.mean(ttfts):.4f} s")
        if tpss:
            print(f"Average TPS: {np.mean(tpss):.2f} tokens/s")
        print(f"Average Total Latency: {np.mean(total_latencies):.4f} s")
        print(f"P95 TTFT: {np.percentile(ttfts, 95):.4f} s")
        print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOSS-Video Local Benchmark Script")
    parser.add_argument("--model", type=str, default="models/moss-video-sft", help="Path to model checkpoint")
    # parser.add_argument("--model", type=str, default="/inspire/hdd/project/video-understanding/public/personal/jihuai/real_time_chat/models/0913_1_1_1_w_1_0", help="Path to model checkpoint")

    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--prompt", type=str, default="Describe the video in detail.")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--fps", type=float, default=1.0, help="Video FPS for sampling")
    parser.add_argument("--min_frames", type=int, default=8, help="Minimum number of frames to sample")
    parser.add_argument("--max_frames", type=int, default=16, help="Maximum number of frames to sample")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0)")
    
    args = parser.parse_args()
    
    # Set CUDA device if specified in environment or args
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1] if ":" in args.device else "0"
    
    benchmarker = MossVideoBenchmarker(args.model, device=args.device)
    benchmarker.run_benchmark(
        args.prompt, 
        args.video, 
        args.runs, 
        args.tokens,
        video_fps=args.fps,
        video_minlen=args.min_frames,
        video_maxlen=args.max_frames
    )
