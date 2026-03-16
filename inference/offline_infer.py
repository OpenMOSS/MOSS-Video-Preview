

import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline video inference for MOSS video models. "
            "This script is intended for base and SFT checkpoints, "
            "and is not suitable for realtime SFT models."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/moss-video-sft",
        help="Model checkpoint directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="data/example_video.mp4",
        help="Path to input video file.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to input image file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the video.",
        help="User prompt/question about the video or image.",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=1.0,
        help="FPS used for video frame sampling.",
    )
    parser.add_argument(
        "--video_minlen",
        type=int,
        default=8,
        help="Minimum number of sampled frames.",
    )
    parser.add_argument(
        "--video_maxlen",
        type=int,
        default=16,
        help="Maximum number of sampled frames.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system prompt to prepend as a system message.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    # 1. Setup environment and device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Load model and processor
    checkpoint = args.checkpoint
    
    print(
        "Note: infer_offline.py is intended for base and SFT checkpoints, "
        "and is not suitable for realtime SFT models."
    )
    print(f"Loading model from {checkpoint}...")
    processor = AutoProcessor.from_pretrained(
        checkpoint, 
        trust_remote_code=True, 
        frame_extract_num_threads=1
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, 
        trust_remote_code=True, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    )
    
    # 3. Prepare inputs
    video_path = args.video_path
    image_path = args.image_path

    if video_path and image_path:
        raise ValueError("Both video_path and image_path are provided. Please provide only one.")
    if not video_path and not image_path:
        raise ValueError("Neither video_path nor image_path is provided. Please provide one.")

    prompt = args.prompt
    if prompt is None:
        prompt = "Describe the video." if video_path else "Describe the image."

    system_prompt = args.system_prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    if video_path:
        content = [
            {"type": "video"},
            {"type": "text", "text": prompt},
        ]
    else:
        content = [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]

    messages.append(
        {
            "role": "user",
            "content": content,
        }
    )
    
    # Apply chat template
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True,tokenize=False)
    
    # Process inputs
    print("Processing inputs...")
    if video_path:
        # Video processing arguments
        video_args = {
            "video_fps": args.video_fps,
            "video_minlen": args.video_minlen,
            "video_maxlen": args.video_maxlen,
        }
        inputs = processor(
            text=input_text,
            videos=[video_path],
            **video_args,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
    else:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=input_text,
            images=[image],
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
    
    # 4. Generate response
    print("Generating response...")
    with torch.no_grad():
        model_output = model.generate(
            **inputs, 
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0
        )
    
    # 5. Decode and print result
    decoded_output = processor.decode(model_output[0], skip_special_tokens=False)
    print("\n" + "="*50)
    print("Model Output:")
    print("="*50)
    print(decoded_output)
    print("="*50)

if __name__ == "__main__":
    main()
