"""Analyze an image using Qwen3-VL (vision-language model)."""

import argparse
import os
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def analyze_image(image_path: str, prompt: str, model_name: str):
    print(f"Loading model {model_name} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    pil_image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1024)

    # Strip the input tokens to get only the generated response
    generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("\n--- Response ---")
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze an image with Qwen3-VL")
    parser.add_argument("--image", type=str,
                        default="data/test_images/kitchen.png",
                        help="Path to image file")
    parser.add_argument("--prompt", type=str,
                        default="Describe this image in detail. What objects do you see and where are they located?",
                        help="Question or prompt about the image")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen3-VL-2B-Instruct",
                        help="HuggingFace model name")
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    analyze_image(image_path, args.prompt, args.model)
