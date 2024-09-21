import argparse
import os

import torch
from PIL import Image
from diffusers.image_processor import VaeImageProcessor

from model.pipeline import CatVTONPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--seed", type=int, default=555, help="A seed for reproducible evaluation."
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps to perform.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="The scale of classifier-free guidance for inference.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

@torch.no_grad()
def main():
    args = parse_args()
    attn_ckpt = "checkpoint"
    base_ckpt = "inpainting"
    image = Image.open("input/person.jpg")
    refer_image = Image.open("input/clothe.jpg")
    mask = Image.open("input/mask.png")
    vae_processor = VaeImageProcessor(vae_scale_factor=8)
    mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
    mixed_precision = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.mixed_precision]

    # Pipeline
    pipeline = CatVTONPipeline(
        attn_ckpt=attn_ckpt,
        base_ckpt=base_ckpt,
        weight_dtype=mixed_precision,
        device="cuda"
    )

    # Inference
    image = vae_processor.preprocess(image, 1028, 768)[0]
    refer_image = vae_processor.preprocess(refer_image, 1028, 768)[0]
    mask = mask_processor.preprocess(mask, 1028, 768)[0]
    generator = torch.Generator(device='cuda').manual_seed(args.seed)

    results = pipeline(
        image=image,
        condition_image=refer_image,
        mask=mask,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
    )

    for i, result in enumerate(results):
        output_path = os.path.join(args.output_dir, f'result_{i}.png')
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        result.save(output_path)


if __name__ == "__main__":
    main()
