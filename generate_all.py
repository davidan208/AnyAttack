"""
AnyAttack - Generate adversarial images for all checkpoints.
Python 3.12 compatible. Supports resume via checking existing output files.

Usage:
    python generate_all.py [--device mps|cuda:0|cpu] [--batch_size 10] [--eps 0.0627]

Structure:
    Clean images:  resources/images/bigscale_1000/nips17/{0..999}.png
    Target images: resources/images/target_images_1000/1/{0..999}.jpg
    Checkpoints:   checkpoints/*.pt
    Output:        outputs/{checkpoint_name}/{0..999}.png
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import torch
import torchvision
from torchvision import transforms
from PIL import Image

# Add project root to path for model imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model import CLIPEncoder, Decoder


def parse_args():
    parser = argparse.ArgumentParser(description="AnyAttack batch adversarial image generation")
    parser.add_argument("--eps", type=float, default=16.0 / 255,
                        help="Perturbation budget (L-inf), default 16/255")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of images per batch")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda:0, mps, or cpu. Auto-detected if omitted.")
    parser.add_argument("--clean_dir", type=str,
                        default=str(PROJECT_ROOT / "resources" / "images" / "bigscale_1000" / "nips17"),
                        help="Directory of clean images")
    parser.add_argument("--target_dir", type=str,
                        default=str(PROJECT_ROOT / "resources" / "images" / "target_images_1000" / "1"),
                        help="Directory of target images")
    parser.add_argument("--checkpoint_dir", type=str,
                        default=str(PROJECT_ROOT / "checkpoints"),
                        help="Directory containing .pt checkpoint files")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "outputs"),
                        help="Base output directory")
    parser.add_argument("--num_images", type=int, default=1000,
                        help="Total number of images (0-indexed)")
    return parser.parse_args()


def get_device(requested: str | None) -> torch.device:
    """Auto-detect best available device."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image(path: str, transform: transforms.Compose) -> torch.Tensor:
    """Load a single image and apply transform."""
    img = Image.open(path).convert("RGB")
    return transform(img)


def load_decoder(checkpoint_path: str, device: torch.device) -> Decoder:
    """Load decoder from checkpoint, handling DDP 'module.' prefix."""
    decoder = Decoder(embed_dim=512).to(device).eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "decoder_state_dict" in state_dict:
        sd = state_dict["decoder_state_dict"]
    else:
        sd = state_dict

    # Strip 'module.' prefix if saved from DDP
    new_sd = {}
    for k, v in sd.items():
        name = k[7:] if k.startswith("module.") else k
        new_sd[name] = v

    decoder.load_state_dict(new_sd)
    return decoder


def get_pending_indices(output_dir: str, num_images: int) -> list[int]:
    """Return sorted list of indices that haven't been generated yet (resume support)."""
    pending = []
    for idx in range(num_images):
        out_path = os.path.join(output_dir, f"{idx}.png")
        if not os.path.exists(out_path):
            pending.append(idx)
    return pending


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Validate directories
    if not os.path.isdir(args.clean_dir):
        print(f"ERROR: Clean image directory not found: {args.clean_dir}")
        sys.exit(1)
    if not os.path.isdir(args.target_dir):
        print(f"ERROR: Target image directory not found: {args.target_dir}")
        sys.exit(1)
    if not os.path.isdir(args.checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {args.checkpoint_dir}")
        sys.exit(1)

    # Find all checkpoints
    checkpoint_paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")))
    if not checkpoint_paths:
        print(f"ERROR: No .pt files found in {args.checkpoint_dir}")
        sys.exit(1)

    print(f"Found {len(checkpoint_paths)} checkpoint(s):")
    for cp in checkpoint_paths:
        print(f"  - {os.path.basename(cp)}")

    # Image transform (same as demo.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load CLIP encoder once (shared across all checkpoints)
    print(f"\nLoading CLIP ViT-B/32 encoder...")
    clip_model = CLIPEncoder("ViT-B/32").to(device)
    clip_model.eval()

    # Determine clean/target file extensions
    clean_ext = ".png"
    target_ext = ".jpg"

    # Process each checkpoint
    for ckpt_path in checkpoint_paths:
        ckpt_name = Path(ckpt_path).stem  # e.g. "coco_cos"
        output_dir = os.path.join(args.output_dir, ckpt_name)
        os.makedirs(output_dir, exist_ok=True)

        # Check which indices still need to be generated
        pending = get_pending_indices(output_dir, args.num_images)
        if not pending:
            print(f"\n[{ckpt_name}] All {args.num_images} images already generated. Skipping.")
            continue

        print(f"\n[{ckpt_name}] Loading decoder from: {os.path.basename(ckpt_path)}")
        decoder = load_decoder(ckpt_path, device)

        print(f"[{ckpt_name}] {len(pending)} images pending (resume from index {pending[0]})")

        # Process in batches
        for batch_start in range(0, len(pending), args.batch_size):
            batch_indices = pending[batch_start:batch_start + args.batch_size]

            # Load batch of clean and target images
            clean_batch = []
            target_batch = []
            valid_indices = []

            for idx in batch_indices:
                clean_path = os.path.join(args.clean_dir, f"{idx}{clean_ext}")
                target_path = os.path.join(args.target_dir, f"{idx}{target_ext}")

                if not os.path.exists(clean_path):
                    print(f"  WARNING: Clean image not found: {clean_path}, skipping index {idx}")
                    continue
                if not os.path.exists(target_path):
                    print(f"  WARNING: Target image not found: {target_path}, skipping index {idx}")
                    continue

                clean_batch.append(load_image(clean_path, transform))
                target_batch.append(load_image(target_path, transform))
                valid_indices.append(idx)

            if not valid_indices:
                continue

            # Stack into tensors
            clean_tensor = torch.stack(clean_batch).to(device)
            target_tensor = torch.stack(target_batch).to(device)

            # Generate adversarial images
            with torch.no_grad():
                img_emb = clip_model.encode_img(target_tensor)
                noise = decoder(img_emb)
                noise = torch.clamp(noise, -args.eps, args.eps)
                adv_images = torch.clamp(clean_tensor + noise, 0, 1)

            # Save each image with its original index name
            for i, idx in enumerate(valid_indices):
                out_path = os.path.join(output_dir, f"{idx}.png")
                torchvision.utils.save_image(adv_images[i], out_path)

            # Progress
            done_count = batch_start + len(valid_indices)
            total_pending = len(pending)
            print(f"  [{ckpt_name}] Progress: {done_count}/{total_pending} "
                  f"(indices {valid_indices[0]}-{valid_indices[-1]})")

        # Final verification
        remaining = get_pending_indices(output_dir, args.num_images)
        if remaining:
            print(f"[{ckpt_name}] WARNING: {len(remaining)} images still missing after processing.")
        else:
            print(f"[{ckpt_name}] Done! All {args.num_images} adversarial images saved to: {output_dir}")

        # Free decoder memory
        del decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== All checkpoints processed ===")


if __name__ == "__main__":
    main()
