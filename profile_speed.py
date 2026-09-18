import argparse
import logging
import os
import random
import time

import numpy as np
import torch

from core.models import build_model, model_label, normalize_model_size, normalize_version
from core.utils.utils import InputPadder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_inputs(height, width, device, pad):
    left = torch.randint(0, 256, (1, 3, height, width), dtype=torch.float32, device=device)
    right = torch.randint(0, 256, (1, 3, height, width), dtype=torch.float32, device=device)

    original_size = (height, width)
    padded_size = original_size
    if pad:
        padder = InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)
        padded_size = tuple(left.shape[-2:])

    return left, right, original_size, padded_size


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark(model, left, right, warmup, total, max_disp, use_amp):
    timings = []
    amp_enabled = use_amp and left.device.type == "cuda"

    for _ in range(total):
        sync_if_needed(left.device)
        start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
            _ = model(left, right, max_disp=max_disp, test_mode=True)
        sync_if_needed(left.device)
        timings.append(time.perf_counter() - start)

    measured = np.array(timings[warmup:], dtype=np.float64)
    return {
        "mean_ms": float(measured.mean() * 1000.0),
        "median_ms": float(np.median(measured) * 1000.0),
        "fps": float(1.0 / measured.mean()),
        "count": int(len(measured)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LiteAnyStereo LAS1 or LAS2 inference speed.")
    parser.add_argument("--version", default="las1", help="model version: las1/v1 or las2/v2")
    parser.add_argument("--model_size", "--model-size", default=None, help="LAS2 model size: s, m, l, or h")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_pad", action="store_true", help="disable InputPadder(divis_by=32)")
    parser.add_argument("--no_amp", action="store_true", help="disable CUDA autocast during benchmarking")
    parser.add_argument("--compile", action="store_true", help="benchmark a torch.compile(model) copy")
    return parser.parse_args()


def main():
    args = parse_args()
    version = normalize_version(args.version)
    model_size = normalize_model_size(version, args.model_size)
    label = model_label(version, model_size)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count())
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    else:
        torch.backends.cudnn.benchmark = True

    if args.total <= args.warmup:
        raise ValueError(f"--total ({args.total}) must be larger than --warmup ({args.warmup}).")

    model = build_model(version, fnet_pretrained=False, model_size=model_size, max_disp=args.max_disp).to(device).eval()
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    left, right, original_size, padded_size = build_inputs(args.height, args.width, device, pad=not args.no_pad)

    logging.info("Model: %s", label)
    logging.info("Checkpoint: none (random initialization)")
    logging.info("Device: %s", device)
    logging.info("Params: %.2f M", count_parameters(model) / 1e6)
    logging.info("Input size: %sx%s", original_size[0], original_size[1])
    logging.info("Padded size: %sx%s", padded_size[0], padded_size[1])
    logging.info(
        "Warmup: %d, total: %d, max_disp: %d, amp: %s",
        args.warmup,
        args.total,
        args.max_disp,
        str(not args.no_amp and device.type == "cuda").lower(),
    )

    stats = benchmark(
        model=model,
        left=left,
        right=right,
        warmup=args.warmup,
        total=args.total,
        max_disp=args.max_disp,
        use_amp=not args.no_amp,
    )

    logging.info(
        "Average runtime: %.2f ms | Median runtime: %.2f ms | FPS: %.2f | Measured iters: %d",
        stats["mean_ms"],
        stats["median_ms"],
        stats["fps"],
        stats["count"],
    )


if __name__ == "__main__":
    main()
