import argparse

import torch
from thop import clever_format, profile

from core.models import build_model, model_label, normalize_model_size, normalize_version


class ProfileWrapper(torch.nn.Module):
    def __init__(self, model, max_disp):
        super().__init__()
        self.model = model
        self.max_disp = max_disp

    def forward(self, left, right):
        return self.model(left, right, max_disp=self.max_disp, test_mode=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute LiteAnyStereo LAS1 or LAS2 MACs.")
    parser.add_argument("--version", default="las1", help="model version: las1/v1 or las2/v2")
    parser.add_argument("--model_size", "--model-size", default=None, help="LAS2 model size: s, m, l, or h")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--max_disp", type=int, default=192, help="maximum disparity")
    args = parser.parse_args()

    version = normalize_version(args.version)
    model_size = normalize_model_size(version, args.model_size)
    label = model_label(version, model_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = torch.randn(1, 3, args.height, args.width, device=device)
    model = build_model(version, fnet_pretrained=False, model_size=model_size, max_disp=args.max_disp).to(device).eval()
    wrapper = ProfileWrapper(model, max_disp=args.max_disp).to(device).eval()

    with torch.no_grad():
        macs, params = profile(wrapper, inputs=(img, img))
    macs, params = clever_format([macs, params], "%.3f")

    print("Model:", label)
    print("Input size:", img.size())
    print("Macs:", macs)
    print("params:", params)
