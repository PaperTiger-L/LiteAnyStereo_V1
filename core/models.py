import os

from .liteanystereo import LiteAnyStereo as LiteAnyStereoV1
from .liteanystereov2 import (
    DEFAULT_LAS2_MODEL_SIZE,
    LAS2_MODEL_SIZES,
    build_liteanystereo as build_las2_model,
    normalize_las2_model_size,
)


VERSION_ALIASES = {
    "1": "las1",
    "v1": "las1",
    "las1": "las1",
    "liteanystereo": "las1",
    "2": "las2",
    "v2": "las2",
    "las2": "las2",
    "liteanystereov2": "las2",
}

LAS2_DEFAULT_CHECKPOINTS = {
    "s": "./checkpoints/LAS2_S.pth",
    "m": "./checkpoints/LAS2_M.pth",
    "l": "./checkpoints/LAS2_L.pth",
    "h": "./checkpoints/LAS2_H.pth",
}

DEFAULT_CHECKPOINTS = {
    "las1": "./checkpoints/LiteAnyStereo.pth",
    "las2": LAS2_DEFAULT_CHECKPOINTS,
}


def normalize_version(version):
    key = str(version).lower()
    if key not in VERSION_ALIASES:
        choices = ", ".join(sorted(VERSION_ALIASES))
        raise ValueError(f"Unknown LiteAnyStereo version '{version}'. Available aliases: {choices}")
    return VERSION_ALIASES[key]


def normalize_model_size(version, model_size=None):
    version = normalize_version(version)
    if version == "las1":
        return None
    return normalize_las2_model_size(model_size)


def model_label(version, model_size=None):
    version = normalize_version(version)
    if version == "las1":
        return "LAS1"
    return f"LAS2-{normalize_las2_model_size(model_size).upper()}"


def build_model(version, fnet_pretrained=False, model_size=None, max_disp=192):
    version = normalize_version(version)
    if version == "las1":
        return LiteAnyStereoV1(fnet_pretrained=fnet_pretrained)
    return build_las2_model(model_size=model_size, fnet_pretrained=fnet_pretrained, max_disp=max_disp)


def default_checkpoint(version, model_size=None):
    version = normalize_version(version)
    if version == "las1":
        return DEFAULT_CHECKPOINTS[version]
    return LAS2_DEFAULT_CHECKPOINTS[normalize_las2_model_size(model_size)]


def resolve_checkpoint(version, restore_ckpt, model_size=None):
    if restore_ckpt is None:
        return default_checkpoint(version, model_size=model_size)
    if str(restore_ckpt).lower() == "none":
        return None
    return restore_ckpt


def load_model_weights(model, checkpoint, strict=True):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state dict or contain a 'model'/'state_dict' entry.")

    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[len("module."):]: value for key, value in checkpoint.items()}

    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(checkpoint, strict=strict)


def require_checkpoint(path):
    if path is not None and not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
