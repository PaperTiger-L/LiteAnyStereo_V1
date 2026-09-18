from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
LOSS_KEYS = (
    "LAMBDA_CVC_C0",
    "LAMBDA_CVC_C2",
    "LAMBDA_D0",
    "LAMBDA_D2",
    "LAMBDA_DISP",
)
LOSS_VARIANTS = {
    "l0_disp_up_only",
    "l1_disp_up_plus_aux",
}


def load_loss_ablation_config(config_path):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The top-level YAML object must be a dictionary")

    required_sections = (
        "DATA_ROOT",
        "DATA_CONFIG",
        "AUGMENTATION",
        "LOGGING",
        "MODEL",
        "TRAIN",
        "LOSS_ABLATION",
    )
    missing_sections = [
        section
        for section in required_sections
        if section not in config
    ]
    if missing_sections:
        raise KeyError(f"Missing required config sections: {missing_sections}")

    if not isinstance(config["DATA_ROOT"], str) or not config["DATA_ROOT"]:
        raise ValueError("DATA_ROOT must be a non-empty string")

    data_config = config["DATA_CONFIG"]
    if not isinstance(data_config, dict):
        raise TypeError("DATA_CONFIG must be a dictionary")
    data_infos = data_config.get("DATA_INFOS")
    if not isinstance(data_infos, list) or not data_infos:
        raise ValueError("DATA_CONFIG.DATA_INFOS must not be empty")
    data_info = data_infos[0]
    if not isinstance(data_info, dict):
        raise TypeError("DATA_INFOS entries must be dictionaries")
    if data_info.get("DATASET") != "CustomDataset":
        raise ValueError("Loss ablation requires DATASET=CustomDataset")
    data_split = data_info.get("DATA_SPLIT")
    if not isinstance(data_split, dict):
        raise TypeError("DATA_INFOS[0].DATA_SPLIT must be a dictionary")
    for split_name in ("TRAININGADD1", "EVALUATINGADD1"):
        if not isinstance(data_split.get(split_name), str) or not data_split[split_name]:
            raise KeyError(
                f"DATA_SPLIT.{split_name} must be a non-empty string"
            )

    augmentation_config = config["AUGMENTATION"]
    if not isinstance(augmentation_config, dict):
        raise TypeError("AUGMENTATION must be a dictionary")
    for crop_key in ("TRAIN_CROP_SIZE", "EVAL_CROP_SIZE"):
        crop_size = augmentation_config.get(crop_key)
        if (
            not isinstance(crop_size, (list, tuple))
            or len(crop_size) != 2
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size <= 0
                for size in crop_size
            )
        ):
            raise ValueError(
                f"AUGMENTATION.{crop_key} must contain two positive integers"
            )

    logging_config = config["LOGGING"]
    if not isinstance(logging_config, dict):
        raise TypeError("LOGGING must be a dictionary")
    for interval_key in ("SCALAR_INTERVAL", "IMAGE_INTERVAL"):
        interval = logging_config.get(interval_key)
        if (
            not isinstance(interval, int)
            or isinstance(interval, bool)
            or interval < 0
        ):
            raise ValueError(
                f"LOGGING.{interval_key} must be a non-negative integer"
            )

    model_config = config["MODEL"]
    if not isinstance(model_config, dict):
        raise TypeError("MODEL must be a dictionary")
    if model_config.get("VERSION") != "las2":
        raise ValueError("Loss ablation requires MODEL.VERSION=las2")
    if model_config.get("MODEL_SIZE") != "s":
        raise ValueError("Loss ablation requires MODEL_SIZE=s")
    max_disp = model_config.get("MAX_DISP")
    if (
        not isinstance(max_disp, int)
        or isinstance(max_disp, bool)
        or max_disp <= 0
        or max_disp % 4 != 0
    ):
        raise ValueError(
            "MODEL.MAX_DISP must be a positive multiple of 4"
        )
    pretrained = model_config.get("PRETRAINED")
    if pretrained is not None and (
        not isinstance(pretrained, str)
        or not pretrained
    ):
        raise ValueError(
            "MODEL.PRETRAINED must be a non-empty path or none"
        )
    if not isinstance(model_config.get("HFE", {}), dict):
        raise TypeError("MODEL.HFE must be a dictionary")
    cost_config = model_config.get("COST_STABILIZATION")
    if not isinstance(cost_config, dict):
        raise TypeError("MODEL.COST_STABILIZATION must be a dictionary")
    if cost_config.get("ENABLED") is not True:
        raise ValueError(
            "MODEL.COST_STABILIZATION.ENABLED must be true"
        )

    train_config = config["TRAIN"]
    if not isinstance(train_config, dict):
        raise TypeError("TRAIN must be a dictionary")
    for path_key in ("OUTPUT_DIR", "LOG_DIR"):
        path_value = train_config.get(path_key)
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(
                f"TRAIN.{path_key} must be a non-empty string"
            )
    for integer_key in ("EPOCHS", "BATCH_SIZE", "NUM_WORKERS"):
        value = train_config.get(integer_key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                f"TRAIN.{integer_key} must be a non-negative integer"
            )
    if train_config["EPOCHS"] <= 0 or train_config["BATCH_SIZE"] <= 0:
        raise ValueError(
            "TRAIN.EPOCHS and TRAIN.BATCH_SIZE must be positive"
        )

    seed = train_config.get("SEED")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
    ):
        raise ValueError("TRAIN.SEED must be a non-negative integer")

    learning_rate = train_config.get("LR")
    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("TRAIN.LR must be a finite positive number")

    weight_decay = train_config.get("WEIGHT_DECAY")
    if (
        not isinstance(weight_decay, (int, float))
        or isinstance(weight_decay, bool)
        or not np.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError(
            "TRAIN.WEIGHT_DECAY must be a finite non-negative number"
        )

    if not isinstance(train_config.get("AMP"), bool):
        raise TypeError("TRAIN.AMP must be boolean")

    scheduler_config = train_config.get("SCHEDULER", {})
    if not isinstance(scheduler_config, dict):
        raise TypeError("TRAIN.SCHEDULER must be a dictionary")
    scheduler_name = scheduler_config.get("NAME", "cosine")
    if scheduler_name not in ("cosine", "none"):
        raise ValueError(
            "TRAIN.SCHEDULER.NAME must be cosine or none"
        )
    min_lr = scheduler_config.get("MIN_LR", 0.0)
    if (
        not isinstance(min_lr, (int, float))
        or isinstance(min_lr, bool)
        or not np.isfinite(min_lr)
        or min_lr < 0
        or min_lr > learning_rate
    ):
        raise ValueError(
            "TRAIN.SCHEDULER.MIN_LR must be finite, non-negative, "
            "and no greater than TRAIN.LR"
        )

    resume_value = train_config.get("RESUME")
    if resume_value is not None and str(resume_value).lower() != "none":
        raise ValueError(
            "Loss ablation currently requires TRAIN.RESUME=none"
        )

    loss_config = train_config.get("LOSS")
    if not isinstance(loss_config, dict):
        raise TypeError("TRAIN.LOSS must be a dictionary")
    for loss_key in LOSS_KEYS:
        loss_weight = loss_config.get(loss_key)
        if (
            not isinstance(loss_weight, (int, float))
            or isinstance(loss_weight, bool)
            or not np.isfinite(loss_weight)
            or loss_weight < 0
        ):
            raise ValueError(
                f"TRAIN.LOSS.{loss_key} must be finite and non-negative"
            )
    if loss_config["LAMBDA_DISP"] <= 0:
        raise ValueError(
            "TRAIN.LOSS.LAMBDA_DISP must be positive"
        )
    if sum(loss_config[key] for key in LOSS_KEYS) <= 0:
        raise ValueError(
            "At least one TRAIN.LOSS weight must be positive"
        )

    loss_ablation_config = config["LOSS_ABLATION"]
    if not isinstance(loss_ablation_config, dict):
        raise TypeError("LOSS_ABLATION must be a dictionary")
    variant_name = loss_ablation_config.get("NAME")
    if variant_name not in LOSS_VARIANTS:
        raise ValueError(
            f"LOSS_ABLATION.NAME must be one of {sorted(LOSS_VARIANTS)}"
        )

    if variant_name == "l0_disp_up_only":
        for loss_key in LOSS_KEYS[:-1]:
            if loss_config[loss_key] != 0.0:
                raise ValueError(
                    "l0_disp_up_only requires all auxiliary Loss weights to be 0"
                )
    elif variant_name == "l1_disp_up_plus_aux":
        for loss_key in LOSS_KEYS[:-1]:
            if loss_config[loss_key] <= 0:
                raise ValueError(
                    "l1_disp_up_plus_aux requires every auxiliary Loss weight to be positive"
                )

    return config, config_path
