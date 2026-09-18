import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.liteanystereov2 import LiteAnyStereoS
from core.liteanystereov2_hfe import LiteAnyStereoSHFE
from core.submodule import disparity_regression
from las2_s_baseline_train_utils import load_baseline_config
from las2_s_hfe_train_utils import (
    CustomDataset,
    collate_stereo_batch,
    load_checkpoint_weights,
)


PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATION_SPLITS = (
    "TRAININGADD1",
    "EVALUATINGADD1",
    "TESTING0",
)
A0_ARCHITECTURES = {
    "las2_s_baseline_las2_s_v1",
}
A1_ARCHITECTURES = {
    "las2_s_hfe_litematch_cvs_v1",
    "las2_s_hfe_fair_final_disp_v1",
    "las2_s_hfe_ablation_v1",
    "las2_s_hfe_loss_ablation_v1",
    "las2_s_hfe_distill_v1",
    "las2_s_hfe_distill_v2",
}


def resolve_project_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    return torch.device(device_name)


def build_logger(log_path):
    logger = logging.getLogger("las2_s_evaluation")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def load_model_yaml(config_path):
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The top-level YAML object must be a dictionary")
    return config


def validate_evaluation_config(config, model_name):
    required_sections = (
        "DATA_ROOT",
        "DATA_CONFIG",
        "AUGMENTATION",
        "MODEL",
        "TRAIN",
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

    data_infos = config["DATA_CONFIG"].get("DATA_INFOS")
    if not isinstance(data_infos, list) or not data_infos:
        raise ValueError("DATA_CONFIG.DATA_INFOS must not be empty")
    data_info = data_infos[0]
    if not isinstance(data_info, dict):
        raise TypeError("DATA_INFOS entries must be dictionaries")
    if data_info.get("DATASET") != "CustomDataset":
        raise ValueError("Evaluation requires DATASET=CustomDataset")
    if not isinstance(data_info.get("DATA_SPLIT"), dict):
        raise TypeError("DATA_INFOS[0].DATA_SPLIT must be a dictionary")

    eval_crop_size = config["AUGMENTATION"].get("EVAL_CROP_SIZE")
    if (
        not isinstance(eval_crop_size, (list, tuple))
        or len(eval_crop_size) != 2
        or any(size <= 0 for size in eval_crop_size)
    ):
        raise ValueError(
            "AUGMENTATION.EVAL_CROP_SIZE must contain two positive values"
        )

    model_config = config["MODEL"]
    if model_config.get("VERSION") != "las2":
        raise ValueError("Evaluation requires MODEL.VERSION=las2")
    if model_config.get("MODEL_SIZE") != "s":
        raise ValueError("Evaluation requires MODEL.MODEL_SIZE=s")
    max_disp = model_config.get("MAX_DISP")
    if not isinstance(max_disp, int) or max_disp <= 0 or max_disp % 4 != 0:
        raise ValueError(
            f"MODEL.MAX_DISP must be a positive multiple of 4, got {max_disp}"
        )

    if model_name == "a1":
        hfe_config = model_config.get("HFE", {})
        if not isinstance(hfe_config, dict):
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
    batch_size = train_config.get("BATCH_SIZE")
    num_workers = train_config.get("NUM_WORKERS")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("TRAIN.BATCH_SIZE must be a positive integer")
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("TRAIN.NUM_WORKERS must be a non-negative integer")
    if not isinstance(train_config.get("AMP"), bool):
        raise TypeError("TRAIN.AMP must be boolean")


def load_evaluation_config(config_path, model_name):
    if model_name == "a0":
        return load_baseline_config(config_path)
    if model_name == "a1":
        config = load_model_yaml(config_path)
        validate_evaluation_config(config, model_name)
        return config
    raise ValueError(f"Unsupported model name: {model_name}")


def build_model(config, model_name):
    model_config = config["MODEL"]
    max_disp = model_config["MAX_DISP"]

    if model_name == "a0":
        return LiteAnyStereoS(fnet_pretrained=False)

    if model_name == "a1":
        hfe_config = model_config.get("HFE", {})
        return LiteAnyStereoSHFE(
            fnet_pretrained=False,
            cutoff_ratio=hfe_config.get("CUTOFF_RATIO", 0.1),
            max_disp=max_disp,
            cost_stabilization=model_config["COST_STABILIZATION"],
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def extract_checkpoint_state(checkpoint):
    if (
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("model"), dict)
    ):
        return checkpoint["model"]

    if isinstance(checkpoint, dict) and isinstance(
        checkpoint.get("state_dict"),
        dict,
    ):
        return checkpoint["state_dict"]

    if isinstance(checkpoint, dict):
        return checkpoint

    raise TypeError("Checkpoint must contain a state dictionary")


def load_model_checkpoint(model, checkpoint_path, model_name, device):
    checkpoint = load_checkpoint_weights(
        checkpoint_path,
        map_location="cpu",
    )

    metadata = {}
    if isinstance(checkpoint, dict):
        metadata = {
            key: checkpoint.get(key)
            for key in (
                "architecture_version",
                "epoch",
                "max_disp",
                "cost_channels",
                "model_size",
            )
            if key in checkpoint
        }

    architecture_version = metadata.get("architecture_version")
    supported_architectures = (
        A0_ARCHITECTURES
        if model_name == "a0"
        else A1_ARCHITECTURES
    )
    if (
        architecture_version is not None
        and architecture_version not in supported_architectures
    ):
        raise ValueError(
            f"Checkpoint architecture {architecture_version!r} is not "
            f"compatible with model {model_name}"
        )

    state_dict = extract_checkpoint_state(checkpoint)
    if state_dict and all(
        key.startswith("module.")
        for key in state_dict
    ):
        state_dict = {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return metadata


def build_evaluation_loader(config, split_name, batch_size, num_workers):
    if split_name not in EVALUATION_SPLITS:
        raise ValueError(
            f"Unsupported split {split_name!r}; "
            f"choose from {EVALUATION_SPLITS}"
        )

    data_info = config["DATA_CONFIG"]["DATA_INFOS"][0]
    data_split = data_info["DATA_SPLIT"]
    list_file = data_split.get(split_name)
    if not isinstance(list_file, str) or not list_file:
        raise KeyError(
            f"DATA_CONFIG.DATA_INFOS[0].DATA_SPLIT.{split_name} "
            "must be a non-empty string"
        )

    dataset = CustomDataset(
        data_root=config["DATA_ROOT"],
        list_file=list_file,
        crop_size=config["AUGMENTATION"]["EVAL_CROP_SIZE"],
        training=False,
        max_disp=config["MODEL"]["MAX_DISP"],
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_stereo_batch,
    )


def empty_metric_sums():
    return {
        "epe_sum": 0.0,
        "sq_error_sum": 0.0,
        "abs_rel_sum": 0.0,
        "sq_rel_sum": 0.0,
        "bad3_count": 0.0,
        "d1_count": 0.0,
        "valid_count": 0.0,
        "relative_valid_count": 0.0,
    }


def add_metric_sums(target, source):
    for key in target:
        target[key] += source[key]


def compute_metric_sums(pred_disp, gt_disp, valid, max_disp):
    if pred_disp.ndim != 4 or pred_disp.shape[1] != 1:
        raise ValueError(
            f"pred_disp must have shape [B, 1, H, W], got {pred_disp.shape}"
        )
    if gt_disp.ndim != 4 or gt_disp.shape[1] != 1:
        raise ValueError(
            f"gt_disp must have shape [B, 1, H, W], got {gt_disp.shape}"
        )
    if pred_disp.shape[-2:] != gt_disp.shape[-2:]:
        pred_disp = F.interpolate(
            pred_disp,
            size=gt_disp.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if pred_disp.shape != gt_disp.shape:
        raise ValueError(
            f"Prediction and GT shapes must match, got "
            f"{pred_disp.shape} and {gt_disp.shape}"
        )
    if valid.shape != gt_disp.shape:
        raise ValueError(
            f"Valid mask and GT shapes must match, got "
            f"{valid.shape} and {gt_disp.shape}"
        )

    valid_mask = (
        torch.isfinite(pred_disp)
        & torch.isfinite(gt_disp)
        & (gt_disp >= 0)
        & (gt_disp < max_disp)
        & (valid > 0.5)
    )

    if not valid_mask.any():
        return empty_metric_sums()

    pred_valid = pred_disp[valid_mask]
    gt_valid = gt_disp[valid_mask]
    absolute_error = (pred_valid - gt_valid).abs()
    target_disp = gt_valid.abs().clamp_min(1e-6)
    relative_error = absolute_error / target_disp

    positive_target = gt_valid > 0
    relative_absolute_error = absolute_error[positive_target]
    relative_target = gt_valid[positive_target]

    if relative_target.numel() > 0:
        abs_rel_sum = (
            relative_absolute_error / relative_target
        ).sum().item()
        sq_rel_sum = (
            relative_absolute_error.square() / relative_target
        ).sum().item()
    else:
        abs_rel_sum = 0.0
        sq_rel_sum = 0.0

    return {
        "epe_sum": absolute_error.sum().item(),
        "sq_error_sum": absolute_error.square().sum().item(),
        "abs_rel_sum": abs_rel_sum,
        "sq_rel_sum": sq_rel_sum,
        "bad3_count": (absolute_error > 3.0).sum().item(),
        "d1_count": (
            (absolute_error > 3.0)
            & (relative_error > 0.05)
        ).sum().item(),
        "valid_count": float(absolute_error.numel()),
        "relative_valid_count": float(relative_target.numel()),
    }


def finalize_metric_sums(metric_sums):
    valid_count = metric_sums["valid_count"]
    if valid_count <= 0:
        raise ValueError("Evaluation set contains no valid disparity pixels")

    epe = metric_sums["epe_sum"] / valid_count
    rmse = (
        metric_sums["sq_error_sum"] / valid_count
    ) ** 0.5
    bad3 = metric_sums["bad3_count"] / valid_count
    d1 = metric_sums["d1_count"] / valid_count

    relative_valid_count = metric_sums["relative_valid_count"]
    if relative_valid_count > 0:
        abs_rel = (
            metric_sums["abs_rel_sum"] / relative_valid_count
        )
        sq_rel = (
            metric_sums["sq_rel_sum"] / relative_valid_count
        )
    else:
        abs_rel = None
        sq_rel = None

    return {
        "epe": float(epe),
        "abs_rel": (
            float(abs_rel) if abs_rel is not None else None
        ),
        "sq_rel": (
            float(sq_rel) if sq_rel is not None else None
        ),
        "rmse": float(rmse),
        "d1": float(d1),
        "d1_percent": float(d1 * 100.0),
        "bad3": float(bad3),
        "bad3_percent": float(bad3 * 100.0),
        "valid_pixels": int(valid_count),
        "relative_valid_pixels": int(relative_valid_count),
    }


def low_resolution_valid_mask(gt_disp, valid, target_size, max_disp):
    gt_low = F.interpolate(
        gt_disp.float(),
        size=target_size,
        mode="nearest",
    )[:, 0]
    valid_low = F.interpolate(
        valid.float(),
        size=target_size,
        mode="nearest",
    )[:, 0] > 0.5
    return (
        valid_low
        & torch.isfinite(gt_low)
        & (gt_low >= 0)
        & (gt_low < max_disp)
    )


def empty_probability_stats():
    return {
        "entropy_sum": 0.0,
        "confidence_sum": 0.0,
        "valid_count": 0.0,
    }


def add_probability_stats(target, source):
    for key in target:
        target[key] += source[key]


def compute_probability_stats(probability, gt_disp, valid, max_disp):
    probability = probability.float()
    probability_for_log = probability.clamp_min(1e-8)
    entropy = -(
        probability_for_log * probability_for_log.log()
    ).sum(dim=1)
    confidence = probability.max(dim=1).values
    valid_low = low_resolution_valid_mask(
        gt_disp=gt_disp,
        valid=valid,
        target_size=probability.shape[-2:],
        max_disp=max_disp,
    )

    if not valid_low.any():
        return empty_probability_stats()

    return {
        "entropy_sum": entropy[valid_low].sum().item(),
        "confidence_sum": confidence[valid_low].sum().item(),
        "valid_count": float(valid_low.sum().item()),
    }


def finalize_probability_stats(stats):
    valid_count = stats["valid_count"]
    if valid_count <= 0:
        return {
            "mean_entropy": None,
            "mean_max_probability": None,
            "valid_pixels": 0,
        }

    return {
        "mean_entropy": float(stats["entropy_sum"] / valid_count),
        "mean_max_probability": float(
            stats["confidence_sum"] / valid_count
        ),
        "valid_pixels": int(valid_count),
    }


def empty_residual_stats():
    return {
        "abs_sum": 0.0,
        "value_count": 0.0,
    }


def add_residual_stats(target, source):
    for key in target:
        target[key] += source[key]


def compute_residual_stats(left_cost, right_cost):
    residual = (right_cost.float() - left_cost.float()).abs()
    return {
        "abs_sum": residual.sum().item(),
        "value_count": float(residual.numel()),
    }


def finalize_residual_stats(stats):
    value_count = stats["value_count"]
    if value_count <= 0:
        return {"mean_absolute_residual": None}
    return {
        "mean_absolute_residual": float(
            stats["abs_sum"] / value_count
        ),
    }


def collect_hfe_stage_outputs(outputs):
    cost_prob_c1 = F.softmax(
        outputs["cost_logits_c1"].float(),
        dim=1,
    )
    disp_low_c1 = disparity_regression(
        cost_prob_c1,
        cost_prob_c1.shape[1],
    ) * 4.0

    return {
        "c0": outputs["disp_low_c0"],
        "c1": disp_low_c1,
        "c2": outputs["disp_low_c2"],
    }, {
        "c0": outputs["cost_prob_c0"],
        "c1": cost_prob_c1,
        "c2": outputs["cost_prob_c2"],
    }


def evaluate_model(
        model,
        model_name,
        data_loader,
        device,
        max_disp,
        use_amp,
        include_stage_diagnostics,
        logger,
):
    model.eval()
    final_metric_sums = empty_metric_sums()
    stage_metric_sums = {
        key: empty_metric_sums()
        for key in ("c0", "c1", "c2")
    }
    probability_stats = {
        key: empty_probability_stats()
        for key in ("c0", "c1", "c2")
    }
    residual_stats = {
        "c1_minus_c0": empty_residual_stats(),
        "c2_minus_c1": empty_residual_stats(),
    }
    num_samples = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader, start=1):
            (
                _,
                left_batch,
                right_batch,
                gt_disp_batch,
                valid_batch,
            ) = batch

            left_batch = left_batch.to(device, non_blocking=True)
            right_batch = right_batch.to(device, non_blocking=True)
            gt_disp_batch = gt_disp_batch.to(device, non_blocking=True)
            valid_batch = valid_batch.to(device, non_blocking=True)
            num_samples += left_batch.shape[0]

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                if model_name == "a1":
                    outputs = model(
                        left_batch,
                        right_batch,
                        max_disp=max_disp,
                        test_mode=False,
                        return_aux=include_stage_diagnostics,
                    )
                    if include_stage_diagnostics:
                        final_prediction = outputs["disp_up"]
                    else:
                        final_prediction = outputs[0]
                else:
                    outputs = None
                    predictions = model(
                        left_batch,
                        right_batch,
                        max_disp=max_disp,
                        test_mode=False,
                    )
                    final_prediction = predictions[0]

            add_metric_sums(
                final_metric_sums,
                compute_metric_sums(
                    pred_disp=final_prediction,
                    gt_disp=gt_disp_batch,
                    valid=valid_batch,
                    max_disp=max_disp,
                ),
            )

            if include_stage_diagnostics:
                if model_name != "a1":
                    raise ValueError(
                        "Stage diagnostics require model=a1"
                    )

                stage_predictions, stage_probabilities = (
                    collect_hfe_stage_outputs(outputs)
                )
                for stage_name in ("c0", "c1", "c2"):
                    add_metric_sums(
                        stage_metric_sums[stage_name],
                        compute_metric_sums(
                            pred_disp=stage_predictions[stage_name],
                            gt_disp=gt_disp_batch,
                            valid=valid_batch,
                            max_disp=max_disp,
                        ),
                    )
                    add_probability_stats(
                        probability_stats[stage_name],
                        compute_probability_stats(
                            probability=stage_probabilities[stage_name],
                            gt_disp=gt_disp_batch,
                            valid=valid_batch,
                            max_disp=max_disp,
                        ),
                    )

                add_residual_stats(
                    residual_stats["c1_minus_c0"],
                    compute_residual_stats(
                        outputs["cost_logits_c0"],
                        outputs["cost_logits_c1"],
                    ),
                )
                add_residual_stats(
                    residual_stats["c2_minus_c1"],
                    compute_residual_stats(
                        outputs["cost_logits_c1"],
                        outputs["cost_logits_c2"],
                    ),
                )

            if (
                batch_index == 1
                or batch_index == len(data_loader)
                or batch_index % 50 == 0
            ):
                logger.info(
                    "Evaluated batch %d/%d",
                    batch_index,
                    len(data_loader),
                )

    result = {
        "num_samples": num_samples,
        "metrics": finalize_metric_sums(final_metric_sums),
    }

    if include_stage_diagnostics:
        result["stage_metrics"] = {
            stage_name: finalize_metric_sums(stage_metric_sums[stage_name])
            for stage_name in ("c0", "c1", "c2")
        }
        result["cost_volume_statistics"] = {
            stage_name: finalize_probability_stats(
                probability_stats[stage_name]
            )
            for stage_name in ("c0", "c1", "c2")
        }
        result["residual_statistics"] = {
            residual_name: finalize_residual_stats(
                residual_stats[residual_name]
            )
            for residual_name in (
                "c1_minus_c0",
                "c2_minus_c1",
            )
        }

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate original LAS2-S or HFE/CVS LAS2-S",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the model YAML configuration",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("a0", "a1"),
        help="a0=original LAS2-S, a1=HFE + Fusion + CVS",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a model checkpoint",
    )
    parser.add_argument(
        "--split",
        default="TESTING0",
        choices=EVALUATION_SPLITS,
        help="Dataset split key from DATA_SPLIT",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for eval.log and metrics.json",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the config AMP setting",
    )
    parser.add_argument(
        "--stage-diagnostics",
        action="store_true",
        help="Also evaluate HFE C0/C1/C2 and cost statistics",
    )
    return parser.parse_args()


def run_evaluation(args):
    config_path = resolve_project_path(args.config)
    checkpoint_path = resolve_project_path(args.checkpoint)
    config = load_evaluation_config(config_path, args.model)
    max_disp = config["MODEL"]["MAX_DISP"]

    if args.stage_diagnostics and args.model != "a1":
        raise ValueError("--stage-diagnostics is only valid with --model a1")

    device = resolve_device(args.device)
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else config["TRAIN"]["BATCH_SIZE"]
    )
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else config["TRAIN"]["NUM_WORKERS"]
    )
    requested_amp = (
        config["TRAIN"].get("AMP", False)
        if args.amp is None
        else args.amp
    )
    use_amp = bool(requested_amp and device.type == "cuda")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(
            f"num_workers must be non-negative, got {num_workers}"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if args.output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "tmp"
            / "experiments"
            / "evaluation"
            / f"{args.model}_{args.split.lower()}_{checkpoint_path.stem}"
        )
    else:
        output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(output_dir / "eval.log")
    logger.info("Model: %s", args.model)
    logger.info("Configuration: %s", config_path)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Split: %s", args.split)
    logger.info("Device: %s", device)
    logger.info("AMP requested: %s; AMP enabled: %s", requested_amp, use_amp)
    logger.info("Batch size: %d; workers: %d", batch_size, num_workers)

    model = build_model(config, args.model)
    checkpoint_metadata = load_model_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        model_name=args.model,
        device=device,
    )
    logger.info(
        "Loaded checkpoint architecture: %s",
        checkpoint_metadata.get("architecture_version"),
    )

    data_loader = build_evaluation_loader(
        config=config,
        split_name=args.split,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    logger.info(
        "Dataset samples: %d; batches: %d",
        len(data_loader.dataset),
        len(data_loader),
    )

    evaluation_result = evaluate_model(
        model=model,
        model_name=args.model,
        data_loader=data_loader,
        device=device,
        max_disp=max_disp,
        use_amp=use_amp,
        include_stage_diagnostics=args.stage_diagnostics,
        logger=logger,
    )

    result = {
        "model": args.model,
        "checkpoint_metadata": checkpoint_metadata,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "max_disp": max_disp,
        "device": str(device),
        "amp_requested": bool(requested_amp),
        "amp_enabled": use_amp,
        "batch_size": batch_size,
        "num_workers": num_workers,
        **evaluation_result,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Metrics written to %s", metrics_path)
    logger.info(
        "Final metrics: EPE=%.6f AbsRel=%s SqRel=%s RMSE=%.6f "
        "D1=%.4f%% Bad-3=%.4f%% valid=%d relative_valid=%d",
        result["metrics"]["epe"],
        (
            f'{result["metrics"]["abs_rel"]:.6f}'
            if result["metrics"]["abs_rel"] is not None
            else "n/a"
        ),
        (
            f'{result["metrics"]["sq_rel"]:.6f}'
            if result["metrics"]["sq_rel"] is not None
            else "n/a"
        ),
        result["metrics"]["rmse"],
        result["metrics"]["d1_percent"],
        result["metrics"]["bad3_percent"],
        result["metrics"]["valid_pixels"],
        result["metrics"]["relative_valid_pixels"],
    )
    return result


def main():
    args = parse_args()
    result = run_evaluation(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
