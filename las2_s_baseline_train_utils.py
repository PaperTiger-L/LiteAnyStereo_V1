from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from core.liteanystereov2 import LiteAnyStereoS
from las2_training_logging import BatchProgressLogger
from las2_s_hfe_train_utils import (
    _log_stereo_images,
    _resolve_project_path,
    compute_stereo_metric_sums,
    load_checkpoint_weights,
)


def load_baseline_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f'Config file not found: {config_path}'
        )

    with config_path.open('r') as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            'The top-level YAML object must be a dictionary'
        )

    required_sections = (
        'DATA_ROOT',
        'DATA_CONFIG',
        'AUGMENTATION',
        'LOGGING',
        'MODEL',
        'TRAIN',
    )
    missing_sections = [
        section
        for section in required_sections
        if section not in config
    ]
    if missing_sections:
        raise KeyError(
            f'Missing required config sections: {missing_sections}'
        )

    if not isinstance(config['DATA_ROOT'], str) or not config['DATA_ROOT']:
        raise ValueError('DATA_ROOT must be a non-empty string')

    data_infos = config['DATA_CONFIG'].get('DATA_INFOS')
    if not isinstance(data_infos, list) or not data_infos:
        raise ValueError(
            'DATA_CONFIG.DATA_INFOS must contain at least one dataset'
        )

    data_info = data_infos[0]
    if data_info.get('DATASET') != 'CustomDataset':
        raise ValueError(
            f"Unsupported dataset: {data_info.get('DATASET')}"
        )

    data_split = data_info.get('DATA_SPLIT')
    if not isinstance(data_split, dict):
        raise TypeError(
            'DATA_INFOS[0].DATA_SPLIT must be a dictionary'
        )
    for split_prefix in ('TRAINING', 'EVALUATING'):
        split_keys = [
            split_key
            for split_key in data_split
            if split_key.startswith(split_prefix)
        ]
        if not split_keys:
            raise KeyError(
                f'DATA_SPLIT must contain at least one {split_prefix} key'
            )
    for split_key, split_path in data_split.items():
        if not isinstance(split_path, str) or not split_path:
            raise ValueError(
                f'DATA_SPLIT.{split_key} must be a non-empty string'
            )

    model_config = config['MODEL']
    if model_config.get('VERSION') != 'las2':
        raise ValueError(
            f"Expected MODEL.VERSION=las2, got "
            f"{model_config.get('VERSION')}"
        )
    if model_config.get('MODEL_SIZE') != 's':
        raise ValueError(
            f"Expected MODEL.MODEL_SIZE=s, got "
            f"{model_config.get('MODEL_SIZE')}"
        )

    max_disp = model_config.get('MAX_DISP')
    if not isinstance(max_disp, int) or max_disp <= 0 or max_disp % 4 != 0:
        raise ValueError(
            f'MODEL.MAX_DISP must be a positive multiple of 4, got {max_disp}'
        )

    train_config = config['TRAIN']
    if not isinstance(train_config, dict):
        raise TypeError('TRAIN must be a dictionary')

    seed = train_config.get('SEED')
    if seed is not None and (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
    ):
        raise ValueError(
            f'TRAIN.SEED must be a non-negative integer, got {seed}'
        )

    for key in ('OUTPUT_DIR', 'LOG_DIR'):
        if not isinstance(train_config.get(key), str) or not train_config[key]:
            raise ValueError(
                f'TRAIN.{key} must be a non-empty string'
            )

    for key in ('EPOCHS', 'BATCH_SIZE', 'NUM_WORKERS'):
        value = train_config.get(key)
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f'TRAIN.{key} must be a non-negative integer, got {value}'
            )
    if train_config['EPOCHS'] == 0 or train_config['BATCH_SIZE'] == 0:
        raise ValueError('EPOCHS and BATCH_SIZE must be positive')

    for key in ('LR', 'WEIGHT_DECAY'):
        value = train_config.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError(
                f'TRAIN.{key} must be numeric, got {value}'
            )
        if key == 'LR' and value <= 0:
            raise ValueError('TRAIN.LR must be positive')
        if key == 'WEIGHT_DECAY' and value < 0:
            raise ValueError(
                'TRAIN.WEIGHT_DECAY must be non-negative'
            )

    if not isinstance(train_config.get('AMP'), bool):
        raise TypeError('TRAIN.AMP must be boolean')

    scheduler_config = train_config.get('SCHEDULER', {})
    if not isinstance(scheduler_config, dict):
        raise TypeError('TRAIN.SCHEDULER must be a dictionary')
    if scheduler_config.get('NAME', 'cosine') not in (
        'cosine',
        'warmup_cosine',
        'none',
    ):
        raise ValueError(
            f"Unsupported scheduler: {scheduler_config.get('NAME')}"
        )

    resume_value = train_config.get('RESUME')
    if resume_value is not None and not isinstance(resume_value, str):
        raise TypeError('TRAIN.RESUME must be a string or null')

    return config


def build_baseline_model(config, device, logger):
    model = LiteAnyStereoS(fnet_pretrained=False)
    checkpoint_value = config['MODEL'].get('PRETRAINED')

    if (
        checkpoint_value is not None
        and str(checkpoint_value).lower() != 'none'
    ):
        checkpoint_path = _resolve_project_path(checkpoint_value)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f'Checkpoint not found: {checkpoint_path}'
            )

        checkpoint = load_checkpoint_weights(
            checkpoint_path,
            map_location='cpu',
        )

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif (
            isinstance(checkpoint, dict)
            and 'state_dict' in checkpoint
        ):
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise TypeError(
                'Checkpoint must contain a state dictionary'
            )

        if state_dict and all(
            key.startswith('module.')
            for key in state_dict
        ):
            state_dict = {
                key[len('module.'):]: value
                for key, value in state_dict.items()
            }

        model.load_state_dict(
            state_dict,
            strict=True,
        )
        if logger is not None:
            logger.info(
                'Loaded full original LAS2-S checkpoint: %s',
                checkpoint_path,
            )

    model = model.to(device)
    return model


def baseline_disparity_loss(
        pred_disp,
        gt_disp,
        valid,
        max_disp,
        beta=1.0,
):
    if pred_disp.ndim != 4 or pred_disp.shape[1] != 1:
        raise ValueError(
            f'pred_disp must have shape [B, 1, H, W], got {pred_disp.shape}'
        )
    if gt_disp.shape != pred_disp.shape:
        raise ValueError(
            f'Prediction and GT shapes must match, got '
            f'{pred_disp.shape} and {gt_disp.shape}'
        )
    if valid.shape != gt_disp.shape:
        raise ValueError(
            f'Valid mask and GT shapes must match, got '
            f'{valid.shape} and {gt_disp.shape}'
        )

    valid_mask = (
        torch.isfinite(pred_disp)
        & torch.isfinite(gt_disp)
        & (gt_disp >= 0)
        & (gt_disp < max_disp)
        & (valid > 0)
    )

    if not valid_mask.any():
        return pred_disp.sum() * 0.0

    return F.smooth_l1_loss(
        pred_disp[valid_mask],
        gt_disp[valid_mask],
        reduction='mean',
        beta=beta,
    )

def validate_epoch(
        model,
        valid_loader,
        device,
        max_disp,
        use_amp,
        writer=None,
        global_step=0,
        image_interval=0,
        logger=None,
        epoch=1,
        total_epochs=1,
        text_interval=0,
        progress_label='Validation',
):
    model.eval()
    num_batches = len(valid_loader)
    if num_batches == 0:
        raise ValueError('valid_loader must contain at least one batch')

    running_loss = 0.0
    metric_sums = {
        'epe_sum': 0.0,
        'bad3_count': 0.0,
        'd1_count': 0.0,
        'valid_count': 0.0,
    }
    progress = BatchProgressLogger(
        logger=logger,
        phase=progress_label,
        epoch=epoch,
        total_epochs=total_epochs,
        num_batches=num_batches,
        interval=text_interval,
        device=device,
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(valid_loader, start=1):
            batch_timing = progress.start_batch(batch_index)
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

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                predictions = model(
                    left_batch,
                    right_batch,
                    max_disp=max_disp,
                    test_mode=False,
                )
                pred_disp = predictions[0]
                loss = baseline_disparity_loss(
                    pred_disp=pred_disp,
                    gt_disp=gt_disp_batch,
                    valid=valid_batch,
                    max_disp=max_disp,
                )

            loss_value = loss.item()
            running_loss += loss_value
            metric_batch = compute_stereo_metric_sums(
                pred_disp=pred_disp,
                gt_disp=gt_disp_batch,
                valid=valid_batch,
                max_disp=max_disp,
            )
            for key in metric_sums:
                metric_sums[key] += metric_batch[key]

            step = global_step + batch_index
            if (
                writer is not None
                and image_interval > 0
                and step % image_interval == 0
            ):
                _log_stereo_images(
                    writer=writer,
                    prefix='valid',
                    left=left_batch,
                    right=right_batch,
                    gt_disp=gt_disp_batch,
                    pred_disp=pred_disp,
                    max_disp=max_disp,
                    step=step,
                )

            valid_count = metric_sums['valid_count']
            running_epe = (
                metric_sums['epe_sum'] / valid_count
                if valid_count > 0
                else 0.0
            )
            running_d1 = (
                metric_sums['d1_count'] / valid_count
                if valid_count > 0
                else 0.0
            )
            progress.finish_batch(
                batch_index=batch_index,
                timing=batch_timing,
                loss=loss_value,
                average_loss=running_loss / batch_index,
                metrics={
                    'EPE': running_epe,
                    'D1': running_d1,
                },
            )

    if metric_sums['valid_count'] == 0:
        raise ValueError(
            'Validation set contains no valid disparity pixels'
        )

    return {
        'loss': running_loss / num_batches,
        'epe': (
            metric_sums['epe_sum']
            / metric_sums['valid_count']
        ),
        'bad3': (
            metric_sums['bad3_count']
            / metric_sums['valid_count']
        ),
        'd1': (
            metric_sums['d1_count']
            / metric_sums['valid_count']
        ),
        'valid_pixels': metric_sums['valid_count'],
    }
