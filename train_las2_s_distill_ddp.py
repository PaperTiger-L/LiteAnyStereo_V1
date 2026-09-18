"""DDP teacher-student distillation for the LAS2-S HFE/CVS student.

The teacher is the strongest original LAS2-S Baseline checkpoint, usually
``baseline_pretrained_seed2026_ddp/best.pth``.  The student is the modified
LAS2-S HFE/CVS model trained with the existing CVC/D0/D2/disp_up supervision
plus an additional teacher disparity distillation loss.

This entry intentionally does not launch training by itself.  Use torchrun, for
example:

    torchrun --standalone --nproc_per_node=8 train_las2_s_distill_ddp.py \
        --config tmp/experiments/outside_grass_20260417_seed2026/configs/modified_l1_clip_distill_teacher_baseline_pretrained_seed2026.yaml \
        --output-subdir modified_l1_clip_distill_teacher_baseline_pretrained_seed2026_ddp
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from las2_training_logging import (
    BatchProgressLogger,
    log_config,
    log_optimization_context,
    resolve_text_interval,
)
from las2_s_baseline_train_utils import build_baseline_model
from train_las2_s_ddp import (
    build_ddp_dataloaders,
    build_ddp_logger,
    set_random_seed,
    _reduce_stat_sums,
)
from las2_s_hfe_train_utils import (
    _log_stereo_images,
    _resolve_project_path,
    build_hfe_model,
    build_optimizer_and_scaler,
    build_scheduler,
    compute_las2_s_hfe_loss,
    load_checkpoint_weights,
    validate_epoch as hfe_validate_epoch,
)
from las2_s_loss_config import load_loss_ablation_config


DISTILL_STAT_KEYS = (
    'loss',
    'loss_supervised',
    'loss_kd_disp',
    'weighted_loss_kd_disp',
    'kd_valid_ratio',
    'teacher_gt_epe',
    'perturb_applied_ratio',
    'loss_cvc_c0',
    'loss_cvc_c2',
    'loss_d0',
    'loss_d2',
    'loss_disp',
    'weighted_loss_cvc_c0',
    'weighted_loss_cvc_c2',
    'weighted_loss_d0',
    'weighted_loss_d2',
    'weighted_loss_disp',
)


HFE_VALID_LOSS_KEYS = (
    'LAMBDA_CVC_C0',
    'LAMBDA_CVC_C2',
    'LAMBDA_D0',
    'LAMBDA_D2',
    'LAMBDA_DISP',
)


def _as_finite_number(value, name, *, minimum=None, allow_none=False):
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(value)
    ):
        raise ValueError(f'{name} must be a finite number, got {value}')
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {value}')
    return value


def validate_distillation_config(config):
    distill_config = config.get('DISTILLATION')
    if not isinstance(distill_config, dict):
        raise TypeError('DISTILLATION must be a dictionary')

    teacher_config = distill_config.get('TEACHER')
    if not isinstance(teacher_config, dict):
        raise TypeError('DISTILLATION.TEACHER must be a dictionary')

    teacher_checkpoint = teacher_config.get('CHECKPOINT')
    if not isinstance(teacher_checkpoint, str) or not teacher_checkpoint:
        raise ValueError('DISTILLATION.TEACHER.CHECKPOINT must be non-empty')

    loss_config = distill_config.get('LOSS')
    if not isinstance(loss_config, dict):
        raise TypeError('DISTILLATION.LOSS must be a dictionary')

    lambda_kd = _as_finite_number(
        loss_config.get('LAMBDA_KD_DISP'),
        'DISTILLATION.LOSS.LAMBDA_KD_DISP',
        minimum=0.0,
    )
    if lambda_kd <= 0:
        raise ValueError('DISTILLATION.LOSS.LAMBDA_KD_DISP must be > 0')

    _as_finite_number(
        loss_config.get('SMOOTH_L1_BETA', 1.0),
        'DISTILLATION.LOSS.SMOOTH_L1_BETA',
        minimum=0.0,
    )
    _as_finite_number(
        loss_config.get('ERROR_CLAMP', None),
        'DISTILLATION.LOSS.ERROR_CLAMP',
        minimum=0.0,
        allow_none=True,
    )
    _as_finite_number(
        loss_config.get('TEACHER_GT_ERROR_MAX', None),
        'DISTILLATION.LOSS.TEACHER_GT_ERROR_MAX',
        minimum=0.0,
        allow_none=True,
    )

    use_gt_valid = loss_config.get('USE_GT_VALID_MASK', True)
    if not isinstance(use_gt_valid, bool):
        raise TypeError('DISTILLATION.LOSS.USE_GT_VALID_MASK must be boolean')

    _as_finite_number(
        loss_config.get('TEACHER_GT_REL_ERROR_MAX', None),
        'DISTILLATION.LOSS.TEACHER_GT_REL_ERROR_MAX',
        minimum=0.0,
        allow_none=True,
    )
    _as_finite_number(
        loss_config.get('REL_ERROR_EPS', 1.0),
        'DISTILLATION.LOSS.REL_ERROR_EPS',
        minimum=1e-6,
    )
    _as_finite_number(
        loss_config.get('SMALL_DISP_MAX', None),
        'DISTILLATION.LOSS.SMALL_DISP_MAX',
        minimum=0.0,
        allow_none=True,
    )
    _as_finite_number(
        loss_config.get('SMALL_DISP_TEACHER_GT_ERROR_MAX', None),
        'DISTILLATION.LOSS.SMALL_DISP_TEACHER_GT_ERROR_MAX',
        minimum=0.0,
        allow_none=True,
    )
    _as_finite_number(
        loss_config.get('GT_EDGE_MAX', None),
        'DISTILLATION.LOSS.GT_EDGE_MAX',
        minimum=0.0,
        allow_none=True,
    )

    perturb_config = distill_config.get('STUDENT_PERTURBATION', {})
    if not isinstance(perturb_config, dict):
        raise TypeError('DISTILLATION.STUDENT_PERTURBATION must be a dictionary')
    perturb_enabled = perturb_config.get('ENABLED', False)
    if not isinstance(perturb_enabled, bool):
        raise TypeError('DISTILLATION.STUDENT_PERTURBATION.ENABLED must be boolean')
    symmetric_stereo = perturb_config.get('SYMMETRIC_STEREO', True)
    if not isinstance(symmetric_stereo, bool):
        raise TypeError(
            'DISTILLATION.STUDENT_PERTURBATION.SYMMETRIC_STEREO must be boolean'
        )
    apply_prob = _as_finite_number(
        perturb_config.get('APPLY_PROB', 1.0),
        'DISTILLATION.STUDENT_PERTURBATION.APPLY_PROB',
        minimum=0.0,
    )
    if apply_prob > 1.0:
        raise ValueError(
            'DISTILLATION.STUDENT_PERTURBATION.APPLY_PROB must be <= 1.0'
        )
    _as_finite_number(
        perturb_config.get('BRIGHTNESS', 0.0),
        'DISTILLATION.STUDENT_PERTURBATION.BRIGHTNESS',
        minimum=0.0,
    )
    _as_finite_number(
        perturb_config.get('CONTRAST', 0.0),
        'DISTILLATION.STUDENT_PERTURBATION.CONTRAST',
        minimum=0.0,
    )
    gamma_min = _as_finite_number(
        perturb_config.get('GAMMA_MIN', 1.0),
        'DISTILLATION.STUDENT_PERTURBATION.GAMMA_MIN',
        minimum=1e-6,
    )
    gamma_max = _as_finite_number(
        perturb_config.get('GAMMA_MAX', 1.0),
        'DISTILLATION.STUDENT_PERTURBATION.GAMMA_MAX',
        minimum=1e-6,
    )
    if gamma_min > gamma_max:
        raise ValueError(
            'DISTILLATION.STUDENT_PERTURBATION.GAMMA_MIN must be <= GAMMA_MAX'
        )
    _as_finite_number(
        perturb_config.get('GAUSSIAN_NOISE_STD', 0.0),
        'DISTILLATION.STUDENT_PERTURBATION.GAUSSIAN_NOISE_STD',
        minimum=0.0,
    )

    return distill_config


def _extract_disparity(prediction):
    if torch.is_tensor(prediction):
        return prediction
    if isinstance(prediction, dict):
        if 'disp_up' not in prediction:
            raise KeyError("Teacher prediction dict must contain 'disp_up'")
        return prediction['disp_up']
    if isinstance(prediction, (tuple, list)):
        if not prediction:
            raise ValueError('Teacher prediction sequence is empty')
        return prediction[0]
    raise TypeError(
        'Teacher prediction must be a Tensor, dict, tuple, or list, '
        f'got {type(prediction).__name__}'
    )


def build_teacher_model(distill_config, max_disp, device, logger):
    del max_disp
    checkpoint = distill_config['TEACHER']['CHECKPOINT']
    teacher_config = {
        'MODEL': {
            'PRETRAINED': checkpoint,
        },
    }
    teacher = build_baseline_model(
        teacher_config,
        device,
        logger,
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def _uniform_batch(batch_size, device, minimum, maximum):
    values = torch.rand(
        batch_size,
        1,
        1,
        1,
        device=device,
    )
    return values * (maximum - minimum) + minimum


def apply_student_photometric_perturbation(left, right, perturb_config):
    if not perturb_config or not perturb_config.get('ENABLED', False):
        return left, right, 0.0

    apply_prob = float(perturb_config.get('APPLY_PROB', 1.0))
    if apply_prob <= 0.0:
        return left, right, 0.0

    batch_size = left.shape[0]
    device = left.device
    sample_mask = (
        torch.rand(batch_size, 1, 1, 1, device=device) < apply_prob
    ).float()
    if sample_mask.sum().item() <= 0:
        return left, right, 0.0

    symmetric_stereo = perturb_config.get('SYMMETRIC_STEREO', True)
    left_float = left.float()
    right_float = right.float()
    left_work = left_float / 255.0
    right_work = right_float / 255.0

    brightness = float(perturb_config.get('BRIGHTNESS', 0.0))
    if brightness > 0.0:
        left_delta = _uniform_batch(batch_size, device, -brightness, brightness)
        right_delta = left_delta if symmetric_stereo else _uniform_batch(
            batch_size,
            device,
            -brightness,
            brightness,
        )
        left_work = left_work + sample_mask * left_delta
        right_work = right_work + sample_mask * right_delta

    contrast = float(perturb_config.get('CONTRAST', 0.0))
    if contrast > 0.0:
        left_factor = _uniform_batch(
            batch_size,
            device,
            1.0 - contrast,
            1.0 + contrast,
        )
        right_factor = left_factor if symmetric_stereo else _uniform_batch(
            batch_size,
            device,
            1.0 - contrast,
            1.0 + contrast,
        )
        left_mean = left_work.mean(dim=(2, 3), keepdim=True)
        right_mean = right_work.mean(dim=(2, 3), keepdim=True)
        left_work = left_mean + (left_work - left_mean) * left_factor
        right_work = right_mean + (right_work - right_mean) * right_factor

    gamma_min = float(perturb_config.get('GAMMA_MIN', 1.0))
    gamma_max = float(perturb_config.get('GAMMA_MAX', 1.0))
    if gamma_min != 1.0 or gamma_max != 1.0:
        left_gamma = _uniform_batch(batch_size, device, gamma_min, gamma_max)
        right_gamma = left_gamma if symmetric_stereo else _uniform_batch(
            batch_size,
            device,
            gamma_min,
            gamma_max,
        )
        left_work = left_work.clamp(0.0, 1.0).pow(left_gamma)
        right_work = right_work.clamp(0.0, 1.0).pow(right_gamma)

    noise_std = float(perturb_config.get('GAUSSIAN_NOISE_STD', 0.0)) / 255.0
    if noise_std > 0.0:
        left_noise = torch.randn_like(left_work)
        right_noise = left_noise if symmetric_stereo else torch.randn_like(right_work)
        left_work = left_work + sample_mask * left_noise * noise_std
        right_work = right_work + sample_mask * right_noise * noise_std

    left_perturbed = (left_work.clamp(0.0, 1.0) * 255.0).to(dtype=left.dtype)
    right_perturbed = (right_work.clamp(0.0, 1.0) * 255.0).to(dtype=right.dtype)

    left_output = torch.where(sample_mask.bool(), left_perturbed, left_float)
    right_output = torch.where(sample_mask.bool(), right_perturbed, right_float)
    return (
        left_output.to(dtype=left.dtype),
        right_output.to(dtype=right.dtype),
        sample_mask.mean().item(),
    )


def _gt_edge_magnitude(gt):
    dx = torch.zeros_like(gt)
    dy = torch.zeros_like(gt)
    dx[..., :, :-1] = (gt[..., :, 1:] - gt[..., :, :-1]).abs()
    dy[..., :-1, :] = (gt[..., 1:, :] - gt[..., :-1, :]).abs()
    return torch.maximum(dx, dy)


def distillation_disparity_loss(
        student_disp,
        teacher_disp,
        gt_disp,
        valid,
        max_disp,
        loss_config,
):
    if student_disp.shape != teacher_disp.shape:
        teacher_disp = F.interpolate(
            teacher_disp.float(),
            size=student_disp.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

    if student_disp.shape != gt_disp.shape:
        raise ValueError(
            'student_disp and gt_disp must have the same shape, got '
            f'{student_disp.shape} and {gt_disp.shape}'
        )
    if valid is not None and valid.shape != gt_disp.shape:
        raise ValueError(
            'valid and gt_disp must have the same shape, got '
            f'{valid.shape} and {gt_disp.shape}'
        )

    student = student_disp.float()
    teacher = teacher_disp.float().detach()
    gt = gt_disp.float()

    teacher_mask = (
        torch.isfinite(student)
        & torch.isfinite(teacher)
        & (teacher >= 0.0)
        & (teacher < float(max_disp))
    )

    gt_valid_mask = (
        torch.isfinite(gt)
        & (gt >= 0.0)
        & (gt < float(max_disp))
    )
    if valid is not None:
        gt_valid_mask = gt_valid_mask & (valid > 0.0)

    if loss_config.get('USE_GT_VALID_MASK', True):
        teacher_mask = teacher_mask & gt_valid_mask

    teacher_error = (teacher - gt).abs()

    teacher_gt_error_max = loss_config.get('TEACHER_GT_ERROR_MAX')
    if teacher_gt_error_max is not None:
        teacher_mask = teacher_mask & gt_valid_mask & (
            teacher_error <= float(teacher_gt_error_max)
        )

    teacher_gt_rel_error_max = loss_config.get('TEACHER_GT_REL_ERROR_MAX')
    if teacher_gt_rel_error_max is not None:
        rel_eps = float(loss_config.get('REL_ERROR_EPS', 1.0))
        teacher_rel_error = teacher_error / gt.abs().clamp_min(rel_eps)
        teacher_mask = teacher_mask & gt_valid_mask & (
            teacher_rel_error <= float(teacher_gt_rel_error_max)
        )

    small_disp_max = loss_config.get('SMALL_DISP_MAX')
    small_disp_error_max = loss_config.get('SMALL_DISP_TEACHER_GT_ERROR_MAX')
    if small_disp_max is not None and small_disp_error_max is not None:
        small_disp_mask = gt < float(small_disp_max)
        teacher_mask = teacher_mask & gt_valid_mask & (
            (~small_disp_mask) | (teacher_error <= float(small_disp_error_max))
        )

    gt_edge_max = loss_config.get('GT_EDGE_MAX')
    if gt_edge_max is not None:
        gt_edge = _gt_edge_magnitude(gt)
        teacher_mask = teacher_mask & gt_valid_mask & (
            gt_edge <= float(gt_edge_max)
        )

    valid_pixels = float(teacher_mask.sum().item())
    total_pixels = float(teacher_mask.numel())

    zero_loss = student.sum() * 0.0
    if valid_pixels <= 0:
        return zero_loss, {
            'kd_valid_ratio': 0.0,
            'teacher_gt_epe': 0.0,
        }

    pixel_loss = F.smooth_l1_loss(
        student[teacher_mask],
        teacher[teacher_mask],
        reduction='none',
        beta=float(loss_config.get('SMOOTH_L1_BETA', 1.0)),
    )

    error_clamp = loss_config.get('ERROR_CLAMP')
    if error_clamp is not None and float(error_clamp) > 0:
        pixel_loss = pixel_loss.clamp(max=float(error_clamp))

    gt_metric_mask = teacher_mask & gt_valid_mask
    if gt_metric_mask.any():
        teacher_gt_epe = (
            teacher[gt_metric_mask] - gt[gt_metric_mask]
        ).abs().mean().item()
    else:
        teacher_gt_epe = 0.0

    return pixel_loss.mean(), {
        'kd_valid_ratio': valid_pixels / total_pixels,
        'teacher_gt_epe': teacher_gt_epe,
    }


def train_epoch_distill_ddp(
        student_model,
        teacher_model,
        train_loader,
        optimizer,
        scaler,
        device,
        max_disp,
        use_amp,
        loss_weights,
        distill_loss_config,
        distill_perturb_config,
        writer,
        global_step,
        scalar_interval,
        image_interval,
        logger,
        epoch,
        total_epochs,
        text_interval,
        grad_clip=0.0,
):
    student_model.train()
    teacher_model.eval()

    num_batches = len(train_loader)
    if num_batches == 0:
        raise ValueError('train_loader must contain at least one batch')

    running_stats = {key: 0.0 for key in DISTILL_STAT_KEYS}
    progress = BatchProgressLogger(
        logger=logger,
        phase='Training',
        epoch=epoch,
        total_epochs=total_epochs,
        num_batches=num_batches,
        interval=text_interval,
        device=device,
    )

    lambda_kd = float(distill_loss_config['LAMBDA_KD_DISP'])

    for batch_index, batch in enumerate(train_loader, start=1):
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

        (
            student_left_batch,
            student_right_batch,
            perturb_applied_ratio,
        ) = apply_student_photometric_perturbation(
            left_batch,
            right_batch,
            distill_perturb_config,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=use_amp):
                teacher_prediction = teacher_model(
                    left_batch,
                    right_batch,
                    max_disp=max_disp,
                    test_mode=True,
                )
            teacher_disp = _extract_disparity(teacher_prediction).detach()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            supervised_loss, supervised_stats, outputs = compute_las2_s_hfe_loss(
                model=student_model,
                left=student_left_batch,
                right=student_right_batch,
                gt_disp=gt_disp_batch,
                valid=valid_batch,
                max_disp=max_disp,
                lambda_cvc_c0=loss_weights['LAMBDA_CVC_C0'],
                lambda_cvc_c2=loss_weights['LAMBDA_CVC_C2'],
                lambda_d0=loss_weights['LAMBDA_D0'],
                lambda_d2=loss_weights['LAMBDA_D2'],
                lambda_disp=loss_weights['LAMBDA_DISP'],
                stage='joint',
            )

        kd_loss, kd_stats = distillation_disparity_loss(
            student_disp=outputs['disp_up'],
            teacher_disp=teacher_disp,
            gt_disp=gt_disp_batch,
            valid=valid_batch,
            max_disp=max_disp,
            loss_config=distill_loss_config,
        )
        weighted_kd_loss = lambda_kd * kd_loss
        total_loss = supervised_loss + weighted_kd_loss

        scaler.scale(total_loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in student_model.parameters()
                    if parameter.requires_grad
                ],
                grad_clip,
            )
        scaler.step(optimizer)
        scaler.update()

        loss_values = {
            key: value.item()
            for key, value in supervised_stats.items()
        }
        stat_values = {
            'loss': total_loss.detach().item(),
            'loss_supervised': supervised_loss.detach().item(),
            'loss_kd_disp': kd_loss.detach().item(),
            'weighted_loss_kd_disp': weighted_kd_loss.detach().item(),
            'kd_valid_ratio': kd_stats['kd_valid_ratio'],
            'teacher_gt_epe': kd_stats['teacher_gt_epe'],
            'perturb_applied_ratio': perturb_applied_ratio,
            'loss_cvc_c0': loss_values['loss_cvc_c0'],
            'loss_cvc_c2': loss_values['loss_cvc_c2'],
            'loss_d0': loss_values['loss_d0'],
            'loss_d2': loss_values['loss_d2'],
            'loss_disp': loss_values['loss_disp'],
            'weighted_loss_cvc_c0': loss_values['weighted_loss_cvc_c0'],
            'weighted_loss_cvc_c2': loss_values['weighted_loss_cvc_c2'],
            'weighted_loss_d0': loss_values['weighted_loss_d0'],
            'weighted_loss_d2': loss_values['weighted_loss_d2'],
            'weighted_loss_disp': loss_values['weighted_loss_disp'],
        }

        for key in running_stats:
            running_stats[key] += stat_values[key]

        step = global_step + batch_index

        if (
            writer is not None
            and scalar_interval > 0
            and step % scalar_interval == 0
        ):
            for key in (
                'loss',
                'loss_supervised',
                'loss_kd_disp',
                'weighted_loss_kd_disp',
                'kd_valid_ratio',
                'teacher_gt_epe',
                'perturb_applied_ratio',
            ):
                writer.add_scalar(f'train/step/{key}', stat_values[key], step)

        if (
            writer is not None
            and image_interval > 0
            and step % image_interval == 0
        ):
            _log_stereo_images(
                writer=writer,
                prefix='train',
                left=student_left_batch,
                right=student_right_batch,
                gt_disp=gt_disp_batch,
                pred_disp=outputs['disp_up'],
                max_disp=max_disp,
                step=step,
            )
            writer.add_images(
                'train/teacher_disp',
                (teacher_disp[:1].detach().float().cpu() / max_disp).clamp(0, 1),
                step,
            )

        progress.finish_batch(
            batch_index=batch_index,
            timing=batch_timing,
            loss=stat_values['loss'],
            average_loss=running_stats['loss'] / batch_index,
            learning_rate=optimizer.param_groups[0]['lr'],
            global_step=step,
            metrics={
                'Sup': stat_values['loss_supervised'],
                'KD': stat_values['loss_kd_disp'],
                'KDr': stat_values['kd_valid_ratio'],
            },
        )

    return _reduce_stat_sums(
        running_stats,
        num_batches,
        device,
        DISTILL_STAT_KEYS,
    )


def save_distill_checkpoint(
        checkpoint_path,
        epoch,
        raw_model,
        optimizer,
        scheduler,
        scaler,
        best_val_loss,
        best_val_epe,
        best_val_sq_rel,
        loss_variant,
        loss_weights,
        distill_config,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    architecture_version = distill_config.get(
        'ARCHITECTURE_VERSION',
        'las2_s_hfe_distill_v1',
    )
    checkpoint = {
        'architecture_version': architecture_version,
        'loss_variant': loss_variant,
        'loss_weights': dict(loss_weights),
        'distillation': dict(distill_config),
        'max_disp': getattr(raw_model, 'max_disp', None),
        'cost_channels': getattr(raw_model, 'cost_channels', None),
        'model_size': 's',
        'epoch': epoch,
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'scaler': scaler.state_dict(),
        'best_val_loss': best_val_loss,
        'best_val_epe': best_val_epe,
        'best_val_sq_rel': best_val_sq_rel,
    }
    torch.save(checkpoint, checkpoint_path)


def load_distill_training_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        scaler,
        device,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f'Distillation checkpoint not found: {checkpoint_path}'
        )

    checkpoint = load_checkpoint_weights(
        checkpoint_path,
        map_location=device,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError('Distillation checkpoint must be a dictionary')

    architecture_version = checkpoint.get('architecture_version')
    supported_architectures = {
        'las2_s_hfe_litematch_cvs_v1',
        'las2_s_hfe_fair_final_disp_v1',
        'las2_s_hfe_ablation_v1',
        'las2_s_hfe_loss_ablation_v1',
        'las2_s_hfe_distill_v1',
        'las2_s_hfe_distill_v2',
    }
    if (
        architecture_version is not None
        and architecture_version not in supported_architectures
    ):
        raise ValueError(
            f'Unsupported checkpoint architecture: {architecture_version}'
        )

    checkpoint_max_disp = checkpoint.get('max_disp')
    model_max_disp = getattr(model, 'max_disp', None)
    if (
        checkpoint_max_disp is not None
        and model_max_disp is not None
        and checkpoint_max_disp != model_max_disp
    ):
        raise ValueError(
            f'Checkpoint max_disp {checkpoint_max_disp} does not match '
            f'model max_disp {model_max_disp}'
        )

    state_dict = checkpoint.get('model')
    if not isinstance(state_dict, dict):
        raise KeyError("Distillation checkpoint must contain 'model'")
    if state_dict and all(key.startswith('module.') for key in state_dict):
        state_dict = {
            key[len('module.'):]: value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)

    if 'optimizer' not in checkpoint:
        raise KeyError("Distillation checkpoint must contain 'optimizer'")
    optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None:
        scheduler_state = checkpoint.get('scheduler')
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        else:
            scheduler.step(checkpoint.get('epoch', 0))

    if 'scaler' not in checkpoint:
        raise KeyError("Distillation checkpoint must contain 'scaler'")
    scaler.load_state_dict(checkpoint['scaler'])

    epoch = checkpoint.get('epoch')
    best_val_loss = checkpoint.get('best_val_loss')
    best_val_epe = checkpoint.get('best_val_epe', float('inf'))
    best_val_sq_rel = checkpoint.get('best_val_sq_rel', float('inf'))

    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f'Invalid checkpoint epoch: {epoch}')
    if not isinstance(best_val_loss, (int, float)):
        raise ValueError(f'Invalid checkpoint best_val_loss: {best_val_loss}')
    if not isinstance(best_val_epe, (int, float)):
        raise ValueError(f'Invalid checkpoint best_val_epe: {best_val_epe}')
    if not isinstance(best_val_sq_rel, (int, float)):
        raise ValueError(
            f'Invalid checkpoint best_val_sq_rel: {best_val_sq_rel}'
        )

    return (
        epoch,
        float(best_val_loss),
        float(best_val_epe),
        float(best_val_sq_rel),
    )


def run_training_ddp(args):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    is_main = rank == 0

    config, _ = load_loss_ablation_config(args.config)
    distill_config = validate_distillation_config(config)

    train_config = config['TRAIN']
    logging_config = config['LOGGING']
    model_config = config['MODEL']
    max_disp = model_config['MAX_DISP']

    base_output_dir = _resolve_project_path(train_config['OUTPUT_DIR'])
    output_dir = (
        base_output_dir.parent / args.output_subdir
        if args.output_subdir
        else base_output_dir
    )
    log_dir = output_dir / 'tensorboard'
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = build_ddp_logger(output_dir / 'train.log') if is_main else None

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
    backend = 'nccl' if use_cuda else 'gloo'
    dist.init_process_group(backend=backend)
    device = torch.device('cuda', local_rank) if use_cuda else torch.device('cpu')

    seed = train_config.get('SEED')
    set_random_seed(seed, rank)

    if is_main:
        log_config(logger, args.config, config)
        logger.info('Job: modified_l1_distill')
        logger.info(
            'DDP distillation: world_size=%d rank=0/%d local_rank=%d',
            world_size,
            world_size,
            local_rank,
        )
        logger.info('Teacher checkpoint: %s', distill_config['TEACHER']['CHECKPOINT'])
        logger.info('Output directory: %s', output_dir)
        logger.info('TensorBoard directory: %s', log_dir)

    train_loader, valid_loader, train_sampler = build_ddp_dataloaders(
        config,
        rank,
        world_size,
    )

    raw_student = build_hfe_model(
        config,
        device,
        logger=logger if is_main else None,
    )
    teacher = build_teacher_model(
        distill_config,
        max_disp,
        device,
        logger if is_main else None,
    )

    student = DDP(
        raw_student,
        device_ids=[local_rank] if use_cuda else None,
        output_device=local_rank if use_cuda else None,
        find_unused_parameters=True,
    )

    learning_rate = train_config['LR'] * args.lr_scale
    optimizer, scaler, use_amp = build_optimizer_and_scaler(
        config,
        student,
        device,
        learning_rate=learning_rate,
    )
    total_epochs = args.epochs if args.epochs is not None else train_config['EPOCHS']
    scheduler = build_scheduler(config, optimizer, epochs=total_epochs)
    text_interval = resolve_text_interval(logging_config)

    if is_main:
        logger.info('----------- DATA LOADERS -----------')
        logger.info(
            'Training dataset: samples=%d per-rank batches=%d '
            'per-rank batch_size=%s effective_batch_size=%d',
            len(train_loader.dataset),
            len(train_loader),
            train_loader.batch_size,
            train_loader.batch_size * world_size,
        )
        logger.info(
            'Validation dataset (rank 0 only): samples=%d batches=%d',
            len(valid_loader.dataset),
            len(valid_loader),
        )
        logger.info('----------- OPTIMIZATION -----------')
        logger.info('LR scale: %.4f (base LR -> %.8f)', args.lr_scale, learning_rate)
        logger.info(
            'Distillation loss: lambda=%.4f beta=%.4f clamp=%s '
            'teacher_gt_error_max=%s teacher_gt_rel_error_max=%s '
            'small_disp_max=%s small_disp_teacher_gt_error_max=%s '
            'gt_edge_max=%s use_gt_valid=%s',
            distill_config['LOSS']['LAMBDA_KD_DISP'],
            distill_config['LOSS'].get('SMOOTH_L1_BETA', 1.0),
            distill_config['LOSS'].get('ERROR_CLAMP'),
            distill_config['LOSS'].get('TEACHER_GT_ERROR_MAX'),
            distill_config['LOSS'].get('TEACHER_GT_REL_ERROR_MAX'),
            distill_config['LOSS'].get('SMALL_DISP_MAX'),
            distill_config['LOSS'].get('SMALL_DISP_TEACHER_GT_ERROR_MAX'),
            distill_config['LOSS'].get('GT_EDGE_MAX'),
            distill_config['LOSS'].get('USE_GT_VALID_MASK', True),
        )
        logger.info(
            'Student perturbation: %s',
            distill_config.get('STUDENT_PERTURBATION', {}),
        )
        log_optimization_context(
            logger=logger,
            model=raw_student,
            optimizer=optimizer,
            scheduler=scheduler,
            use_amp=use_amp,
            device=device,
            text_interval=text_interval,
        )

    writer = SummaryWriter(log_dir=str(log_dir)) if is_main else None

    start_epoch = 0
    best_val_loss = float('inf')
    best_val_epe = float('inf')
    best_val_sq_rel = float('inf')

    resume_value = train_config.get('RESUME')
    if resume_value is not None and str(resume_value).lower() != 'none':
        resume_path = _resolve_project_path(resume_value)
        (
            start_epoch,
            best_val_loss,
            best_val_epe,
            best_val_sq_rel,
        ) = load_distill_training_checkpoint(
            checkpoint_path=resume_path,
            model=raw_student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        if is_main:
            logger.info('Resumed student training from: %s', resume_path)

    if start_epoch > total_epochs:
        raise ValueError(
            f'Checkpoint epoch {start_epoch} exceeds configured EPOCHS {total_epochs}'
        )

    loss_weights = train_config['LOSS']
    for loss_key in HFE_VALID_LOSS_KEYS:
        if loss_key not in loss_weights:
            raise KeyError(f'TRAIN.LOSS.{loss_key} is required')

    variant_name = config['LOSS_ABLATION']['NAME']
    scalar_interval = logging_config['SCALAR_INTERVAL']
    image_interval = logging_config['IMAGE_INTERVAL']
    num_batches_per_rank = len(train_loader)
    global_step = start_epoch * num_batches_per_rank

    grad_clip = train_config.get('GRAD_CLIP', 0.0)
    if (
        not isinstance(grad_clip, (int, float))
        or isinstance(grad_clip, bool)
        or grad_clip < 0
    ):
        raise ValueError(f'TRAIN.GRAD_CLIP must be non-negative, got {grad_clip}')
    if is_main and grad_clip > 0:
        logger.info('Gradient clipping enabled: max_norm=%.4f', grad_clip)

    try:
        for epoch_index in range(start_epoch, total_epochs):
            epoch_number = epoch_index + 1
            if is_main:
                logger.info('Starting epoch [%d/%d]', epoch_number, total_epochs)
            train_sampler.set_epoch(epoch_index)

            train_stats = train_epoch_distill_ddp(
                student_model=student,
                teacher_model=teacher,
                train_loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                max_disp=max_disp,
                use_amp=use_amp,
                loss_weights=loss_weights,
                distill_loss_config=distill_config['LOSS'],
                distill_perturb_config=distill_config.get(
                    'STUDENT_PERTURBATION',
                    {},
                ),
                writer=writer,
                global_step=global_step,
                scalar_interval=scalar_interval,
                image_interval=image_interval,
                logger=logger,
                epoch=epoch_number,
                total_epochs=total_epochs,
                text_interval=text_interval,
                grad_clip=grad_clip,
            )
            global_step += num_batches_per_rank

            is_best = False
            is_best_sq_rel = False
            if is_main:
                valid_stats = hfe_validate_epoch(
                    model=raw_student,
                    valid_loader=valid_loader,
                    device=device,
                    max_disp=max_disp,
                    use_amp=use_amp,
                    lambda_cvc_c0=loss_weights['LAMBDA_CVC_C0'],
                    lambda_cvc_c2=loss_weights['LAMBDA_CVC_C2'],
                    lambda_d0=loss_weights['LAMBDA_D0'],
                    lambda_d2=loss_weights['LAMBDA_D2'],
                    lambda_disp=loss_weights['LAMBDA_DISP'],
                    writer=writer,
                    global_step=global_step,
                    image_interval=image_interval,
                    stage='joint',
                    logger=logger,
                    epoch=epoch_number,
                    total_epochs=total_epochs,
                    text_interval=text_interval,
                    progress_label='Validation',
                )

                for key, value in train_stats.items():
                    writer.add_scalar(f'train/epoch/{key}', value, epoch_number)
                for key, value in valid_stats.items():
                    writer.add_scalar(f'valid/epoch/{key}', value, epoch_number)
                writer.add_scalar(
                    'train/epoch/learning_rate',
                    optimizer.param_groups[0]['lr'],
                    epoch_number,
                )

                best_val_loss = min(best_val_loss, valid_stats['loss'])
                is_best = valid_stats['epe'] < best_val_epe
                if is_best:
                    best_val_epe = valid_stats['epe']
                valid_sq_rel = valid_stats.get('sq_rel', float('inf'))
                is_best_sq_rel = valid_sq_rel < best_val_sq_rel
                if is_best_sq_rel:
                    best_val_sq_rel = valid_sq_rel

            if scheduler is not None:
                scheduler.step()

            if is_main:
                save_distill_checkpoint(
                    checkpoint_path=output_dir / 'latest.pth',
                    epoch=epoch_number,
                    raw_model=raw_student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_val_loss=best_val_loss,
                    best_val_epe=best_val_epe,
                    best_val_sq_rel=best_val_sq_rel,
                    loss_variant=variant_name,
                    loss_weights=loss_weights,
                    distill_config=distill_config,
                )
                if is_best:
                    save_distill_checkpoint(
                        checkpoint_path=output_dir / 'best.pth',
                        epoch=epoch_number,
                        raw_model=raw_student,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        best_val_loss=best_val_loss,
                        best_val_epe=best_val_epe,
                        best_val_sq_rel=best_val_sq_rel,
                        loss_variant=variant_name,
                        loss_weights=loss_weights,
                        distill_config=distill_config,
                    )
                if is_best_sq_rel:
                    save_distill_checkpoint(
                        checkpoint_path=output_dir / 'best_sqrel.pth',
                        epoch=epoch_number,
                        raw_model=raw_student,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        best_val_loss=best_val_loss,
                        best_val_epe=best_val_epe,
                        best_val_sq_rel=best_val_sq_rel,
                        loss_variant=variant_name,
                        loss_weights=loss_weights,
                        distill_config=distill_config,
                    )
                writer.flush()
                logger.info(
                    'Epoch [%d/%d] lr=%.8f train_total=%.6f '
                    'train_sup=%.6f train_kd=%.6f kd_ratio=%.4f '
                    'perturb=%.4f teacher_gt_epe=%.6f valid_epe=%.6f '
                    'valid_d1=%.6f valid_sqrel=%.6f best_epe=%.6f '
                    'best_sqrel=%.6f best_updated=%s best_sqrel_updated=%s',
                    epoch_number,
                    total_epochs,
                    optimizer.param_groups[0]['lr'],
                    train_stats['loss'],
                    train_stats['loss_supervised'],
                    train_stats['loss_kd_disp'],
                    train_stats['kd_valid_ratio'],
                    train_stats['perturb_applied_ratio'],
                    train_stats['teacher_gt_epe'],
                    valid_stats['epe'],
                    valid_stats['d1'],
                    valid_stats.get('sq_rel', float('inf')),
                    best_val_epe,
                    best_val_sq_rel,
                    is_best,
                    is_best_sq_rel,
                )

            dist.barrier()
    finally:
        if writer is not None:
            writer.close()
        dist.destroy_process_group()

    if is_main:
        logger.info('Distillation training completed through epoch %d', total_epochs)


def main():
    parser = argparse.ArgumentParser(
        description='DDP teacher-student distillation for LAS2-S HFE/CVS',
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to the distillation YAML configuration',
    )
    parser.add_argument(
        '--output-subdir',
        default=None,
        help='Replace the final component of TRAIN.OUTPUT_DIR',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override TRAIN.EPOCHS',
    )
    parser.add_argument(
        '--lr-scale',
        type=float,
        default=1.0,
        help='Multiply TRAIN.LR by this factor',
    )
    args = parser.parse_args()
    run_training_ddp(args)


if __name__ == '__main__':
    main()
