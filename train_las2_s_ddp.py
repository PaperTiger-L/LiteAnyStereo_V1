"""Multi-GPU DDP training entry for the LAS2-S Baseline and Modified-L1 jobs.

Launch with torchrun, one process per GPU:

    torchrun --standalone --nproc_per_node=8 train_las2_s_ddp.py \
        --config tmp/experiments/outside_grass_20260417_seed2026/configs/baseline_seed2026.yaml \
        --job baseline --output-subdir baseline_seed2026_ddp8

Design notes:
  - Training batches are sharded across ranks with DistributedSampler;
    the effective batch size becomes TRAIN.BATCH_SIZE * world_size.
  - Only rank 0 owns the text log, TensorBoard writer, validation pass
    and checkpoint files, so each run writes one train.log, tensorboard
    directory, latest.pth, and best.pth.
  - Checkpoints store the unwrapped model state dict, so they remain
    compatible with evaluate_las2_s.py and RESUME on a single card.
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from las2_training_logging import (
    BatchProgressLogger,
    log_config,
    log_optimization_context,
    resolve_text_interval,
)
from las2_s_baseline_train_utils import (
    baseline_disparity_loss,
    build_baseline_model,
    load_baseline_config,
    validate_epoch as baseline_validate_epoch,
)
from las2_s_hfe_train_utils import (
    CustomDataset,
    _log_stereo_images,
    _resolve_project_path,
    _seed_data_worker,
    build_hfe_model,
    build_optimizer_and_scaler,
    build_scheduler,
    collate_stereo_batch,
    collect_data_split_files,
    compute_las2_s_hfe_loss,
    load_training_checkpoint,
    _tensor_stats_to_cpu_scalars,
    validate_epoch as hfe_validate_epoch,
)
from las2_s_loss_config import load_loss_ablation_config


HFE_STAT_KEYS = (
    'loss',
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


def load_ddp_config(job, config_path):
    if job == 'baseline':
        return load_baseline_config(config_path)
    if job in ('modified_l1', 'optimized_multiscale'):
        config, _ = load_loss_ablation_config(config_path)
        architecture = config['MODEL'].get(
            'ARCHITECTURE',
            'las2_s_hfe_cvs_v1',
        )
        if job == 'optimized_multiscale':
            expected_architecture = 'las2_s_hfe_multiscale_v1'
            if architecture != expected_architecture:
                raise ValueError(
                    'optimized_multiscale requires '
                    f'MODEL.ARCHITECTURE={expected_architecture}, '
                    f'got {architecture}'
                )
        elif architecture != 'las2_s_hfe_cvs_v1':
            raise ValueError(
                'modified_l1 requires MODEL.ARCHITECTURE='
                'las2_s_hfe_cvs_v1, '
                f'got {architecture}'
            )
        return config
    raise ValueError(f'Unsupported job: {job}')


def build_ddp_logger(log_path):
    logger = logging.getLogger('las2_s_ddp_training')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        log_path,
        mode='w',
        encoding='utf-8',
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def set_random_seed(seed, rank):
    if seed is None:
        return
    # Offset by rank so random crops differ across GPUs.
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def build_ddp_dataloaders(config, rank, world_size):
    data_root = config['DATA_ROOT']
    data_info = config['DATA_CONFIG']['DATA_INFOS'][0]
    data_split = data_info['DATA_SPLIT']
    augmentation_config = config['AUGMENTATION']
    model_config = config['MODEL']
    train_config = config['TRAIN']

    max_disp = model_config['MAX_DISP']
    seed = train_config.get('SEED')
    train_generator = None
    valid_generator = None
    worker_init_fn = None
    sampler_seed = 0
    if seed is not None:
        sampler_seed = seed
        train_generator = torch.Generator()
        train_generator.manual_seed(seed + rank)
        valid_generator = torch.Generator()
        valid_generator.manual_seed(seed + 1)
        worker_init_fn = _seed_data_worker

    train_dataset = CustomDataset(
        data_root=data_root,
        list_file=collect_data_split_files(data_split, 'TRAINING'),
        crop_size=augmentation_config['TRAIN_CROP_SIZE'],
        training=True,
        max_disp=max_disp,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=sampler_seed,
        drop_last=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['BATCH_SIZE'],
        shuffle=False,
        sampler=train_sampler,
        num_workers=train_config['NUM_WORKERS'],
        pin_memory=True,
        collate_fn=collate_stereo_batch,
        worker_init_fn=worker_init_fn,
        generator=train_generator,
    )

    valid_loader = None
    if rank == 0:
        valid_dataset = CustomDataset(
            data_root=data_root,
            list_file=collect_data_split_files(data_split, 'EVALUATING'),
            crop_size=augmentation_config['EVAL_CROP_SIZE'],
            training=False,
            max_disp=max_disp,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=train_config['BATCH_SIZE'],
            shuffle=False,
            num_workers=train_config['NUM_WORKERS'],
            pin_memory=True,
            collate_fn=collate_stereo_batch,
            worker_init_fn=worker_init_fn,
            generator=valid_generator,
        )
    return train_loader, valid_loader, train_sampler


def _reduce_stat_sums(stat_sums, num_batches_local, device, key_order):
    """Average per-rank running sums into a global per-batch mean dict."""
    values = [stat_sums[key] for key in key_order] + [float(num_batches_local)]
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_batches = tensor[-1].item()
    return {
        key: tensor[index].item() / total_batches
        for index, key in enumerate(key_order)
    }


def train_epoch_baseline_ddp(
        model,
        train_loader,
        optimizer,
        scaler,
        device,
        max_disp,
        use_amp,
        writer,
        global_step,
        scalar_interval,
        image_interval,
        logger,
        epoch,
        total_epochs,
        text_interval,
        grad_clip=0.0,
        scheduler=None,
):
    model.train()
    num_batches = len(train_loader)
    if num_batches == 0:
        raise ValueError('train_loader must contain at least one batch')

    running_loss = 0.0
    progress = BatchProgressLogger(
        logger=logger,
        phase='Training',
        epoch=epoch,
        total_epochs=total_epochs,
        num_batches=num_batches,
        interval=text_interval,
        device=device,
    )

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

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
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

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_applied = (
            scaler.get_scale() >= scale_before_step
        )

        loss_value = loss.item()
        running_loss += loss_value
        step = global_step + batch_index

        if (
            writer is not None
            and scalar_interval > 0
            and step % scalar_interval == 0
        ):
            writer.add_scalar('train/step/loss', loss_value, step)

        if (
            writer is not None
            and image_interval > 0
            and step % image_interval == 0
        ):
            _log_stereo_images(
                writer=writer,
                prefix='train',
                left=left_batch,
                right=right_batch,
                gt_disp=gt_disp_batch,
                pred_disp=pred_disp,
                max_disp=max_disp,
                step=step,
            )

        progress.finish_batch(
            batch_index=batch_index,
            timing=batch_timing,
            loss=loss_value,
            average_loss=running_loss / batch_index,
            learning_rate=optimizer.param_groups[0]['lr'],
            global_step=step,
        )
        if (
            scheduler is not None
            and getattr(scheduler, 'step_per_batch', False)
            and optimizer_step_applied
        ):
            scheduler.step()

    return _reduce_stat_sums(
        {'loss': running_loss},
        num_batches,
        device,
        ('loss',),
    )


def train_epoch_hfe_ddp(
        model,
        train_loader,
        optimizer,
        scaler,
        device,
        max_disp,
        use_amp,
        loss_weights,
        writer,
        global_step,
        scalar_interval,
        image_interval,
        logger,
        epoch,
        total_epochs,
        text_interval,
        grad_clip=0.0,
        disp_band_weight=None,
        disp_near_add=None,
        scheduler=None,
):
    model.train()
    num_batches = len(train_loader)
    if num_batches == 0:
        raise ValueError('train_loader must contain at least one batch')

    running_stats = {key: 0.0 for key in HFE_STAT_KEYS}
    progress = BatchProgressLogger(
        logger=logger,
        phase='Training',
        epoch=epoch,
        total_epochs=total_epochs,
        num_batches=num_batches,
        interval=text_interval,
        device=device,
    )

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

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            total_loss, loss_stats, outputs = compute_las2_s_hfe_loss(
                model=model,
                left=left_batch,
                right=right_batch,
                gt_disp=gt_disp_batch,
                valid=valid_batch,
                max_disp=max_disp,
                lambda_cvc_c0=loss_weights['LAMBDA_CVC_C0'],
                lambda_cvc_c2=loss_weights['LAMBDA_CVC_C2'],
                lambda_d0=loss_weights['LAMBDA_D0'],
                lambda_d2=loss_weights['LAMBDA_D2'],
                lambda_disp=loss_weights['LAMBDA_DISP'],
                stage='joint',
                disp_band_weight=disp_band_weight,
                disp_near_add=disp_near_add,
            )

        scaler.scale(total_loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                grad_clip,
            )
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_applied = (
            scaler.get_scale() >= scale_before_step
        )

        loss_values = _tensor_stats_to_cpu_scalars(loss_stats)
        for key in running_stats:
            running_stats[key] += loss_values[key]

        step = global_step + batch_index

        if (
            writer is not None
            and scalar_interval > 0
            and step % scalar_interval == 0
        ):
            for key in ('loss', 'loss_disp', 'loss_cvc_c0', 'loss_cvc_c2'):
                writer.add_scalar(
                    f'train/step/{key}',
                    loss_values[key],
                    step,
                )

        if (
            writer is not None
            and image_interval > 0
            and step % image_interval == 0
        ):
            _log_stereo_images(
                writer=writer,
                prefix='train',
                left=left_batch,
                right=right_batch,
                gt_disp=gt_disp_batch,
                pred_disp=outputs['disp_up'],
                max_disp=max_disp,
                step=step,
            )

        progress.finish_batch(
            batch_index=batch_index,
            timing=batch_timing,
            loss=loss_values['loss'],
            average_loss=running_stats['loss'] / batch_index,
            learning_rate=optimizer.param_groups[0]['lr'],
            global_step=step,
            metrics={
                'Disp': loss_values['loss_disp'],
                'CVC0': loss_values['loss_cvc_c0'],
                'CVC2': loss_values['loss_cvc_c2'],
            },
        )
        if (
            scheduler is not None
            and getattr(scheduler, 'step_per_batch', False)
            and optimizer_step_applied
        ):
            scheduler.step()

    return _reduce_stat_sums(
        running_stats,
        num_batches,
        device,
        HFE_STAT_KEYS,
    )


def save_ddp_checkpoint(
        checkpoint_path,
        epoch,
        raw_model,
        optimizer,
        scheduler,
        scaler,
        best_val_loss,
        best_val_epe,
        job,
        loss_variant=None,
        loss_weights=None,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if job == 'baseline':
        checkpoint = {
            'architecture_version': 'las2_s_baseline_las2_s_v1',
            'model_size': 's',
            'max_disp': getattr(raw_model, 'max_disp', None),
        }
    else:
        checkpoint = {
            'architecture_version': getattr(
                raw_model,
                'architecture_version',
                'las2_s_hfe_loss_ablation_v1',
            ),
            'loss_variant': loss_variant,
            'loss_weights': dict(loss_weights),
            'max_disp': getattr(raw_model, 'max_disp', None),
            'cost_channels': getattr(raw_model, 'cost_channels', None),
            'model_size': 's',
        }

    checkpoint.update({
        'epoch': epoch,
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        'scaler': scaler.state_dict(),
        'best_val_loss': best_val_loss,
        'best_val_epe': best_val_epe,
    })
    torch.save(checkpoint, checkpoint_path)


def run_training_ddp(args):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    is_main = rank == 0

    config = load_ddp_config(args.job, args.config)
    train_config = config['TRAIN']
    logging_config = config['LOGGING']
    model_config = config['MODEL']
    max_disp = model_config['MAX_DISP']

    base_output_dir = _resolve_project_path(train_config['OUTPUT_DIR'])
    if args.output_subdir:
        output_dir = base_output_dir.parent / args.output_subdir
    else:
        output_dir = base_output_dir
    log_dir = output_dir / 'tensorboard'
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = build_ddp_logger(output_dir / 'train.log') if is_main else None

    # Set the CUDA device before initializing NCCL so every rank binds
    # its own GPU instead of all defaulting to cuda:0. Falls back to
    # gloo/CPU when no CUDA is visible, which keeps smoke tests possible.
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
        logger.info('Job: %s', args.job)
        logger.info(
            'DDP training: world_size=%d rank=0/%d local_rank=%d',
            world_size,
            world_size,
            local_rank,
        )
        logger.info('Random seed: %s (+ per-rank offset)', seed)
        logger.info('Output directory: %s', output_dir)
        logger.info('TensorBoard directory: %s', log_dir)

    train_loader, valid_loader, train_sampler = build_ddp_dataloaders(
        config,
        rank,
        world_size,
    )

    if args.job == 'baseline':
        raw_model = build_baseline_model(
            config,
            device,
            logger if is_main else None,
        )
    else:
        raw_model = build_hfe_model(
            config,
            device,
            logger=logger if is_main else None,
        )

    model = DDP(
        raw_model,
        device_ids=[local_rank] if use_cuda else None,
        output_device=local_rank if use_cuda else None,
        find_unused_parameters=False,
    )

    if not np.isfinite(args.lr_scale) or args.lr_scale <= 0:
        raise ValueError(
            f'--lr-scale must be a finite positive number, got {args.lr_scale}'
        )
    learning_rate = train_config['LR'] * args.lr_scale
    optimizer, scaler, use_amp = build_optimizer_and_scaler(
        config,
        model,
        device,
        learning_rate=learning_rate,
    )
    total_epochs = (
        args.epochs
        if args.epochs is not None
        else train_config['EPOCHS']
    )
    scheduler = build_scheduler(
        config,
        optimizer,
        epochs=total_epochs,
        steps_per_epoch=len(train_loader),
        lr_scale=args.lr_scale,
    )
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
        logger.info(
            'DataLoader: workers=%d per rank pin_memory=True',
            train_loader.num_workers,
        )
        logger.info(
            '----------- OPTIMIZATION -----------',
        )
        logger.info('LR scale: %.4f (base LR -> %.8f)', args.lr_scale, learning_rate)
        log_optimization_context(
            logger=logger,
            model=raw_model,
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

    resume_value = train_config.get('RESUME')
    if (
        resume_value is not None
        and str(resume_value).lower() != 'none'
    ):
        resume_path = _resolve_project_path(resume_value)
        (
            start_epoch,
            best_val_loss,
            best_val_epe,
        ) = load_training_checkpoint(
            checkpoint_path=resume_path,
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        if is_main:
            logger.info('Resumed training from: %s', resume_path)

    if start_epoch > total_epochs:
        raise ValueError(
            f'Checkpoint epoch {start_epoch} exceeds '
            f'configured EPOCHS {total_epochs}'
        )

    loss_weights = train_config.get('LOSS')
    variant_name = (
        config['LOSS_ABLATION']['NAME']
        if args.job in ('modified_l1', 'optimized_multiscale')
        else None
    )

    scalar_interval = logging_config['SCALAR_INTERVAL']
    image_interval = logging_config['IMAGE_INTERVAL']
    num_batches_per_rank = len(train_loader)
    global_step = start_epoch * num_batches_per_rank

    # Optional gradient clipping: TRAIN.GRAD_CLIP (absent or 0 disables it,
    # preserving the previous unclipped behavior).
    grad_clip = train_config.get('GRAD_CLIP', 0.0)
    if (
        not isinstance(grad_clip, (int, float))
        or isinstance(grad_clip, bool)
        or grad_clip < 0
    ):
        raise ValueError(
            f'TRAIN.GRAD_CLIP must be a non-negative number, got {grad_clip}'
        )
    if is_main and grad_clip > 0:
        logger.info('Gradient clipping enabled: max_norm=%.4f', grad_clip)

    # Optional disparity band weighting for the three disparity losses
    # (TRAIN.LOSS.DISP_BAND_WEIGHT); absent => unweighted, old behavior.
    disp_band_weight = (
        loss_weights.get('DISP_BAND_WEIGHT')
        if isinstance(loss_weights, dict)
        else None
    )
    if is_main and disp_band_weight is not None:
        logger.info(
            'Disparity band weighting enabled: gt>=%s weighted x%s',
            disp_band_weight.get('THRESHOLD'),
            disp_band_weight.get('WEIGHT'),
        )

    # Optional additive near-field term on disp_up
    # (TRAIN.LOSS.DISP_NEAR_ADD); purely additive, far-field untouched.
    disp_near_add = (
        loss_weights.get('DISP_NEAR_ADD')
        if isinstance(loss_weights, dict)
        else None
    )
    if is_main and disp_near_add is not None:
        logger.info(
            'Additive near-field loss enabled: gt>=%s added x%s',
            disp_near_add.get('THRESHOLD'),
            disp_near_add.get('WEIGHT'),
        )

    try:
        for epoch_index in range(start_epoch, total_epochs):
            epoch_number = epoch_index + 1
            if is_main:
                logger.info(
                    'Starting epoch [%d/%d]',
                    epoch_number,
                    total_epochs,
                )
            train_sampler.set_epoch(epoch_index)

            common_kwargs = {
                'model': model,
                'train_loader': train_loader,
                'optimizer': optimizer,
                'scaler': scaler,
                'device': device,
                'max_disp': max_disp,
                'use_amp': use_amp,
                'writer': writer,
                'global_step': global_step,
                'scalar_interval': scalar_interval,
                'image_interval': image_interval,
                'logger': logger,
                'epoch': epoch_number,
                'total_epochs': total_epochs,
                'text_interval': text_interval,
                'grad_clip': grad_clip,
                'scheduler': scheduler,
            }
            if args.job == 'baseline':
                train_stats = train_epoch_baseline_ddp(**common_kwargs)
            else:
                train_stats = train_epoch_hfe_ddp(
                    loss_weights=loss_weights,
                    disp_band_weight=disp_band_weight,
                    disp_near_add=disp_near_add,
                    **common_kwargs,
                )
            global_step += num_batches_per_rank

            is_best = False
            if is_main:
                if args.job == 'baseline':
                    valid_stats = baseline_validate_epoch(
                        model=raw_model,
                        valid_loader=valid_loader,
                        device=device,
                        max_disp=max_disp,
                        use_amp=use_amp,
                        writer=writer,
                        global_step=global_step,
                        image_interval=image_interval,
                        logger=logger,
                        epoch=epoch_number,
                        total_epochs=total_epochs,
                        text_interval=text_interval,
                        progress_label='Validation',
                        valid_image_count=logging_config.get(
                            'VALID_IMAGE_COUNT',
                            4,
                        ),
                        valid_image_interval=logging_config.get(
                            'VALID_IMAGE_INTERVAL',
                            1,
                        ),
                        valid_error_max=logging_config.get(
                            'VALID_ERROR_MAX',
                            5.0,
                        ),
                    )
                else:
                    valid_stats = hfe_validate_epoch(
                        model=raw_model,
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
                        disp_band_weight=disp_band_weight,
                        disp_near_add=disp_near_add,
                        valid_image_count=logging_config.get(
                            'VALID_IMAGE_COUNT',
                            4,
                        ),
                        valid_image_interval=logging_config.get(
                            'VALID_IMAGE_INTERVAL',
                            1,
                        ),
                        valid_error_max=logging_config.get(
                            'VALID_ERROR_MAX',
                            5.0,
                        ),
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

            if (
                scheduler is not None
                and not getattr(scheduler, 'step_per_batch', False)
            ):
                scheduler.step()

            if is_main:
                save_ddp_checkpoint(
                    checkpoint_path=output_dir / 'latest.pth',
                    epoch=epoch_number,
                    raw_model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_val_loss=best_val_loss,
                    best_val_epe=best_val_epe,
                    job=args.job,
                    loss_variant=variant_name,
                    loss_weights=loss_weights,
                )
                if is_best:
                    save_ddp_checkpoint(
                        checkpoint_path=output_dir / 'best.pth',
                        epoch=epoch_number,
                        raw_model=raw_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        best_val_loss=best_val_loss,
                        best_val_epe=best_val_epe,
                        job=args.job,
                        loss_variant=variant_name,
                        loss_weights=loss_weights,
                    )
                writer.flush()
                logger.info(
                    'Checkpoints saved: latest=%s best_updated=%s',
                    output_dir / 'latest.pth',
                    is_best,
                )
                if args.job == 'baseline':
                    logger.info(
                        'Epoch [%d/%d] lr=%.8f train_loss=%.6f '
                        'valid_loss=%.6f valid_epe=%.6f '
                        'valid_d1=%.6f best_epe=%.6f',
                        epoch_number,
                        total_epochs,
                        optimizer.param_groups[0]['lr'],
                        train_stats['loss'],
                        valid_stats['loss'],
                        valid_stats['epe'],
                        valid_stats['d1'],
                        best_val_epe,
                    )
                else:
                    logger.info(
                        'Epoch [%d/%d] lr=%.8f train_loss=%.6f '
                        'valid_loss=%.6f valid_epe=%.6f valid_d1=%.6f '
                        'c0_epe=%.6f c2_epe=%.6f '
                        'weighted=(%.6f,%.6f,%.6f,%.6f,%.6f) '
                        'best_epe=%.6f',
                        epoch_number,
                        total_epochs,
                        optimizer.param_groups[0]['lr'],
                        train_stats['loss'],
                        valid_stats['loss'],
                        valid_stats['epe'],
                        valid_stats['d1'],
                        valid_stats['low_c0_epe'],
                        valid_stats['low_c2_epe'],
                        valid_stats['weighted_loss_cvc_c0'],
                        valid_stats['weighted_loss_cvc_c2'],
                        valid_stats['weighted_loss_d0'],
                        valid_stats['weighted_loss_d2'],
                        valid_stats['weighted_loss_disp'],
                        best_val_epe,
                    )

            dist.barrier()
    finally:
        if writer is not None:
            writer.close()
        dist.destroy_process_group()

    if is_main:
        logger.info('Training completed through epoch %d', total_epochs)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-GPU DDP training for LAS2-S Baseline / Modified-L1',
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to the job YAML configuration',
    )
    parser.add_argument(
        '--job',
        required=True,
        choices=('baseline', 'modified_l1', 'optimized_multiscale'),
        help='Which trainer variant this DDP run reproduces',
    )
    parser.add_argument(
        '--output-subdir',
        default=None,
        help=(
            'Replace the final component of TRAIN.OUTPUT_DIR so the DDP '
            'run does not overwrite the single-card output'
        ),
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override TRAIN.EPOCHS (default: use the YAML value)',
    )
    parser.add_argument(
        '--lr-scale',
        type=float,
        default=1.0,
        help=(
            'Multiply TRAIN.LR by this factor (use e.g. %g or the '
            'world size to follow the linear scaling rule for large '
            'effective batches)' % 8
        ),
    )
    args = parser.parse_args()
    run_training_ddp(args)


if __name__ == '__main__':
    main()
