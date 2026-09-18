import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass(frozen=True)
class _BatchTiming:
    should_log: bool
    data_time: float
    step_started_at: Optional[float]


def _format_duration(seconds):
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def _format_config_value(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _iter_config_values(value, prefix):
    if isinstance(value, dict):
        if not value:
            yield prefix, value
            return
        for key, child in value.items():
            yield from _iter_config_values(child, f'{prefix}.{key}')
        return

    if isinstance(value, (list, tuple)):
        if not value or all(
            not isinstance(child, (dict, list, tuple))
            for child in value
        ):
            yield prefix, list(value)
            return
        for index, child in enumerate(value):
            yield from _iter_config_values(child, f'{prefix}[{index}]')
        return

    yield prefix, value


def log_config(logger, config_path, config):
    logger.info('Configuration file: %s', Path(config_path).resolve())
    logger.info('----------- CONFIGURATION -----------')
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info('----------- %s -----------', key)
        for path, leaf_value in _iter_config_values(value, f'cfg.{key}'):
            logger.info('%s: %s', path, _format_config_value(leaf_value))


def resolve_text_interval(logging_config):
    interval = logging_config.get(
        'TEXT_INTERVAL',
        logging_config.get('SCALAR_INTERVAL', 0),
    )
    if (
        not isinstance(interval, int)
        or isinstance(interval, bool)
        or interval < 0
    ):
        raise ValueError(
            'LOGGING.TEXT_INTERVAL must be a non-negative integer, '
            f'got {interval}'
        )
    return interval


def log_data_context(logger, train_loader, valid_loader):
    logger.info('----------- DATA LOADERS -----------')
    logger.info(
        'Training dataset: samples=%d batches=%d batch_size=%s',
        len(train_loader.dataset),
        len(train_loader),
        train_loader.batch_size,
    )
    logger.info(
        'Validation dataset: samples=%d batches=%d batch_size=%s',
        len(valid_loader.dataset),
        len(valid_loader),
        valid_loader.batch_size,
    )
    logger.info(
        'DataLoader: workers=%d pin_memory=%s',
        train_loader.num_workers,
        train_loader.pin_memory,
    )


def log_optimization_context(
        logger,
        model,
        optimizer,
        scheduler,
        use_amp,
        device,
        text_interval,
        label='Optimization',
):
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    learning_rates = sorted({
        float(group['lr'])
        for group in optimizer.param_groups
    })
    weight_decays = sorted({
        float(group.get('weight_decay', 0.0))
        for group in optimizer.param_groups
    })

    logger.info('----------- %s -----------', label.upper())
    logger.info(
        'Parameters: total=%d trainable=%d',
        total_parameters,
        trainable_parameters,
    )
    logger.info(
        'Optimizer: %s learning_rates=%s weight_decays=%s',
        optimizer.__class__.__name__,
        learning_rates,
        weight_decays,
    )
    logger.info(
        'Scheduler: %s',
        scheduler.__class__.__name__ if scheduler is not None else 'none',
    )
    logger.info('AMP enabled: %s', use_amp)
    logger.info('Text progress interval: %d batches', text_interval)
    if device.type == 'cuda':
        logger.info(
            'CUDA device: %s',
            torch.cuda.get_device_name(device),
        )


def _synchronize_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class BatchProgressLogger:
    def __init__(
            self,
            logger,
            phase,
            epoch,
            total_epochs,
            num_batches,
            interval,
            device,
    ):
        if num_batches <= 0:
            raise ValueError('num_batches must be positive')
        if interval < 0:
            raise ValueError('interval must be non-negative')

        self.logger = logger
        self.phase = phase
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.num_batches = num_batches
        self.interval = interval
        self.device = torch.device(device)
        self.phase_started_at = time.perf_counter()
        self.last_batch_finished_at = self.phase_started_at

    def _should_log(self, batch_index):
        return (
            self.logger is not None
            and self.interval > 0
            and (
                batch_index == 1
                or batch_index % self.interval == 0
                or batch_index == self.num_batches
            )
        )

    def start_batch(self, batch_index):
        now = time.perf_counter()
        data_time = now - self.last_batch_finished_at
        should_log = self._should_log(batch_index)
        step_started_at = None
        if should_log:
            _synchronize_device(self.device)
            step_started_at = time.perf_counter()
        return _BatchTiming(
            should_log=should_log,
            data_time=data_time,
            step_started_at=step_started_at,
        )

    def finish_batch(
            self,
            batch_index,
            timing,
            loss,
            average_loss,
            learning_rate=None,
            global_step=None,
            metrics=None,
    ):
        if not timing.should_log:
            self.last_batch_finished_at = time.perf_counter()
            return

        _synchronize_device(self.device)
        finished_at = time.perf_counter()
        step_time = finished_at - timing.step_started_at
        elapsed = finished_at - self.phase_started_at
        average_batch_time = elapsed / batch_index
        eta = average_batch_time * (self.num_batches - batch_index)

        fields = [
            f'{self.phase} Epoch: {self.epoch}/{self.total_epochs}',
            f'Iter: {batch_index:4d}/{self.num_batches}',
        ]
        if global_step is not None:
            fields.append(f'Step:{global_step}')
        fields.extend([
            f'Loss:{float(loss):.6f}({float(average_loss):.6f})',
        ])
        if learning_rate is not None:
            fields.append(f'LR:{float(learning_rate):.4e}')
        fields.extend([
            f'DataTime:{timing.data_time:.2f}s',
            f'StepTime:{step_time:.2f}s',
            f'Elapsed:{_format_duration(elapsed)}',
            f'ETA:{_format_duration(eta)}',
        ])

        if metrics:
            for key, value in metrics.items():
                fields.append(f'{key}:{float(value):.6f}')

        if self.device.type == 'cuda':
            memory_mib = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            fields.append(f'GPU:{memory_mib:.0f}MiB')

        self.logger.info(' '.join(fields))
        self.last_batch_finished_at = finished_at
