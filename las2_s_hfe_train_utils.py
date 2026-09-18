import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from core.liteanystereov2_hfe import LiteAnyStereoSHFE
from las2_training_logging import BatchProgressLogger

from core.utils import frame_utils
from core.losses_cvc import (
    cvc_loss,
    final_disparity_loss,
    intermediate_disparity_loss,
)

def load_config(config_path):
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

    if not isinstance(config['DATA_ROOT'], str):
        raise ValueError(
            'DATA_ROOT must be a string'
        )

    if not config['DATA_ROOT']:
        raise ValueError(
            'DATA_ROOT must not be empty'
        )

    data_config = config['DATA_CONFIG']

    if not isinstance(data_config, dict):
        raise TypeError(
            'DATA_CONFIG must be a dictionary'
        )

    data_infos = data_config.get('DATA_INFOS')

    if not isinstance(data_infos, list) or not data_infos:
        raise ValueError(
            'DATA_CONFIG.DATA_INFOS must contain at least one dataset'
        )

    data_info = data_infos[0]

    if not isinstance(data_info, dict):
        raise TypeError(
            'Each DATA_INFOS item must be a dictionary'
        )
    if data_info.get('DATASET') != "CustomDataset":
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
            key
            for key in data_split
            if key.startswith(split_prefix)
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

    augmentation_config = config['AUGMENTATION']

    if not isinstance(augmentation_config, dict):
        raise TypeError(
            'AUGMENTATION must be a dictionary'
        )

    logging_config = config['LOGGING']

    if not isinstance(logging_config, dict):
        raise TypeError(
            'LOGGING must be a dictionary'
        )

    for interval_key in (
        'SCALAR_INTERVAL',
        'IMAGE_INTERVAL',
    ):
        interval = logging_config.get(interval_key)
        if not isinstance(interval, int) or interval < 0:
            raise ValueError(
                f'{interval_key} must be a non-negative integer, '
                f'got {interval}'
            )
    for crop_key in (
        "TRAIN_CROP_SIZE",
        "EVAL_CROP_SIZE",
    ):
        crop_size = augmentation_config.get(crop_key)

        if not isinstance(crop_size, (list, tuple)):
            raise TypeError(
                f"{crop_key} must be a list or tuple"
            )

        if len(crop_size) != 2:
            raise ValueError(
                f"{crop_key} must contain [height, width], "
                f"got {crop_size}"
            )

        if crop_size[0] <= 0 or crop_size[1] <= 0:
            raise ValueError(
                f"{crop_key} must contain positive values, "
                f"got {crop_size}"
            )

    model_config = config["MODEL"]

    if not isinstance(model_config, dict):
        raise TypeError(
            "MODEL must be a dictionary"
        )

    if model_config.get("VERSION") != "las2":
        raise ValueError(
            f"This training entry only supports VERSION=las2, "
            f"got {model_config.get('VERSION')}"
        )

    if model_config.get("MODEL_SIZE") != "s":
        raise ValueError(
            f"This training entry only supports MODEL_SIZE=s, "
            f"got {model_config.get('MODEL_SIZE')}"
        )

    max_disp = model_config.get("MAX_DISP")

    if not isinstance(max_disp, int) or max_disp <= 0:
        raise ValueError(
            f"MAX_DISP must be a positive integer, got {max_disp}"
        )

    if max_disp % 4 != 0:
        raise ValueError(
            f"MAX_DISP must be divisible by 4, got {max_disp}"
        )

    cost_stabilization = model_config.get(
        "COST_STABILIZATION",
    )

    if not isinstance(cost_stabilization, dict):
        raise TypeError(
            "MODEL.COST_STABILIZATION must be a dictionary"
        )

    if cost_stabilization.get("ENABLED") is not True:
        raise ValueError(
            "MODEL.COST_STABILIZATION.ENABLED must be true"
        )

    hidden_channels = cost_stabilization.get("HIDDEN_CHANNELS")
    if hidden_channels is not None and (
        not isinstance(hidden_channels, int)
        or isinstance(hidden_channels, bool)
        or hidden_channels <= 0
    ):
        raise ValueError(
            "MODEL.COST_STABILIZATION.HIDDEN_CHANNELS must be "
            "a positive integer or null"
        )

    negative_slope = cost_stabilization.get(
        "LEAKY_RELU_SLOPE",
        0.1,
    )
    if (
        not isinstance(negative_slope, (int, float))
        or isinstance(negative_slope, bool)
        or not np.isfinite(negative_slope)
        or negative_slope < 0
    ):
        raise ValueError(
            "MODEL.COST_STABILIZATION.LEAKY_RELU_SLOPE must be "
            "a finite non-negative number"
        )

    residual_scale_init = cost_stabilization.get(
        "RESIDUAL_SCALE_INIT",
        0.0,
    )
    if (
        not isinstance(residual_scale_init, (int, float))
        or isinstance(residual_scale_init, bool)
        or not np.isfinite(residual_scale_init)
    ):
        raise ValueError(
            "MODEL.COST_STABILIZATION.RESIDUAL_SCALE_INIT must be "
            "a finite number"
        )

    train_config = config["TRAIN"]

    if not isinstance(train_config, dict):
        raise TypeError(
            "TRAIN must be a dictionary"
        )

    for path_key in (
        'OUTPUT_DIR',
        'LOG_DIR',
    ):
        path_value = train_config.get(path_key)
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(
                f'{path_key} must be a non-empty string, '
                f'got {path_value}'
            )

    resume_value = train_config.get('RESUME')
    if resume_value is not None and not isinstance(resume_value, str):
        raise TypeError(
            f'RESUME must be a string or null, got '
            f'{type(resume_value).__name__}'
        )

    epochs = train_config.get("EPOCHS")
    batch_size = train_config.get("BATCH_SIZE")
    num_workers = train_config.get("NUM_WORKERS")
    learning_rate = train_config.get("LR")
    weight_decay = train_config.get("WEIGHT_DECAY")
    amp = train_config.get("AMP")

    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError(
            f"EPOCHS must be a positive integer, got {epochs}"
        )

    stage1_epochs = train_config.get('STAGE1_EPOCHS')
    stage2_epochs = train_config.get('STAGE2_EPOCHS')
    if not isinstance(stage1_epochs, int) or stage1_epochs <= 0:
        raise ValueError(
            'STAGE1_EPOCHS must be a positive integer'
        )
    if not isinstance(stage2_epochs, int) or stage2_epochs <= 0:
        raise ValueError(
            'STAGE2_EPOCHS must be a positive integer'
        )
    if stage1_epochs + stage2_epochs != epochs:
        raise ValueError(
            'STAGE1_EPOCHS + STAGE2_EPOCHS must equal EPOCHS, '
            f'got {stage1_epochs} + {stage2_epochs} != {epochs}'
        )

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(
            f"BATCH_SIZE must be a positive integer, got {batch_size}"
        )

    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError(
            f"NUM_WORKERS must be a non-negative integer, "
            f"got {num_workers}"
        )

    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError(
            f"LR must be positive, got {learning_rate}"
        )

    stage2_learning_rate = train_config.get(
        'LR_STAGE2',
        learning_rate,
    )
    if (
        not isinstance(stage2_learning_rate, (int, float))
        or stage2_learning_rate <= 0
    ):
        raise ValueError(
            f"LR_STAGE2 must be positive, got {stage2_learning_rate}"
        )

    for learning_rate_key in (
        'LR_STAGE2_REFINE',
        'LR_STAGE2_CVS',
        'LR_STAGE2_HFE',
        'LR_STAGE2_BACKBONE',
    ):
        group_learning_rate = train_config.get(
            learning_rate_key,
            stage2_learning_rate,
        )
        if (
            not isinstance(group_learning_rate, (int, float))
            or isinstance(group_learning_rate, bool)
            or not np.isfinite(group_learning_rate)
            or group_learning_rate <= 0
        ):
            raise ValueError(
                f'{learning_rate_key} must be a finite positive number, '
                f'got {group_learning_rate}'
            )

    c2_regression_ratio = train_config.get(
        'STAGE2_C2_MAX_REGRESSION_RATIO',
        1.03,
    )
    if (
        not isinstance(c2_regression_ratio, (int, float))
        or isinstance(c2_regression_ratio, bool)
        or not np.isfinite(c2_regression_ratio)
        or c2_regression_ratio < 1.0
    ):
        raise ValueError(
            'STAGE2_C2_MAX_REGRESSION_RATIO must be a finite number >= 1'
        )

    if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
        raise ValueError(
            f"WEIGHT_DECAY must be non-negative, got {weight_decay}"
        )

    scheduler_config = train_config.get('SCHEDULER', {})
    if not isinstance(scheduler_config, dict):
        raise TypeError(
            'SCHEDULER must be a dictionary'
        )

    scheduler_name = scheduler_config.get('NAME', 'cosine')
    if scheduler_name not in ('cosine', 'none'):
        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}"
        )

    min_lr = scheduler_config.get('MIN_LR', 0.0)
    if not isinstance(min_lr, (int, float)) or min_lr < 0:
        raise ValueError(
            f'MIN_LR must be non-negative, got {min_lr}'
        )
    if min_lr > learning_rate or min_lr > stage2_learning_rate:
        raise ValueError(
            'MIN_LR must not exceed either stage learning rate, '
            f'got MIN_LR={min_lr}, LR={learning_rate}, '
            f'LR_STAGE2={stage2_learning_rate}'
        )

    if not isinstance(amp, bool):
        raise TypeError(
            f"AMP must be boolean, got {type(amp).__name__}"
        )

    loss_config = train_config.get('LOSS')
    if not isinstance(loss_config, dict):
        raise TypeError(
            'TRAIN.LOSS must be a dictionary'
        )

    loss_keys = (
        'LAMBDA_CVC_C0',
        'LAMBDA_CVC_C2',
        'LAMBDA_D0',
        'LAMBDA_D2',
        'LAMBDA_DISP',
    )

    for loss_key in loss_keys:
        loss_weight = loss_config.get(loss_key)
        if (
            not isinstance(loss_weight, (int, float))
            or isinstance(loss_weight, bool)
            or not np.isfinite(loss_weight)
            or loss_weight < 0
        ):
            raise ValueError(
                f'{loss_key} must be a finite non-negative number, '
                f'got {loss_weight}'
            )

    if sum(loss_config[key] for key in loss_keys) <= 0:
        raise ValueError(
            'At least one TRAIN.LOSS weight must be positive'
        )

    stage1_loss_config = train_config.get(
        'LOSS_STAGE1',
        loss_config,
    )
    if not isinstance(stage1_loss_config, dict):
        raise TypeError(
            'TRAIN.LOSS_STAGE1 must be a dictionary'
        )
    for loss_key in loss_keys:
        loss_weight = stage1_loss_config.get(loss_key)
        if (
            not isinstance(loss_weight, (int, float))
            or isinstance(loss_weight, bool)
            or not np.isfinite(loss_weight)
            or loss_weight < 0
        ):
            raise ValueError(
                f'LOSS_STAGE1.{loss_key} must be a finite '
                f'non-negative number, got {loss_weight}'
            )
    if sum(stage1_loss_config[key] for key in loss_keys) <= 0:
        raise ValueError(
            'At least one TRAIN.LOSS_STAGE1 weight must be positive'
        )

    return config


def _split_sort_key(split_key):
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r'(\d+)', split_key)
    ]


def collect_data_split_files(data_split, split_prefix):
    split_files = []
    for split_key in sorted(data_split, key=_split_sort_key):
        if not split_key.startswith(split_prefix):
            continue
        split_path = data_split[split_key]
        if not isinstance(split_path, str) or not split_path:
            raise ValueError(
                f'DATA_SPLIT.{split_key} must be a non-empty string'
            )
        split_files.append(split_path)

    if not split_files:
        raise KeyError(
            f'DATA_SPLIT must contain at least one {split_prefix} key'
        )
    return split_files


class CustomDataset(Dataset):
    def __init__(self,
                 data_root,
                 list_file,
                 crop_size,
                 training=False,
                 max_disp=192,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.crop_size = tuple(crop_size)
        self.training = training
        self.max_disp = max_disp

        if isinstance(list_file, (list, tuple)):
            list_files = list(list_file)
        else:
            list_files = [list_file]
        if not list_files:
            raise ValueError('list_file must contain at least one dataset list')

        self.list_files = []
        for list_path_value in list_files:
            if not isinstance(list_path_value, (str, Path)):
                raise TypeError(
                    'Each dataset list path must be a string or Path, '
                    f'got {type(list_path_value).__name__}'
                )
            list_path = Path(list_path_value)
            if not list_path.is_absolute():
                list_path = self.data_root / list_path
            self.list_files.append(list_path)
        self.list_file = self.list_files[0]

        if len(self.crop_size) != 2:
            raise ValueError(
                f'crop_size must contain [height, width], got {crop_size}'
            )

        if self.crop_size[0] <= 0 or self.crop_size[1] <= 0:
            raise ValueError(
                f'crop_size must be positive, got {crop_size}'
            )

        self.samples = []
        for list_path in self.list_files:
            if not list_path.exists():
                raise FileNotFoundError(
                    f'Dataset list not found: {list_path}'
                )

            with list_path.open('r') as file:
                for line_number, line in enumerate(file, start=1):
                    fields = line.strip().split()

                    if not fields:
                        continue

                    if len(fields) != 3:
                        raise ValueError(
                            f"Expected 3 fields at "
                            f"{list_path}:{line_number}, "
                            f"got {len(fields)}: {line.strip()}"
                        )

                    left_path, right_path, disp_path = fields

                    self.samples.append(
                        (
                            self._resolve_sample_path(left_path),
                            self._resolve_sample_path(right_path),
                            self._resolve_sample_path(disp_path),
                        )
                    )
        if not self.samples:
            raise ValueError(
                f'No valid samples found in {self.list_files}'
            )

    def _resolve_sample_path(self, path_value):
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self.data_root / path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        left_path, right_path, disp_path = self.samples[index]

        left_image = np.asarray(
            frame_utils.read_gen(str(left_path))
        ).astype(np.uint8)

        right_image = np.asarray(
            frame_utils.read_gen(str(right_path))
        ).astype(np.uint8)
        disp = np.asarray(
            frame_utils.read_gen(str(disp_path))
        ).astype(np.float32)

        if left_image.ndim == 2:
            left_image = np.repeat(
                left_image[..., None],
                3,
                axis=2,
            )
        else:
            left_image = left_image[..., :3]

        if right_image.ndim == 2:
            right_image = np.repeat(
                right_image[..., None],
                3,
                axis=2,
            )
        else:
            right_image = right_image[..., :3]

        if disp.ndim != 2:
            raise ValueError(
                f"Disparity must have shape [H, W], got {disp.shape} "
                f"for {disp_path}"
            )

        if left_image.shape[:2] != disp.shape:
            raise ValueError(
                f"Left image and disparity shapes do not match: "
                f"{left_image.shape[:2]} vs {disp.shape} "
                f"for {disp_path}"
            )

        if right_image.shape[:2] != disp.shape:
            raise ValueError(
                f"Right image and disparity shapes do not match: "
                f"{right_image.shape[:2]} vs {disp.shape} "
                f"for {disp_path}"
            )

        valid = (
            np.isfinite(disp)
            & (disp >= 0)
            & (disp < self.max_disp)
        )

        height, width = disp.shape
        crop_height, crop_width = self.crop_size

        if crop_height > height or crop_width > width:
            raise ValueError(
                f"Crop size {self.crop_size} is larger than sample size "
                f"{(height, width)} for {left_path}"
            )

        if self.training:
            top = np.random.randint(
                0,
                height - crop_height + 1,
            )
            left = np.random.randint(
                0,
                width - crop_width + 1,
            )
        else:
            top = (height - crop_height) // 2
            left = (width - crop_width) // 2

        bottom = top + crop_height
        right = left + crop_width

        left_image = left_image[top:bottom, left:right]
        right_image = right_image[top:bottom, left:right]

        disp = disp[top:bottom, left:right]
        valid = valid[top:bottom, left:right]

        left_image = torch.from_numpy(
            np.ascontiguousarray(left_image)
        ).permute(2, 0, 1).float()

        right_image = torch.from_numpy(
            np.ascontiguousarray(right_image)
        ).permute(2, 0, 1).float()

        disp = torch.from_numpy(
            np.ascontiguousarray(disp)
        ).unsqueeze(0).float()

        valid = torch.from_numpy(
            np.ascontiguousarray(valid)
        ).unsqueeze(0).float()

        file_info = [
            str(left_path),
            str(right_path),
            str(disp_path),
        ]

        return (
            file_info,
            left_image,
            right_image,
            disp,
            valid,
        )

def collate_stereo_batch(batch):
    file_infos, left_images, right_images, disps, valids = zip(*batch)

    file_info_batch = list(file_infos)

    left_batch = torch.stack(
        left_images,
        dim=0,
    )

    right_batch = torch.stack(
        right_images,
        dim=0,
    )

    gt_disp_batch = torch.stack(
        disps,
        dim=0,
    )

    valid_batch = torch.stack(
        valids,
        dim=0,
    )

    return (
        file_info_batch,
        left_batch,
        right_batch,
        gt_disp_batch,
        valid_batch,
    )

def _seed_data_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_hfe_model(config, device, logger=None):
    model_config = config['MODEL']
    hfe_config = model_config.get('HFE', {})

    model = LiteAnyStereoSHFE(
        fnet_pretrained=False,
        cutoff_ratio=hfe_config.get('CUTOFF_RATIO', 0.1),
        max_disp=model_config['MAX_DISP'],
        cost_stabilization=model_config['COST_STABILIZATION'],
    )

    checkpoint_value = model_config.get('PRETRAINED')

    if (
        checkpoint_value is not None
        and str(checkpoint_value).lower() != 'none'
    ):
        checkpoint_path = Path(checkpoint_value)

        if not checkpoint_path.is_absolute():
            checkpoint_path = (
                Path(__file__).resolve().parent
                / checkpoint_path
            )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f'Checkpoint not found: {checkpoint_path}'
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=True,
        )

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif(
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

        model_state = model.state_dict()
        compatible_state_dict = {
            key: value
            for key, value in state_dict.items()
            if (
                key in model_state
                and torch.is_tensor(value)
                and value.shape == model_state[key].shape
            )
        }

        if not compatible_state_dict:
            raise RuntimeError(
                'No compatible checkpoint parameters matched the HFE model'
            )

        incompatible_keys = model.load_state_dict(
            compatible_state_dict,
            strict=False,
        )

        missing_keys = incompatible_keys.missing_keys
        skipped_keys = len(state_dict) - len(compatible_state_dict)

        log = logger.info if logger is not None else print
        log(
            f'Loaded checkpoint: {checkpoint_path}'
        )
        log(
            f'Matched keys: {len(compatible_state_dict)}'
        )
        log(
            f'Skipped keys: {skipped_keys}'
        )
        log(
            f'Missing keys: {len(missing_keys)}'
        )

    model = model.to(device)
    return model

def build_optimizer_and_scaler(
        config,
        model,
        device,
        learning_rate=None,
        parameter_groups=None,
):
    device = torch.device(device)

    train_config = config["TRAIN"]

    if learning_rate is None:
        learning_rate = train_config["LR"]
    weight_decay = train_config["WEIGHT_DECAY"]
    amp_enabled = train_config["AMP"]

    if parameter_groups is None:
        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError(
                'No trainable parameters are available for the optimizer'
            )
        optimizer_parameters = [
            {
                'params': trainable_parameters,
                'lr': learning_rate,
                'name': 'all',
            },
        ]
    else:
        named_parameters = dict(model.named_parameters())
        used_names = set()
        optimizer_parameters = []

        for group_name, group_config in parameter_groups.items():
            prefixes = group_config['prefixes']
            group_learning_rate = group_config['learning_rate']
            parameters = []
            for name, parameter in named_parameters.items():
                if (
                    parameter.requires_grad
                    and name not in used_names
                    and any(name.startswith(prefix) for prefix in prefixes)
                ):
                    parameters.append(parameter)
                    used_names.add(name)

            if parameters:
                optimizer_parameters.append(
                    {
                        'params': parameters,
                        'lr': group_learning_rate,
                        'name': group_name,
                    }
                )

        remaining_parameters = [
            parameter
            for name, parameter in named_parameters.items()
            if parameter.requires_grad and name not in used_names
        ]
        if remaining_parameters:
            optimizer_parameters.append(
                {
                    'params': remaining_parameters,
                    'lr': learning_rate,
                    'name': 'remaining',
                }
            )

        if not optimizer_parameters:
            raise ValueError(
                'No trainable parameters are available for the optimizer'
            )

    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        weight_decay=weight_decay,
    )

    use_amp = (
        amp_enabled
        and device.type == "cuda"
    )

    # torch.amp.GradScaler only exists from torch 2.3; fall back to the
    # classic API so the same script runs on older torch builds.
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_amp,
        )
    else:
        scaler = torch.cuda.amp.GradScaler(
            enabled=use_amp,
        )

    return optimizer, scaler, use_amp


def build_scheduler(config, optimizer, epochs=None):
    train_config = config['TRAIN']
    scheduler_config = train_config.get(
        'SCHEDULER',
        {},
    )
    scheduler_name = scheduler_config.get(
        'NAME',
        'cosine',
    )

    if scheduler_name == 'none':
        return None

    if epochs is None:
        epochs = train_config['EPOCHS']

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=scheduler_config.get('MIN_LR', 0.0),
    )

def _log_stereo_images(
        writer,
        prefix,
        left,
        right,
        gt_disp,
        pred_disp,
        max_disp,
        step,
):
    if writer is None:
        return

    left_image = (
        left[:1].detach().float().cpu() / 255.0
    ).clamp(0.0, 1.0)
    right_image = (
        right[:1].detach().float().cpu() / 255.0
    ).clamp(0.0, 1.0)
    gt_image = (
        gt_disp[:1].detach().float().cpu() / max_disp
    ).clamp(0.0, 1.0)
    pred_image = (
        pred_disp[:1].detach().float().cpu() / max_disp
    ).clamp(0.0, 1.0)

    writer.add_images(
        f'{prefix}/left',
        left_image,
        step,
    )
    writer.add_images(
        f'{prefix}/right',
        right_image,
        step,
    )
    writer.add_images(
        f'{prefix}/gt_disp',
        gt_image,
        step,
    )
    writer.add_images(
        f'{prefix}/pred_disp',
        pred_image,
        step,
    )

def compute_stereo_metric_sums(
        pred_disp,
        gt_disp,
        valid,
        max_disp,
):
    if pred_disp.shape != gt_disp.shape:
        raise ValueError(
            f'Prediction and GT shapes must match, got '
            f'{pred_disp.shape} and {gt_disp.shape}'
        )

    valid_mask = (
        torch.isfinite(pred_disp)
        & torch.isfinite(gt_disp)
        & (gt_disp >= 0)
        & (gt_disp < max_disp)
    )

    if valid is not None:
        if valid.shape != gt_disp.shape:
            raise ValueError(
                f'Valid mask and GT shapes must match, got '
                f'{valid.shape} and {gt_disp.shape}'
            )
        valid_mask = valid_mask & (valid > 0)

    if not valid_mask.any():
        return {
            'epe_sum': 0.0,
            'sq_error_sum': 0.0,
            'abs_rel_sum': 0.0,
            'sq_rel_sum': 0.0,
            'bad3_count': 0.0,
            'd1_count': 0.0,
            'valid_count': 0.0,
            'relative_valid_count': 0.0,
        }

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
        'epe_sum': absolute_error.sum().item(),
        'sq_error_sum': absolute_error.square().sum().item(),
        'abs_rel_sum': abs_rel_sum,
        'sq_rel_sum': sq_rel_sum,
        'bad3_count': (absolute_error > 3.0).sum().item(),
        'd1_count': (
            (absolute_error > 3.0)
            & (relative_error > 0.05)
        ).sum().item(),
        'valid_count': float(absolute_error.numel()),
        'relative_valid_count': float(relative_target.numel()),
    }


def compute_multiscale_metric_sums(
        pred_disp,
        gt_disp,
        valid,
        max_disp,
):
    prediction = F.interpolate(
        pred_disp,
        size=gt_disp.shape[-2:],
        mode='bilinear',
        align_corners=False,
    )
    return compute_stereo_metric_sums(
        pred_disp=prediction,
        gt_disp=gt_disp,
        valid=valid,
        max_disp=max_disp,
    )


def validate_epoch(
        model,
        valid_loader,
        device,
        max_disp,
        use_amp,
        lambda_cvc_c0,
        lambda_cvc_c2,
        lambda_d0,
        lambda_d2,
        lambda_disp,
        writer=None,
        global_step=0,
        image_interval=0,
        stage='joint',
        logger=None,
        epoch=1,
        total_epochs=1,
        text_interval=0,
        progress_label='Validation',
        disp_band_weight=None,
        disp_near_add=None,
):
    device = torch.device(device)

    if stage not in ('stage1', 'stage2', 'joint'):
        raise ValueError(
            f'Unsupported training stage: {stage}'
        )

    model.eval()

    num_batches = len(valid_loader)
    if num_batches == 0:
        raise ValueError('valid_loader must contain at least one batch')

    running_stats = {
        'loss': 0.0,
        'loss_cvc_c0': 0.0,
        'loss_cvc_c2': 0.0,
        'loss_d0': 0.0,
        'loss_d2': 0.0,
        'loss_disp': 0.0,
        'weighted_loss_cvc_c0': 0.0,
        'weighted_loss_cvc_c2': 0.0,
        'weighted_loss_d0': 0.0,
        'weighted_loss_d2': 0.0,
        'weighted_loss_disp': 0.0,
    }
    metric_sums = {
        'epe_sum': 0.0,
        'sq_error_sum': 0.0,
        'abs_rel_sum': 0.0,
        'sq_rel_sum': 0.0,
        'bad3_count': 0.0,
        'd1_count': 0.0,
        'valid_count': 0.0,
        'relative_valid_count': 0.0,
        'low_c0_epe_sum': 0.0,
        'low_c0_bad3_count': 0.0,
        'low_c0_d1_count': 0.0,
        'low_c0_valid_count': 0.0,
        'low_c2_epe_sum': 0.0,
        'low_c2_bad3_count': 0.0,
        'low_c2_d1_count': 0.0,
        'low_c2_valid_count': 0.0,
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

            left_batch = left_batch.to(
                device,
                non_blocking=True,
            )

            right_batch = right_batch.to(
                device,
                non_blocking=True,
            )

            gt_disp_batch = gt_disp_batch.to(
                device,
                non_blocking=True,
            )

            valid_batch = valid_batch.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                _, loss_stats, outputs = compute_las2_s_hfe_loss(
                    model=model,
                    left=left_batch,
                    right=right_batch,
                    gt_disp=gt_disp_batch,
                    valid=valid_batch,
                    max_disp=max_disp,
                    lambda_cvc_c0=lambda_cvc_c0,
                    lambda_cvc_c2=lambda_cvc_c2,
                    lambda_d0=lambda_d0,
                    lambda_d2=lambda_d2,
                    lambda_disp=lambda_disp,
                    stage=stage,
                    disp_band_weight=disp_band_weight,
                    disp_near_add=disp_near_add,
                )

            loss_values = {
                key: value.item()
                for key, value in loss_stats.items()
            }
            for key in running_stats:
                running_stats[key] += loss_values[key]

            batch_metrics = compute_stereo_metric_sums(
                pred_disp=outputs['disp_up'],
                gt_disp=gt_disp_batch,
                valid=valid_batch,
                max_disp=max_disp,
            )
            batch_low_c0_metrics = compute_multiscale_metric_sums(
                pred_disp=outputs['disp_low_c0'],
                gt_disp=gt_disp_batch,
                valid=valid_batch,
                max_disp=max_disp,
            )
            batch_low_c2_metrics = compute_multiscale_metric_sums(
                pred_disp=outputs['disp_low_c2'],
                gt_disp=gt_disp_batch,
                valid=valid_batch,
                max_disp=max_disp,
            )
            metric_sums['epe_sum'] += batch_metrics['epe_sum']
            metric_sums['sq_error_sum'] += batch_metrics['sq_error_sum']
            metric_sums['abs_rel_sum'] += batch_metrics['abs_rel_sum']
            metric_sums['sq_rel_sum'] += batch_metrics['sq_rel_sum']
            metric_sums['bad3_count'] += batch_metrics['bad3_count']
            metric_sums['d1_count'] += batch_metrics['d1_count']
            metric_sums['valid_count'] += batch_metrics['valid_count']
            metric_sums['relative_valid_count'] += batch_metrics['relative_valid_count']
            metric_sums['low_c0_epe_sum'] += batch_low_c0_metrics['epe_sum']
            metric_sums['low_c0_bad3_count'] += batch_low_c0_metrics['bad3_count']
            metric_sums['low_c0_d1_count'] += batch_low_c0_metrics['d1_count']
            metric_sums['low_c0_valid_count'] += batch_low_c0_metrics['valid_count']
            metric_sums['low_c2_epe_sum'] += batch_low_c2_metrics['epe_sum']
            metric_sums['low_c2_bad3_count'] += batch_low_c2_metrics['bad3_count']
            metric_sums['low_c2_d1_count'] += batch_low_c2_metrics['d1_count']
            metric_sums['low_c2_valid_count'] += batch_low_c2_metrics['valid_count']

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
                    pred_disp=outputs['disp_up'],
                    max_disp=max_disp,
                    step=step,
                )

            valid_count = metric_sums['valid_count']
            low_c2_valid_count = metric_sums['low_c2_valid_count']
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
            running_c2_epe = (
                metric_sums['low_c2_epe_sum'] / low_c2_valid_count
                if low_c2_valid_count > 0
                else 0.0
            )
            progress.finish_batch(
                batch_index=batch_index,
                timing=batch_timing,
                loss=loss_values['loss'],
                average_loss=running_stats['loss'] / batch_index,
                metrics={
                    'EPE': running_epe,
                    'D1': running_d1,
                    'C2EPE': running_c2_epe,
                },
            )

    if metric_sums['valid_count'] == 0:
        raise ValueError(
            'Validation set contains no valid disparity pixels'
        )

    valid_stats = {
        key: value / num_batches
        for key, value in running_stats.items()
    }
    valid_stats.update(
        {
            'epe': (
                metric_sums['epe_sum']
                / metric_sums['valid_count']
            ),
            'rmse': (
                metric_sums['sq_error_sum']
                / metric_sums['valid_count']
            ) ** 0.5,
            'abs_rel': (
                metric_sums['abs_rel_sum']
                / max(metric_sums['relative_valid_count'], 1.0)
            ),
            'sq_rel': (
                metric_sums['sq_rel_sum']
                / max(metric_sums['relative_valid_count'], 1.0)
            ),
            'bad3': (
                metric_sums['bad3_count']
                / metric_sums['valid_count']
            ),
            'd1': (
                metric_sums['d1_count']
                / metric_sums['valid_count']
            ),
            'low_c0_epe': (
                metric_sums['low_c0_epe_sum']
                / metric_sums['low_c0_valid_count']
            ),
            'low_c0_d1': (
                metric_sums['low_c0_d1_count']
                / metric_sums['low_c0_valid_count']
            ),
            'low_c2_epe': (
                metric_sums['low_c2_epe_sum']
                / metric_sums['low_c2_valid_count']
            ),
            'low_c2_d1': (
                metric_sums['low_c2_d1_count']
                / metric_sums['low_c2_valid_count']
            ),
            'valid_pixels': metric_sums['valid_count'],
            'relative_valid_pixels': metric_sums['relative_valid_count'],
        }
    )

    return valid_stats

def load_training_checkpoint(
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
            f'Training checkpoint not found: {checkpoint_path}'
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            'Training checkpoint must be a dictionary'
        )

    architecture_version = checkpoint.get('architecture_version')
    supported_architectures = {
        'las2_s_hfe_litematch_cvs_v1',
        'las2_s_hfe_fair_final_disp_v1',
        'las2_s_baseline_las2_s_v1',
    }
    if (
        architecture_version is not None
        and architecture_version not in supported_architectures
    ):
        raise ValueError(
            f'Unsupported checkpoint architecture: '
            f'{architecture_version}'
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
        raise KeyError(
            "Training checkpoint must contain a 'model' state dictionary"
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

    if 'optimizer' not in checkpoint:
        raise KeyError(
            "Training checkpoint must contain 'optimizer' state"
        )
    optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None:
        scheduler_state = checkpoint.get('scheduler')
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        else:
            scheduler.step(checkpoint.get('epoch', 0))

    if 'scaler' not in checkpoint:
        raise KeyError(
            "Training checkpoint must contain 'scaler' state"
        )
    scaler.load_state_dict(checkpoint['scaler'])

    epoch = checkpoint.get('epoch')
    best_val_loss = checkpoint.get('best_val_loss')
    best_val_epe = checkpoint.get(
        'best_val_epe',
        float('inf'),
    )

    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError(
            f"Invalid checkpoint epoch: {epoch}"
        )
    if not isinstance(best_val_loss, (int, float)):
        raise ValueError(
            f"Invalid checkpoint best_val_loss: {best_val_loss}"
        )
    if not isinstance(best_val_epe, (int, float)):
        raise ValueError(
            f"Invalid checkpoint best_val_epe: {best_val_epe}"
        )

    return (
        epoch,
        float(best_val_loss),
        float(best_val_epe),
    )


def _resolve_project_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path

def compute_las2_s_hfe_loss(
        model,
        left,
        right,
        gt_disp,
        valid=None,
        max_disp=192,
        lambda_cvc_c0=1.0,
        lambda_cvc_c2=1.0,
        lambda_d0=1.0,
        lambda_d2=1.0,
        lambda_disp=1.0,
        stage='joint',
        disp_band_weight=None,
        disp_near_add=None,
):
    if stage not in ('stage1', 'stage2', 'joint'):
        raise ValueError(
            f'Unsupported training stage: {stage}'
        )

    outputs = model(
        left,
        right,
        max_disp=max_disp,
        test_mode=False,
        return_aux=True,
    )

    zero_loss = outputs['disp_up'].sum() * 0.0
    loss_cvc_c0 = zero_loss
    loss_cvc_c2 = zero_loss
    loss_d0 = zero_loss
    loss_d2 = zero_loss
    loss_disp = zero_loss

    if stage in ('stage1', 'stage2', 'joint'):
        loss_cvc_c0 = cvc_loss(
            cost_prob=outputs['cost_prob_c0'],
            gt_disp=gt_disp,
            valid=valid,
            max_disp=max_disp,
        )
        loss_cvc_c2 = cvc_loss(
            cost_prob=outputs['cost_prob_c2'],
            gt_disp=gt_disp,
            valid=valid,
            max_disp=max_disp,
        )
        loss_d0 = intermediate_disparity_loss(
            pred_disp=outputs['disp_low_c0'],
            gt_disp=gt_disp,
            valid=valid,
            max_disp=max_disp,
            band_weight=disp_band_weight,
        )
        loss_d2 = intermediate_disparity_loss(
            pred_disp=outputs['disp_low_c2'],
            gt_disp=gt_disp,
            valid=valid,
            max_disp=max_disp,
            band_weight=disp_band_weight,
        )

    if stage in ('stage2', 'joint'):
        loss_disp = final_disparity_loss(
            pred_disp=outputs['disp_up'],
            gt_disp=gt_disp,
            valid=valid,
            max_disp=max_disp,
            band_weight=disp_band_weight,
            near_add=disp_near_add,
        )

    weighted_loss_cvc_c0 = lambda_cvc_c0 * loss_cvc_c0
    weighted_loss_cvc_c2 = lambda_cvc_c2 * loss_cvc_c2
    weighted_loss_d0 = lambda_d0 * loss_d0
    weighted_loss_d2 = lambda_d2 * loss_d2
    weighted_loss_disp = lambda_disp * loss_disp

    total_loss = (
        weighted_loss_cvc_c0
        + weighted_loss_cvc_c2
        + weighted_loss_d0
        + weighted_loss_d2
        + weighted_loss_disp
    )

    loss_stats = {
        'loss': total_loss.detach(),
        'loss_cvc_c0': loss_cvc_c0.detach(),
        'loss_cvc_c2': loss_cvc_c2.detach(),
        'loss_d0': loss_d0.detach(),
        'loss_d2': loss_d2.detach(),
        'loss_disp': loss_disp.detach(),
        'weighted_loss_cvc_c0': weighted_loss_cvc_c0.detach(),
        'weighted_loss_cvc_c2': weighted_loss_cvc_c2.detach(),
        'weighted_loss_d0': weighted_loss_d0.detach(),
        'weighted_loss_d2': weighted_loss_d2.detach(),
        'weighted_loss_disp': weighted_loss_disp.detach(),
    }

    return total_loss, loss_stats, outputs
