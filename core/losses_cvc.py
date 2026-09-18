import torch
import torch.nn.functional as F


def _validate_band_weight(band_weight):
    """Validate and normalize the optional disparity band-weighting config.

    band_weight must be a mapping with:
      THRESHOLD: float >= 0  - pixels with gt disparity >= THRESHOLD are
                 considered "near" and receive the amplified weight
      WEIGHT:    float > 0   - multiplier applied to near-band pixels
    Returns (threshold, weight) or None when band_weight is None.
    """
    if band_weight is None:
        return None
    if not isinstance(band_weight, dict):
        raise TypeError(
            'DISP_BAND_WEIGHT must be a mapping with THRESHOLD and WEIGHT, '
            f'got {type(band_weight).__name__}'
        )
    threshold = band_weight.get('THRESHOLD')
    weight = band_weight.get('WEIGHT')
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or threshold < 0
    ):
        raise ValueError(
            f'DISP_BAND_WEIGHT.THRESHOLD must be a non-negative number, '
            f'got {threshold}'
        )
    if (
        not isinstance(weight, (int, float))
        or isinstance(weight, bool)
        or weight <= 0
    ):
        raise ValueError(
            f'DISP_BAND_WEIGHT.WEIGHT must be a positive number, got {weight}'
        )
    return float(threshold), float(weight)


def _weighted_mean_smooth_l1(
        pred_valid,
        gt_valid,
        beta,
        threshold,
        weight,
):
    """Smooth L1 over selected pixels with near-band weighting.

    Pixels whose gt disparity >= threshold get `weight`, others 1.0.
    The result is sum(w * loss) / sum(w), keeping overall scale stable.
    """
    elementwise = F.smooth_l1_loss(
        pred_valid,
        gt_valid,
        reduction='none',
        beta=beta,
    )
    pixel_weight = torch.where(
        gt_valid >= threshold,
        torch.full_like(gt_valid, weight),
        torch.ones_like(gt_valid),
    )
    return (elementwise * pixel_weight).sum() / pixel_weight.sum()


def _validate_near_add(near_add):
    """Validate the optional additive near-field loss config.

    near_add must be a mapping with:
      THRESHOLD: float >= 0  - pixels with gt disparity >= THRESHOLD get an
                 extra loss term
      WEIGHT:    float > 0   - coefficient of that extra term
    Unlike DISP_BAND_WEIGHT (which re-normalizes the whole loss and thereby
    dilutes far-field gradients), this term is purely additive: the base
    full-image loss is untouched and the near-band mean is added on top.
    Returns (threshold, weight) or None when near_add is None.
    """
    if near_add is None:
        return None
    if not isinstance(near_add, dict):
        raise TypeError(
            'DISP_NEAR_ADD must be a mapping with THRESHOLD and WEIGHT, '
            f'got {type(near_add).__name__}'
        )
    threshold = near_add.get('THRESHOLD')
    weight = near_add.get('WEIGHT')
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or threshold < 0
    ):
        raise ValueError(
            f'DISP_NEAR_ADD.THRESHOLD must be a non-negative number, '
            f'got {threshold}'
        )
    if (
        not isinstance(weight, (int, float))
        or isinstance(weight, bool)
        or weight <= 0
    ):
        raise ValueError(
            f'DISP_NEAR_ADD.WEIGHT must be a positive number, got {weight}'
        )
    return float(threshold), float(weight)


def _masked_average_disparity(
        gt_disp,
        valid,
        target_size,
        max_disp=None,
):
    if gt_disp.ndim != 4 or gt_disp.shape[1] != 1:
        raise ValueError(
            f'gt_disp must have shape [B, 1, H, W], got {gt_disp.shape}'
        )

    gt_disp = gt_disp.float()
    valid_pixels = (
        torch.isfinite(gt_disp)
        & (gt_disp >= 0)
    )

    if max_disp is not None:
        if max_disp <= 0:
            raise ValueError(
                f'max_disp must be positive, got {max_disp}'
            )
        valid_pixels = valid_pixels & (gt_disp < max_disp)

    if valid is not None:
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)

        if valid.ndim != 4 or valid.shape[1] != 1:
            raise ValueError(
                'valid must have shape [B, H, W] or [B, 1, H, W], '
                f'got {valid.shape}'
            )

        if valid.shape != gt_disp.shape:
            raise ValueError(
                'valid and gt_disp must have the same shape, '
                f'got {valid.shape} and {gt_disp.shape}'
            )

        valid_pixels = valid_pixels & (
            valid.to(device=gt_disp.device) > 0.5
        )

    valid_float = valid_pixels.to(dtype=gt_disp.dtype)
    masked_disp = torch.where(
        valid_pixels,
        gt_disp,
        torch.zeros_like(gt_disp),
    )

    valid_ratio = F.adaptive_avg_pool2d(
        valid_float,
        output_size=target_size,
    )
    disp_sum_ratio = F.adaptive_avg_pool2d(
        masked_disp,
        output_size=target_size,
    )

    gt_disp_low = disp_sum_ratio / valid_ratio.clamp_min(
        torch.finfo(gt_disp.dtype).eps
    )
    valid_low = valid_ratio[:, 0] > 0

    return gt_disp_low, valid_low


def cvc_loss(
        cost_prob,
        gt_disp,
        valid=None,
        max_disp=192,
        eps=1e-7,
):
    if cost_prob.ndim != 4:
        raise ValueError(
            f'cost_prob must have shape [B, D, H, W], got {cost_prob.shape}'
        )
    if gt_disp.ndim != 4 or gt_disp.shape[1] != 1:
          raise ValueError(
              f"gt_disp must have shape [B, 1, H, W], got {gt_disp.shape}"
        )

    if cost_prob.shape[0] != gt_disp.shape[0]:
        raise ValueError(
            "cost_prob and gt_disp must have the same batch size"
        )

    if max_disp <= 0 or max_disp % 4 != 0:
        raise ValueError(
            f"max_disp must be a positive multiple of 4, got {max_disp}"
        )

    num_bins = cost_prob.shape[1]    
    expected_bins = max_disp // 4

    if num_bins != expected_bins:
        raise ValueError(
            f'Expected {expected_bins} cost bins for max_disp={max_disp}, '
            f'got {num_bins}'
        )

    target_size = cost_prob.shape[-2:]

    gt_disp_low = F.interpolate(
        gt_disp.float(),
        size=target_size,
        mode='nearest',
    )
    gt_disp_bins = gt_disp_low[:, 0] / 4.0

    valid_low = (
        torch.isfinite(gt_disp_bins)
        & (gt_disp_bins >= 0)
        & (gt_disp_bins < num_bins)
    )

    if valid is not None:
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)

        if valid.ndim != 4 or valid.shape[1] != 1:
            raise ValueError(
                'valid must have shape [B, H, W] or [B, 1, H, W], '
                f'got {valid.shape}'
            )

        if valid.shape != gt_disp.shape:
            raise ValueError(
                'valid and gt_disp must have the same shape, '
                f'got {valid.shape} and {gt_disp.shape}'
            )

        valid_low_from_input = F.interpolate(
            valid.to(
                device=gt_disp.device,
                dtype=torch.float32,
            ),
            size=target_size,
            mode='nearest',
        )[:, 0] > 0.5

        valid_low = valid_low & valid_low_from_input

    target_indices = torch.round(gt_disp_bins).long()

    target_indices = target_indices.clamp(
        min=0,
        max=num_bins - 1,
    )

    target_one_hot = F.one_hot(
        target_indices,
        num_classes=num_bins,
    ).permute(
        0,
        3,
        1,
        2,
    ).to(
        dtype=cost_prob.dtype,
    )

    cost_prob = cost_prob.clamp(
        min=eps,
        max=1.0 - eps,
    )

    with torch.autocast(
        device_type=cost_prob.device.type,
        enabled=False,
    ):
        per_bin_loss = F.binary_cross_entropy(
            cost_prob.float(),
            target_one_hot.float(),
            reduction='none'
        )

    per_pixel_loss = per_bin_loss.mean(
        dim=1,
    )

    valid_weight = valid_low.to(
        dtype=per_pixel_loss.dtype,
    )

    normalizer = valid_weight.sum().clamp_min(1.0)

    return(
        per_pixel_loss * valid_weight
    ).sum() / normalizer

def intermediate_disparity_loss(
        pred_disp,
        gt_disp,
        valid=None,
        max_disp=None,
        beta=1.0,
        band_weight=None,
):
    if pred_disp.ndim != 4 or pred_disp.shape[1] != 1:
        raise ValueError(
            f'pred_disp must have shape [B, 1, H, W], got {pred_disp.shape}'
        )
    if gt_disp.ndim != 4 or gt_disp.shape[1] != 1:
        raise ValueError(
            f'gt_disp must have shape [B, 1, H, W], got {gt_disp.shape}'
        )

    if pred_disp.shape[0] != gt_disp.shape[0]:
        raise ValueError(
            'pred_disp and gt_disp must have the same batch size'
        )
    gt_disp = gt_disp.to(
        device=pred_disp.device,
        dtype=pred_disp.dtype,
    )

    target_size = pred_disp.shape[-2:]

    gt_disp_low, valid_disp = _masked_average_disparity(
        gt_disp=gt_disp,
        valid=valid,
        target_size=target_size,
        max_disp=max_disp,
    )
    gt_disp_low = gt_disp_low.to(
        device=pred_disp.device,
        dtype=pred_disp.dtype,
    )

    if not valid_disp.any():
        return pred_disp.sum() * 0.0

    pred_valid = pred_disp[:, 0][valid_disp]
    gt_valid = gt_disp_low[:, 0][valid_disp]

    band = _validate_band_weight(band_weight)
    if band is None:
        return F.smooth_l1_loss(
            pred_valid,
            gt_valid,
            reduction="mean",
            beta=beta,
        )
    threshold, weight = band
    return _weighted_mean_smooth_l1(
        pred_valid,
        gt_valid,
        beta,
        threshold,
        weight,
    )

def final_disparity_loss(
    pred_disp,
    gt_disp,
    valid=None,
    max_disp=None,
    beta=1.0,
    band_weight=None,
    near_add=None,
):
    if pred_disp.ndim != 4 or pred_disp.shape[1] != 1:
        raise ValueError(
            f"pred_disp must have shape [B, 1, H, W], got {pred_disp.shape}"
        )

    if gt_disp.ndim != 4 or gt_disp.shape[1] != 1:
        raise ValueError(
            f"gt_disp must have shape [B, 1, H, W], got {gt_disp.shape}"
        )

    if pred_disp.shape != gt_disp.shape:
        raise ValueError(
            "pred_disp and gt_disp must have the same shape, "
            f"got {pred_disp.shape} and {gt_disp.shape}"
        )

    gt_disp = gt_disp.to(
        device=pred_disp.device,
        dtype=pred_disp.dtype,
    )

    valid_disp = (
        torch.isfinite(gt_disp[:, 0])
        & (gt_disp[:, 0] >= 0)
    )

    if max_disp is not None:
        if max_disp <= 0:
            raise ValueError(
                f"max_disp must be positive, got {max_disp}"
            )

        valid_disp = valid_disp & (
            gt_disp[:, 0] < max_disp
        )

    if valid is not None:
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)

        if valid.ndim != 4 or valid.shape[1] != 1:
            raise ValueError(
                f"valid must have shape [B, H, W] or [B, 1, H, W], got {valid.shape}"
            )

        if valid.shape != gt_disp.shape:
            raise ValueError(
                "valid and gt_disp must have the same shape, "
                f"got {valid.shape} and {gt_disp.shape}"
            )

        valid_disp = valid_disp & (
            valid[:, 0] > 0.5
        )

    if not valid_disp.any():
        return pred_disp.sum() * 0.0

    pred_valid = pred_disp[:, 0][valid_disp]
    gt_valid = gt_disp[:, 0][valid_disp]

    band = _validate_band_weight(band_weight)
    if band is None:
        base = F.smooth_l1_loss(
            pred_valid,
            gt_valid,
            reduction="mean",
            beta=beta,
        )
    else:
        threshold, weight = band
        base = _weighted_mean_smooth_l1(
            pred_valid,
            gt_valid,
            beta,
            threshold,
            weight,
        )

    near = _validate_near_add(near_add)
    if near is None:
        return base
    near_threshold, near_weight = near
    near_mask = gt_valid >= near_threshold
    if not near_mask.any():
        return base
    near_term = F.smooth_l1_loss(
        pred_valid[near_mask],
        gt_valid[near_mask],
        reduction="mean",
        beta=beta,
    )
    return base + near_weight * near_term


