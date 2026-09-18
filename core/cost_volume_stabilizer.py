import torch
import torch.nn as nn


class FeatureGuidedCostResidual(nn.Module):
    def __init__(
            self,
            cost_channels,
            guide_channels,
            hidden_channels,
            negative_slope=0.1,
            residual_scale_init=0.0,
    ):
        super().__init__()

        if cost_channels <= 0:
            raise ValueError(
                f'cost_channels must be positive, got {cost_channels}'
            )
        if guide_channels <= 0:
            raise ValueError(
                f'guide_channels must be positive, got {guide_channels}'
            )
        if hidden_channels <= 0:
            raise ValueError(
                f'hidden_channels must be positive, got {hidden_channels}'
            )
        if negative_slope < 0:
            raise ValueError(
                f'negative_slope must be non-negative, got {negative_slope}'
            )

        self.cost_channels = cost_channels
        self.guide_channels = guide_channels

        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )

        self.refinement = nn.Sequential(
            nn.Conv2d(
                cost_channels + guide_channels,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(
                negative_slope=negative_slope,
                inplace=True,
            ),
            nn.Conv2d(
                hidden_channels,
                cost_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )

    def forward(self, cost, guide):
        if cost.ndim != 4:
            raise ValueError(
                f'cost must have shape [B, D, H, W], got {cost.shape}'
            )
        if guide.ndim != 4:
            raise ValueError(
                f'guide must have shape [B, C, H, W], got {guide.shape}'
            )
        if cost.shape[0] != guide.shape[0]:
            raise ValueError(
                'cost and guide must have the same batch size, '
                f'got {cost.shape} and {guide.shape}'
            )
        if cost.shape[2:] != guide.shape[2:]:
            raise ValueError(
                'cost and guide must have the same spatial size, '
                f'got {cost.shape} and {guide.shape}'
            )
        if cost.shape[1] != self.cost_channels:
            raise ValueError(
                f'Expected cost channels {self.cost_channels}, '
                f'got {cost.shape[1]}'
            )
        if guide.shape[1] != self.guide_channels:
            raise ValueError(
                f'Expected guide channels {self.guide_channels}, '
                f'got {guide.shape[1]}'
            )

        refinement_input = torch.cat(
            [cost, guide],
            dim=1,
        )
        correction = self.refinement(refinement_input)

        return cost + (
            self.residual_scale * correction
        )


class CostVolumeStabilizer(nn.Module):
    def __init__(
            self,
            cost_channels,
            guide_channels,
            hidden_channels=None,
            negative_slope=0.1,
            residual_scale_init=0.0,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = cost_channels

        self.cost_channels = cost_channels
        self.guide_channels = guide_channels
        self.hidden_channels = hidden_channels

        self.stage_1 = FeatureGuidedCostResidual(
            cost_channels=cost_channels,
            guide_channels=guide_channels,
            hidden_channels=hidden_channels,
            negative_slope=negative_slope,
            residual_scale_init=residual_scale_init,
        )
        self.stage_2 = FeatureGuidedCostResidual(
            cost_channels=cost_channels,
            guide_channels=guide_channels,
            hidden_channels=hidden_channels,
            negative_slope=negative_slope,
            residual_scale_init=residual_scale_init,
        )

    def forward(self, cost_c0, guide):
        cost_c1 = self.stage_1(
            cost_c0,
            guide,
        )
        cost_c2 = self.stage_2(
            cost_c1,
            guide,
        )

        return {
            'cost_c0': cost_c0,
            'cost_c1': cost_c1,
            'cost_c2': cost_c2,
        }
