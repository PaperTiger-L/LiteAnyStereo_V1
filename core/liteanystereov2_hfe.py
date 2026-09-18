import torch
import torch.nn as nn
import torch.nn.functional as F
from .cost_volume_stabilizer import CostVolumeStabilizer
from .fnet_hfe import FASTERNET_T0_MODEL, FeatureNetFasterNetHFE
from .submodule import (
    BasicConv2d,
    BasicDeconv2d,
    FPNLayer,
    build_correlation_volume,
    context_upsample,
    disparity_regression,
)


def build_liteanystereo_hfe(
        fnet_pretrained=False,
        cutoff_ratio=0.1,
        max_disp=192,
        cost_stabilization=None,
):
    return LiteAnyStereoSHFE(
        fnet_pretrained=fnet_pretrained,
        cutoff_ratio=cutoff_ratio,
        max_disp=max_disp,
        cost_stabilization=cost_stabilization,
    )


class LiteAnyStereoSHFE(nn.Module):
    """LAS2-S stereo model with high-frequency feature encoding."""

    def __init__(
            self,
            fnet_pretrained=False,
            cutoff_ratio=0.1,
            max_disp=192,
            cost_stabilization=None,
    ):
        super().__init__()

        if max_disp <= 0 or max_disp % 4 != 0:
            raise ValueError(
                f'max_disp must be a positive multiple of 4, got {max_disp}'
            )

        if cost_stabilization is None:
            cost_stabilization = {}

        if not isinstance(cost_stabilization, dict):
            raise TypeError(
                'cost_stabilization must be a dictionary'
            )

        if not cost_stabilization.get('ENABLED', True):
            raise ValueError(
                'Cost stabilization must be enabled for LiteAnyStereoSHFE'
            )

        self.model_size = "s"
        self.fnet_name = FASTERNET_T0_MODEL
        self.max_disp = max_disp
        self.cost_channels = max_disp // 4
        self.fnet = FeatureNetFasterNetHFE(
            pretrained=fnet_pretrained,
            cutoff_ratio=cutoff_ratio,
        )
        self.fnet_channels = self.fnet.feature_channels

        hidden_channels = cost_stabilization.get(
            'HIDDEN_CHANNELS'
        )
        if hidden_channels is None:
            hidden_channels = self.cost_channels

        self.cost_stabilizer = CostVolumeStabilizer(
            cost_channels=self.cost_channels,
            guide_channels=self.fnet_channels[0],
            hidden_channels=hidden_channels,
            negative_slope=cost_stabilization.get(
                'LEAKY_RELU_SLOPE',
                0.1,
            ),
            residual_scale_init=cost_stabilization.get(
                'RESIDUAL_SCALE_INIT',
                0.0,
            ),
        )

        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        self.refine_1 = nn.Sequential(
            BasicConv2d(
                self.fnet_channels[0],
                self.fnet_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.InstanceNorm2d,
                act_layer=nn.LeakyReLU,
            ),
            BasicConv2d(
                self.fnet_channels[0],
                self.fnet_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.InstanceNorm2d,
                act_layer=nn.ReLU,
            ),
        )

        self.stem_2 = nn.Sequential(
            BasicConv2d(
                3,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.LeakyReLU,
            ),
            BasicConv2d(
                16,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU,
            ),
        )
        self.refine_2 = FPNLayer(self.fnet_channels[0], 16)
        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)

    def normalize_image(self, img):
        """Normalize RGB images in 0-255 range using ImageNet statistics."""
        return ((img / 255.0 - self.image_mean) / self.image_std).contiguous()

    def forward(
            self,
            left,
            right,
            max_disp=192,
            test_mode=False,
            kd_mode=False,
            jetson_mode=False,
            return_aux=False,
    ):
        del jetson_mode

        if max_disp <= 0 or max_disp % 4 != 0:
            raise ValueError(
                f"max_disp must be a positive multiple of 4, got {max_disp}"
            )


        num_disparities = max_disp // 4
        if num_disparities != self.cost_channels:
            raise ValueError(
                f'Expected max_disp={self.max_disp}, got {max_disp}'
            )

        left = self.normalize_image(left)
        right = self.normalize_image(right)

        features_left = self.fnet(left)
        features_right = self.fnet(right)

        cost_c0 = build_correlation_volume(
            features_left[0],
            features_right[0],
            num_disparities,
        )
        stabilized_costs = self.cost_stabilizer(
            cost_c0,
            features_left[0],
        )
        cost_c1 = stabilized_costs['cost_c1']
        cost_c2 = stabilized_costs['cost_c2']

        cost_prob_c0 = F.softmax(
            cost_c0,
            dim=1,
        )
        cost_prob_c2 = F.softmax(
            cost_c2,
            dim=1,
        )

        disp_bins_c0 = disparity_regression(
            cost_prob_c0,
            num_disparities,
        )
        disp_bins_c2 = disparity_regression(
            cost_prob_c2,
            num_disparities,
        )

        disp_low_c0 = disp_bins_c0 * 4.0
        disp_low_c2 = disp_bins_c2 * 4.0

        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(
            xspx,
            self.stem_2(left),
        )
        xspx = self.refine_3(xspx)

        spx_pred = F.softmax(
            xspx,
            dim=1,
        )

        disp_up = context_upsample(
            disp_low_c2,
            spx_pred.float(),
        )

        if test_mode:
            return disp_up

        disp_linear = F.interpolate(
            disp_low_c2,
            left.shape[2:],
            mode='bilinear',
            align_corners=False,
        )

        if return_aux:
            outputs = {
                "disp_up": disp_up,
                "disp_low": disp_low_c2,
                "disp_low_c0": disp_low_c0,
                "disp_low_c2": disp_low_c2,
                "cost_prob": cost_prob_c2,
                "cost_prob_c0": cost_prob_c0,
                "cost_prob_c2": cost_prob_c2,
                "cost_logits_c0": cost_c0,
                "cost_logits_c1": cost_c1,
                "cost_logits_c2": cost_c2,
            }
            if kd_mode:
                outputs["features_left"] = features_left
                outputs["features_right"] = features_right

            return outputs
        if kd_mode:
            return [disp_up, disp_linear], features_left, features_right
        return [disp_up, disp_linear]
