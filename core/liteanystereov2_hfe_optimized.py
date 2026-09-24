import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregation_fasternet import Aggregation
from .fnet_hfe import FASTERNET_T0_MODEL
from .fnet_hfe_multiscale import FeatureNetFasterNetHFEMultiscale
from .submodule import (
    BasicConv2d,
    BasicDeconv2d,
    FPNLayer,
    build_correlation_volume,
    context_upsample,
    disparity_regression,
)


OPTIMIZED_ARCHITECTURE_VERSION = "las2_s_hfe_multiscale_v1"


def build_liteanystereo_hfe_optimized(
        fnet_pretrained=False,
        cutoff_ratio=0.1,
        max_disp=192,
):
    return LiteAnyStereoSHFEOptimized(
        fnet_pretrained=fnet_pretrained,
        cutoff_ratio=cutoff_ratio,
        max_disp=max_disp,
    )


class LiteAnyStereoSHFEOptimized(nn.Module):
    """HFE LAS2-S with four-scale fusion and multiscale cost aggregation.

    Backbone and HFE features are fused at matching 1/4, 1/8, 1/16, and
    1/32 resolutions before the fused 1/32 feature enters the top-down FPN.
    The resulting left/right 1/4 features build the raw cost volume, after
    which the baseline's multiscale cost aggregation replaces the two local
    CVS residual stages. ``cost_logits_c1`` and ``cost_logits_c2`` both expose
    the aggregated cost for compatibility with the existing CVC/D0/D2
    training interface; no second model or inference-time fusion path is used.
    Auxiliary D0/D2 disparities remain in the 1/4-resolution coordinate system;
    only full-resolution prediction branches convert them back to image pixels.
    """

    def __init__(
            self,
            fnet_pretrained=False,
            cutoff_ratio=0.1,
            max_disp=192,
    ):
        super().__init__()

        if max_disp <= 0 or max_disp % 4 != 0:
            raise ValueError(
                f"max_disp must be a positive multiple of 4, got {max_disp}"
            )

        self.model_size = "s"
        self.fnet_name = FASTERNET_T0_MODEL
        self.max_disp = max_disp
        self.cost_channels = max_disp // 4
        self.aggregation_name = "fasternet_multiscale"
        self.architecture_version = OPTIMIZED_ARCHITECTURE_VERSION
        self.aux_disparity_scale = 4.0

        self.fnet = FeatureNetFasterNetHFEMultiscale(
            pretrained=fnet_pretrained,
            cutoff_ratio=cutoff_ratio,
        )
        self.fnet_channels = self.fnet.feature_channels

        self.cost_agg = Aggregation(
            backbone_channels=self.fnet_channels,
            in_channels=self.cost_channels,
            left_att=True,
            blocks=[1, 2, 4],
            expanse_ratio=4,
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
        self.refine_3 = BasicDeconv2d(
            16,
            9,
            kernel_size=4,
            stride=2,
            padding=1,
        )

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
                f"Expected max_disp={self.max_disp}, got {max_disp}"
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

        cost_c2 = self.cost_agg(
            cost_c0,
            features_left,
        )
        cost_c1 = cost_c2

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

        disp_low_c0 = disp_bins_c0
        disp_low_c2 = disp_bins_c2
        disp_low_c2_pixels = (
            disp_low_c2 * self.aux_disparity_scale
        )

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
            disp_low_c2_pixels,
            spx_pred.float(),
        )

        if test_mode:
            return disp_up

        disp_linear = F.interpolate(
            disp_low_c2_pixels,
            left.shape[2:],
            mode="bilinear",
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
