import torch.nn as nn
import timm

from .hfe import HighFrequencyEncoder, ResidualFeatureFusion
from .submodule import BasicConv2d, FPNLayer


FASTERNET_T0_MODEL = "fasternet_t0"
LAS2_FEATURE_CHANNELS = [40, 80, 160, 320]


class FeatureNetFasterNetHFE(nn.Module):
    """FasterNet feature pyramid fused with high-frequency features."""

    def __init__(self, pretrained=True, cutoff_ratio=0.1):
        super().__init__()

        self.backbone = timm.create_model(
            FASTERNET_T0_MODEL,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        self.feature_channels = list(self.backbone.feature_info.channels())
        if self.feature_channels != LAS2_FEATURE_CHANNELS:
            raise ValueError(
                f"Expected {FASTERNET_T0_MODEL} channels {LAS2_FEATURE_CHANNELS}, "
                f"got {self.feature_channels}"
            )

        channels = self.feature_channels

        self.hfe = HighFrequencyEncoder(
            in_channels=3,
            output_channels=tuple(channels[:3]),
            cutoff_ratio=cutoff_ratio,
        )

        self.fusion_1_4 = ResidualFeatureFusion(channels[0])
        self.fusion_1_8 = ResidualFeatureFusion(channels[1])
        self.fusion_1_16 = ResidualFeatureFusion(channels[2])

        self.fpn_layer4 = FPNLayer(channels[3], channels[2])
        self.fpn_layer3 = FPNLayer(channels[2], channels[1])
        self.fpn_layer2 = FPNLayer(channels[1], channels[0])

        self.out_conv = BasicConv2d(
            channels[0],
            channels[0],
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            norm_layer=nn.InstanceNorm2d,
        )

    def forward(self, images):
        c2, c3, c4, c5 = self.backbone(images)

        hfe_1_4, hfe_1_8, hfe_1_16 = self.hfe(images)

        c2 = self.fusion_1_4(c2, hfe_1_4)
        c3 = self.fusion_1_8(c3, hfe_1_8)
        c4 = self.fusion_1_16(c4, hfe_1_16)

        p4 = self.fpn_layer4(c5, c4)
        p3 = self.fpn_layer3(p4, c3)
        p2 = self.fpn_layer2(p3, c2)
        p2 = self.out_conv(p2)

        return [p2, p3, p4, c5]
