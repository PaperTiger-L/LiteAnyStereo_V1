import timm
import torch.nn as nn

from .hfe import HighPassFilter, ResidualConvBlock, ResidualFeatureFusion
from .submodule import BasicConv2d, FPNLayer


FASTERNET_T0_MODEL = "fasternet_t0"
LAS2_FEATURE_CHANNELS = [40, 80, 160, 320]


class HighFrequencyEncoderPyramid(nn.Module):
    """Four-scale high-frequency encoder from 1/4 through 1/32 resolution."""

    def __init__(
            self,
            in_channels=3,
            output_channels=(40, 80, 160, 320),
            cutoff_ratio=0.1,
    ):
        super().__init__()

        if len(output_channels) != 4:
            raise ValueError(
                "output_channels must contain four channel values"
            )

        (
            channels_1_4,
            channels_1_8,
            channels_1_16,
            channels_1_32,
        ) = output_channels

        self.high_pass_filter = HighPassFilter(
            cutoff_ratio=cutoff_ratio,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.stage_1_4 = nn.Sequential(
            ResidualConvBlock(
                32,
                channels_1_4,
                stride=2,
            ),
            ResidualConvBlock(
                channels_1_4,
                channels_1_4,
                stride=1,
            ),
        )

        self.stage_1_8 = nn.Sequential(
            ResidualConvBlock(
                channels_1_4,
                channels_1_8,
                stride=2,
            ),
            ResidualConvBlock(
                channels_1_8,
                channels_1_8,
                stride=1,
            ),
        )

        self.stage_1_16 = nn.Sequential(
            ResidualConvBlock(
                channels_1_8,
                channels_1_16,
                stride=2,
            ),
            ResidualConvBlock(
                channels_1_16,
                channels_1_16,
                stride=1,
            ),
        )

        self.stage_1_32 = nn.Sequential(
            ResidualConvBlock(
                channels_1_16,
                channels_1_32,
                stride=2,
            ),
            ResidualConvBlock(
                channels_1_32,
                channels_1_32,
                stride=1,
            ),
        )

    def forward(self, images):
        high_frequency_images = self.high_pass_filter(images)

        x = self.stem(high_frequency_images)
        feature_1_4 = self.stage_1_4(x)
        feature_1_8 = self.stage_1_8(feature_1_4)
        feature_1_16 = self.stage_1_16(feature_1_8)
        feature_1_32 = self.stage_1_32(feature_1_16)

        return [
            feature_1_4,
            feature_1_8,
            feature_1_16,
            feature_1_32,
        ]


class FeatureNetFasterNetHFEMultiscale(nn.Module):
    """FasterNet and four-scale HFE fusion followed by top-down FPN."""

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
                f"Expected {FASTERNET_T0_MODEL} channels "
                f"{LAS2_FEATURE_CHANNELS}, got {self.feature_channels}"
            )

        channels = self.feature_channels

        self.hfe = HighFrequencyEncoderPyramid(
            in_channels=3,
            output_channels=tuple(channels),
            cutoff_ratio=cutoff_ratio,
        )

        self.fusion_1_4 = ResidualFeatureFusion(channels[0])
        self.fusion_1_8 = ResidualFeatureFusion(channels[1])
        self.fusion_1_16 = ResidualFeatureFusion(channels[2])
        self.fusion_1_32 = ResidualFeatureFusion(channels[3])

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

        (
            hfe_1_4,
            hfe_1_8,
            hfe_1_16,
            hfe_1_32,
        ) = self.hfe(images)

        fused_1_4 = self.fusion_1_4(c2, hfe_1_4)
        fused_1_8 = self.fusion_1_8(c3, hfe_1_8)
        fused_1_16 = self.fusion_1_16(c4, hfe_1_16)
        fused_1_32 = self.fusion_1_32(c5, hfe_1_32)

        p4 = self.fpn_layer4(fused_1_32, fused_1_16)
        p3 = self.fpn_layer3(p4, fused_1_8)
        p2 = self.fpn_layer2(p3, fused_1_4)
        p2 = self.out_conv(p2)

        return [p2, p3, p4, fused_1_32]
