import torch
import torch.nn as nn

class HighPassFilter(nn.Module):
    def __init__(self, cutoff_ratio=0.1):
        super().__init__()

        if not 0.0 < cutoff_ratio < 1.0:
            raise ValueError('cutoff_ratio must be between 0 and 1')

        self.cutoff_ratio = cutoff_ratio

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape [B, C, H, W], got {images.shape}")

        _, _, height, width = images.shape

        frequency_y = (
            torch.fft.fftshift(
                torch.fft.fftfreq(
                    height,
                    device=images.device,
                    dtype=torch.float32,
                )
            ) * 2.0
        ).view(1, 1, height, 1)

        frequency_x = (
            torch.fft.fftshift(
                torch.fft.fftfreq(
                    width,
                    device=images.device,
                    dtype=torch.float32,
                )
            ) * 2.0
        ).view(1, 1, 1, width)

        radius = torch.sqrt(frequency_y.square() + frequency_x.square())

        high_pass_mask = (radius >= self.cutoff_ratio).to(torch.float32)

        images_float = images.float()

        spectrum = torch.fft.fft2(
            images_float,
            dim=(-2, -1),
            norm='ortho',
        )

        spectrum = torch.fft.fftshift( 
            spectrum,
            dim=(-2, -1),
        )

        filtered_spectrum = spectrum * high_pass_mask

        filtered_spectrum = torch.fft.ifftshift(
            filtered_spectrum,
            dim=(-2, -1),
        )

        high_frequency_images = torch.fft.ifft2(
            filtered_spectrum,
            dim=(-2, -1),
            norm='ortho',
        ).real

        return high_frequency_images.to(dtype=images.dtype)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels)
            )
        self.output_act = nn.GELU()

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + shortcut
        out = self.output_act(out)

        return out

class HighFrequencyEncoder(nn.Module):
    def __init__(self, in_channels=3, output_channels=(40, 80, 160), cutoff_ratio=0.1):
        super().__init__()

        if len(output_channels) != 3:
            raise ValueError('output_channels must contain three channel values')

        channels_1_4, channels_1_8, channels_1_16 = output_channels

        self.high_pass_filter = HighPassFilter(
            cutoff_ratio=cutoff_ratio,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels,
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

    def forward(self, images):
        high_frequency_images = self.high_pass_filter(images)

        x = self.stem(high_frequency_images)

        feature_1_4  = self.stage_1_4(x)
        feature_1_8  = self.stage_1_8(feature_1_4)
        feature_1_16  = self.stage_1_16(feature_1_8)

        return [feature_1_4,
                feature_1_8,
                feature_1_16
                ]


class ResidualFeatureFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.residual_scale = nn.Parameter(
            torch.tensor(0.01)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, backbone_feature, hfe_feature):
        if backbone_feature.shape[2:] != hfe_feature.shape[2:]:
            raise ValueError(
                'Backbone and HFE feature must have the same spatial size, '
                f'got {backbone_feature.shape} and {hfe_feature.shape}'
            )

        fused_input = torch.cat(
            [backbone_feature, hfe_feature],
            dim=1,
        )

        return backbone_feature + (
            self.residual_scale * self.fusion(fused_input)
        )