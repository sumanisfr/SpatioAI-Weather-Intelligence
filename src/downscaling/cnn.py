"""
PyTorch U-Net downscaling convolutional architecture for SpatioAI Phase 5.
Supports configurable depth, channel width, residual learning (Prediction = Bilinear + Residual),
and positive output activations (Softplus/ReLU) to enforce physical constraints.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> LeakyReLU) * 2 with padding to preserve spatial dimensions."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling block with MaxPool2d followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upscaling block with bilinear interpolation / ConvTranspose2d followed by DoubleConv and skip connection."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Handle potential padding discrepancies between encoder and decoder dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        # Concatenate along channel axis (skip connection)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetDownscaler(nn.Module):
    """
    U-Net convolutional neural network for high-resolution weather downscaling.

    Features:
        - Multi-scale encoder-decoder with skip connections
        - Configurable base channels and depth
        - Residual learning: y_pred = x_interpolated + residual (when residual_learning=True)
        - Physical positivity constraint via Softplus activation
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        bilinear: bool = True,
        residual_learning: bool = True,
        output_activation: str = "softplus",
    ):
        """
        Args:
            in_channels: Number of input channels (e.g., 1 for precipitation, or 5 for multi-variable).
            out_channels: Number of output channels (e.g., 1 for high-res precipitation).
            base_channels: Number of filters in first encoder layer (default: 32).
            bilinear: Use bilinear upsampling instead of transpose convolutions.
            residual_learning: Learn fine-scale residual correction on top of bilinear baseline.
            output_activation: Activation for positivity constraint ('softplus', 'relu', or 'none').
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.residual_learning = residual_learning
        self.output_activation = output_activation.lower()

        # Encoder (Downsampling path)
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.bottleneck = DownBlock(base_channels * 8, (base_channels * 16) // factor)

        # Decoder (Upsampling path with skip connections)
        self.up1 = UpBlock(base_channels * 16, (base_channels * 8) // factor, bilinear)
        self.up2 = UpBlock(base_channels * 8, (base_channels * 4) // factor, bilinear)
        self.up3 = UpBlock(base_channels * 4, (base_channels * 2) // factor, bilinear)
        self.up4 = UpBlock(base_channels * 2, base_channels, bilinear)

        # Final 1x1 projection
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # Optional softplus for non-negative weather fields (precipitation)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor resampled to high-resolution grid [Batch, In_Channels, H, W].

        Returns:
            High-resolution prediction [Batch, Out_Channels, H, W].
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        # Decoder with skip connections
        d = self.up1(x5, x4)
        d = self.up2(d, x3)
        d = self.up3(d, x2)
        d = self.up4(d, x1)

        # Output projection
        raw_out = self.out_conv(d)

        if self.residual_learning:
            # Additive residual on top of coarse baseline
            # If in_channels == out_channels, add directly to input slice
            base = x[:, : self.out_channels, :, :]
            pred = base + raw_out
        else:
            pred = raw_out

        if self.output_activation == "softplus":
            return self.softplus(pred)
        elif self.output_activation == "relu":
            return F.relu(pred)
        else:
            return pred
