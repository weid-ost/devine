import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    "conv -> conv -> (Dropout) -> (BatchNorm)"

    def __init__(
        self,
        in_channels,
        out_channels,
        batch_norm: bool = False,
        dropout: bool = False,
        activation: nn.Module | None = None,
    ):
        super().__init__()

        self.activation = activation if activation is not None else nn.ReLU()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            self.activation,
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            self.activation,
        )

        if dropout:
            self.double_conv.append(nn.Dropout(out_channels))
        if batch_norm:
            self.double_conv.append(nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvDown(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        batch_norm: bool = False,
        dropout: bool = False,
        **double_conv_kwargs,
    ):
        super().__init__()

        self.conv_down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels,
                out_channels,
                batch_norm,
                dropout,
                **double_conv_kwargs,
            ),
        )

    def forward(self, x):
        return self.conv_down(x)


class UpConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        factor: int = 2,
        kernel_size=3,
        activation: nn.Module | None = None,
    ):
        super().__init__()

        self.activation = activation if activation is not None else nn.ReLU()

        self.conv_up = nn.Sequential(
            nn.Upsample(scale_factor=factor, mode="bilinear", align_corners=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            self.activation,
        )

    def forward(self, x):
        return self.conv_up(x)


class StackDoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        **double_conv_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.double_conv = DoubleConv(in_channels, out_channels, **double_conv_kwargs)

    def forward(self, x1, x2):
        """
        x1 (M x N), x2 (K x L) -> (M x N)
        """

        # This is done to match the shape of the layers in the encode stage and the decode stage for the stacking operation.
        # See https://github.com/milesial/Pytorch-UNet/blob/21d7850f2af30a9695bbeea75f3136aa538cfc4a/unet/unet_parts.py#L56
        diff_y = x1.size()[2] - x2.size()[2]
        diff_x = x1.size()[3] - x2.size()[3]

        padding = [
            diff_x // 2,
            diff_x - diff_x // 2,
            diff_y // 2,
            diff_y - diff_y // 2,
        ]
        x2 = F.pad(x2, padding)

        x = torch.cat([x1, x2], dim=1)
        return self.double_conv(x)


class OutputLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding="same"),
        )

    def forward(self, x):
        return self.output_conv(x)
