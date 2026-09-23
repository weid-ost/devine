"""Data and ML layer

Module contains functions and classes to load and save models (data layer) and to train and define model architectures (ML layer).
"""

import json
from pathlib import Path
from typing import Literal, Sequence, Type

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn

from .model_parts import (
    DoubleConv,
    OutputLayer,
    StackDoubleConv,
    UpConv,
)

ActivationLike = Type[nn.Module] | nn.Module | str


class WindUNet(LightningModule):
    """U-Net-style model predicting three wind components."""

    def __init__(
        self,
        input_channels: int,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        activation: ActivationLike = nn.ReLU,
        loss_fn: Sequence[nn.Module] | None = None,
        metric_fns: Sequence[nn.Module] | None = None,
        comp_losses: bool = False,
    ):
        super().__init__()

        # Le Toumelin adds zero padding 0 1 0 1 (top bottom left right): 79x69 -> 80x70
        # self.zero_padding = nn.ZeroPad2d((0, 1, 0, 1))

        channels: int = 32

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.activation = resolve_activation(activation)

        self.conv1 = DoubleConv(input_channels, channels, activation=self.activation)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(channels, 2 * channels, activation=self.activation)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(2 * channels, 4 * channels, activation=self.activation)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(4 * channels, 8 * channels, activation=self.activation)

        # NOTE: kernel_size 2 throws a performance and memory warning. Not sure how this
        # was handled in tensorflow before the port to pytorch. Kept it as as for now.
        self.up4 = UpConv(
            8 * channels,
            4 * channels,
            kernel_size=2,
            activation=self.activation,
        )

        self.stack_conv3 = StackDoubleConv(
            2 * 4 * channels, 4 * channels, activation=self.activation
        )

        # NOTE: kernel_size 2 throws a performance and memory warning. Not sure how this
        # was handled in tensorflow before the port to pytorch. Kept it as as for now.
        self.up3 = UpConv(
            4 * channels,
            2 * channels,
            kernel_size=2,
            activation=self.activation,
        )

        self.stack_conv2 = StackDoubleConv(
            4 * channels, 2 * channels, activation=self.activation
        )

        # NOTE: kernel_size 2 throws a performance and memory warning. Not sure how this
        # was handled in tensorflow before the port to pytorch. Kept it as as for now.
        self.up2 = UpConv(
            2 * channels,
            1 * channels,
            kernel_size=2,
            activation=self.activation,
        )

        self.stack_conv1 = StackDoubleConv(
            2 * channels, 1 * channels, activation=self.activation
        )

        self.out_layer = OutputLayer(1 * channels, 3)

        self.loss_fn = loss_fn if loss_fn is not None else [nn.MSELoss()]
        self.metric_fns = metric_fns
        self.comp_losses = comp_losses

        self.dropout = nn.Dropout()

        self.save_hyperparameters()

    def forward(self, x):
        """Forward pass returning predicted wind components."""

        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        up4 = self.up4(conv4)

        # The argument order in the stacking conv layer is important.
        # The shape of the first argument is imposed on the second via padding.
        # conv3 (M x N), up4 (K x L) -> (M x N)
        up4 = self.stack_conv3(conv3, up4)

        up3 = self.up3(up4)

        up3 = self.stack_conv2(conv2, up3)

        up2 = self.up2(up3)

        up2 = self.stack_conv1(conv1, up2)

        output = self.out_layer(up2)
        return output

    def training_step(self, batch):
        """Compute training loss and optional per-component breakdown."""
        X, y = batch
        x_hat = self(X)
        loss = torch.tensor([0.0]).to("cuda")
        for l_fn in self.loss_fn:
            loss_ = l_fn(x_hat, y)
            self.log(
                f"train_loss_{l_fn.__class__.__name__}",
                loss_,
                on_step=True,
                on_epoch=True,
            )
            loss += loss_

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
        )

        if self.comp_losses:
            for l_fn in self.loss_fn:
                if l_fn.__class__.__name__ == "NMSELoss":
                    loss_u, loss_v, loss_w = l_fn(
                        x_hat, y, comp_losses=self.comp_losses
                    )
                else:
                    loss_u = l_fn(
                        x_hat[:, 0].unsqueeze(dim=1), y[:, 0].unsqueeze(dim=1)
                    )
                    loss_v = l_fn(
                        x_hat[:, 1].unsqueeze(dim=1), y[:, 1].unsqueeze(dim=1)
                    )
                    loss_w = l_fn(
                        x_hat[:, 2].unsqueeze(dim=1), y[:, 2].unsqueeze(dim=1)
                    )
                self.log(
                    f"train_loss_u_{l_fn.__class__.__name__}",
                    loss_u,
                    on_step=True,
                    on_epoch=True,
                )
                self.log(
                    f"train_loss_v_{l_fn.__class__.__name__}",
                    loss_v,
                    on_step=True,
                    on_epoch=True,
                )
                self.log(
                    f"train_loss_w_{l_fn.__class__.__name__}",
                    loss_w,
                    on_step=True,
                    on_epoch=True,
                )

        if self.metric_fns is not None:
            for metric_fn in self.metric_fns:
                metric = metric_fn(x_hat, y)
                self.log(
                    f"{metric_fn.__class__.__name__}",
                    metric,
                    on_step=True,
                    on_epoch=True,
                )

        return loss

    def validation_step(self, batch):
        """Compute validation loss and optional per-component breakdown."""
        X, y = batch
        x_hat = self(X)
        losses = []
        for l_fn in self.loss_fn:
            loss_ = l_fn(x_hat, y)
            self.log(
                f"val_loss_{l_fn.__class__.__name__}",
                loss_,
                on_step=True,
                on_epoch=True,
            )
            losses.append(loss_)
        loss = torch.sum(torch.tensor(losses))
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
        )

        if self.comp_losses:
            for l_fn in self.loss_fn:
                if l_fn.__class__.__name__ == "NMSELoss":
                    loss_u, loss_v, loss_w = l_fn(
                        x_hat, y, comp_losses=self.comp_losses
                    )

                else:
                    loss_u = l_fn(
                        x_hat[:, 0].unsqueeze(dim=1), y[:, 0].unsqueeze(dim=1)
                    )
                    loss_v = l_fn(
                        x_hat[:, 1].unsqueeze(dim=1), y[:, 1].unsqueeze(dim=1)
                    )
                    loss_w = l_fn(
                        x_hat[:, 2].unsqueeze(dim=1), y[:, 2].unsqueeze(dim=1)
                    )
                self.log(
                    f"val_loss_u_{l_fn.__class__.__name__}",
                    loss_u,
                    on_step=True,
                    on_epoch=True,
                )
                self.log(
                    f"val_loss_v_{l_fn.__class__.__name__}",
                    loss_v,
                    on_step=True,
                    on_epoch=True,
                )
                self.log(
                    f"val_loss_w_{l_fn.__class__.__name__}",
                    loss_w,
                    on_step=True,
                    on_epoch=True,
                )

        if self.metric_fns is not None:
            for metric_fn in self.metric_fns:
                metric = metric_fn(x_hat, y)
                self.log(
                    f"{metric_fn.__class__.__name__}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                )

    def configure_optimizers(self):
        """Configure the Adam optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer


class DujardinLoss(nn.Module):
    """Loss combining magnitude-aware scaling with MSE on the vertical component."""

    def forward(self, y, yhat):
        """Compute loss balancing horizontal and vertical wind components."""
        #    Parameters
        epsilon = 4
        tau = 0.425
        #    Scaling u and v to avoid squeezed distributions of predicted velocity
        vel_target = torch.sqrt(torch.sum(y[:, :2] ** 2, 1, keepdim=True))
        vel_input = torch.sqrt(torch.sum(yhat[:, :2] ** 2, 1, keepdim=True))
        beta = (epsilon + vel_target) / (epsilon + vel_input)
        err = torch.sum((yhat[:, :2] - y[:, :2] * beta) ** 2, 1)
        #    Pinball term to reduce the bias of the prediction
        loss = err * tau
        ind_neg = (vel_input - vel_target) < 0
        ind_neg = ind_neg[:, 0]
        loss[ind_neg] = err[ind_neg] * (1 - tau)
        loss = torch.mean(loss)

        w = y[:, 2]
        what = yhat[:, 2]
        return loss + F.mse_loss(what, w)


class NMSELoss(nn.Module):
    """Normalised MSE with optional per-component returns."""

    def __init__(self, std):
        super().__init__()

        self.std = std

    def forward(
        self, yhat, y, comp_losses: bool = False
    ) -> float | tuple[float, float, float]:
        """Compute NMSE, optionally returning per-component losses."""
        weights = 1 / self.std.square()
        loss_u = F.mse_loss(yhat[:, 0], y[:, 0]) / weights[0]
        loss_v = F.mse_loss(yhat[:, 1], y[:, 1]) / weights[1]
        loss_w = F.mse_loss(yhat[:, 2], y[:, 2]) / weights[2]

        if comp_losses:
            return loss_u, loss_v, loss_w

        return loss_u + loss_v + loss_w


class DirectionLoss(nn.Module):
    """Penalise angular difference between prediction and target."""

    def __init__(self, reduction: Literal["mean", "sum"] = "mean"):
        super().__init__()
        redu_dict = {"mean": torch.mean, "sum": torch.sum}

        self.redu_fn = redu_dict[reduction]

    def forward(self, yhat, y):
        ab = (yhat * y).sum(dim=1, keepdim=True)
        a_norm = (yhat * yhat).sum(dim=1, keepdim=True).sqrt()
        b_norm = (y * y).sum(dim=1, keepdim=True).sqrt()
        loss = 1.0 - ab / (a_norm * b_norm + 1e-5)
        return loss.mean()


class L2Loss(nn.Module):
    ""

    def __init__(self, reduction: Literal["mean", "sum"] = "mean"):
        super().__init__()
        redu_dict = {"mean": torch.mean, "sum": torch.sum}

        self.redu_fn = redu_dict[reduction]

    def forward(self, yhat, y):
        return self.redu_fn((yhat - y).square().mean(axis=(0, 2, 3)).sqrt())


MSELoss = nn.MSELoss
L1Loss = nn.L1Loss
HuberLoss = nn.HuberLoss


def load_weight_json(filename: str | Path, transpose: bool = False):
    """Load weights from JSON into tensors, optionally transposing spatial dims.

    This function is only used for the pretrained 'devine' model.
    """
    with open(filename) as f:
        weights_ = json.load(f)

    weights = {}
    for k, v in weights_.items():
        arr = np.array(v)
        if len(arr.shape) > 1 and transpose:
            arr = np.transpose(arr, (0, 1, 3, 2))
        weights[k] = torch.from_numpy(arr)
    return weights


def torch_tensor_serializer(obj):
    """JSON serializer helper for torch tensors."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"{type(obj)} not serializable")


def tensorflow_to_pytorch_weights(pytorch_model, weights_file: str | Path):
    """Convert stored TensorFlow-formatted weights into a PyTorch state dict."""
    weights = load_weight_json(weights_file, transpose=True)

    state_dict = pytorch_model.state_dict()
    new_state_dict = {}

    for a, b in zip(weights, state_dict):
        w_shape = weights[a].shape
        assert w_shape == state_dict[b].shape
        new_state_dict[b] = weights[a]

    return new_state_dict


def resolve_activation(activation: ActivationLike) -> nn.Module:
    """Resolve activation given as a class, instance, or string name."""
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation()

    if isinstance(activation, nn.Module):
        return activation

    if isinstance(activation, str):
        cls = getattr(nn, activation)
        return cls()

    raise TypeError(f"activation must be of type ActivationLike")
