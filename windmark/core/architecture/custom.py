# Copyright Grantham Taylor.

from functools import reduce
import math
import warnings
from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped

from windmark.core.constructs.general import Tokens
from windmark.core.dev.interface import TensorField


@jaxtyped(typechecker=beartype)
def smoothen(
    targets: Int[torch.Tensor, "_N *L"],  # noqa: F821
    size: int,
    sigma: float,
) -> Float[torch.Tensor, "..."]:  # noqa: F821
    """Apply gaussian smoothing to continuous targets with fixed offset for special tokens

    Arguments:
        targets (Int[torch.Tensor, "_N *L"]): Target label indices.
        size (int): The number of quantiles to smoothen over.
        sigma (float): Gaussian smoothing factor

    Returns:
        Float[torch.Tensor, "..."]: Smoothened quantile targets.
    """

    device = targets.device

    dim: int = reduce(lambda x, y: x * y, list(targets.shape))

    range_tensor = torch.arange(0, size + len(Tokens), device=device).float()

    # expand and reshape to match the batch and sequence dimensions
    range_tensor = range_tensor.unsqueeze(0).unsqueeze(0)
    labels_expanded = targets.float().unsqueeze(-1)

    # create gaussian distribution for each label in the sequence
    gaussian = torch.exp(-0.5 * ((range_tensor - labels_expanded) ** 2) / sigma**2)
    gaussian /= gaussian.sum(dim=-1, keepdim=True)

    # one-hot encoding for labels at or below the threshold
    one_hot = torch.zeros_like(gaussian).scatter_(-1, targets.unsqueeze(-1), 1.0)

    # determine which labels are above the threshold
    is_above_threshold = targets >= len(Tokens)

    # prevent gaussian bleeding for labels above the threshold
    start_bleed = torch.zeros_like(targets, dtype=torch.float32) + len(Tokens)
    start_positions = torch.where(is_above_threshold, start_bleed, targets.float())
    prevent_bleed_mask = range_tensor >= start_positions.unsqueeze(-1)

    # re-normalize
    gaussian_masked = gaussian * prevent_bleed_mask.float()
    gaussian_masked /= gaussian_masked.sum(dim=-1, keepdim=True)

    # combine using the condition
    return torch.where(is_above_threshold.unsqueeze(-1), gaussian_masked, one_hot).reshape(dim, -1)


# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# also found at: https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/optimizers/lr_scheduler.py


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr and
    base_lr followed by a cosine annealing schedule between base_lr and eta_min.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch * (base_lr - self.warmup_start_lr) / max(1, self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


def validate(inputs: TensorField) -> None:
    """
    Validate the input values and indicators.

    Args:
        inputs (TensorField): The input values to be validated.

    Raises:
        ValueError: If the shape of values and indicators are not the same.
        ValueError: If values are not imputed if not null, padded, or masked.
        ValueError: If values are not less than 1.0.
        ValueError: If values are not greater than or equal to 0.0.
    """

    values = inputs.content
    indicators = inputs.lookup

    if values.shape != indicators.shape:
        raise ValueError("values and indicators must always have the same shape")

    if not torch.all(values.mul(indicators).eq(0.0), dim=None):
        raise ValueError("values should be imputed if not null, padded, or masked")

    if not torch.all(values.lt(1.0), dim=None):
        raise ValueError("values should be less than 1.0")

    if not torch.all(values.ge(0.0), dim=None):
        raise ValueError("values should be greater than or equal to 0.0")


@jaxtyped(typechecker=beartype)
def jitter(inputs: TensorField, jitter: torch.Tensor, is_training: bool) -> torch.Tensor:
    """
    Applies jitter to the input tensor based on the given jitter value and training mode.

    Args:
        inputs (TensorField): The input tensor field.
        jitter (torch.Tensor): The jitter value to be applied.
        is_training (bool): A flag indicating whether the model is in training mode.

    Returns:
        torch.Tensor: The input tensor with jitter applied.

    """
    values = inputs.content
    indicators = inputs.lookup

    dampener = torch.tensor(1 - torch.finfo(torch.half).tiny)

    if is_training:
        jitter = torch.rand_like(values).sub(torch.rand_like(values)).mul(jitter).mul(indicators == Tokens.VAL)
    else:
        jitter = torch.zeros_like(values)

    return values.add(jitter).clamp(min=0.0, max=dampener)
