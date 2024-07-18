from functools import reduce

from beartype import beartype
import torch
from jaxtyping import Float, Int, jaxtyped

from windmark.core.constructs.general import Tokens


@jaxtyped(typechecker=beartype)
def smoothen(
    targets: Int[torch.Tensor, "N *L"],  # noqa: F821
    size: int,
    sigma: float,
) -> Float[torch.Tensor, "..."]:  # noqa: F821
    """Apply gaussian smoothing to continuous targets with fixed offset for special tokens

    Arguments:
        targets (Int[torch.Tensor, "N *L"]): Target label indices.
        size (int): The number of quantiles to smoothen over.
        sigma (float): Gaussian smoothing factor

    Returns:
        Float[torch.Tensor]: Smoothened quantile targets.
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
