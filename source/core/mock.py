import torch
from tensordict import TensorDict

from source.core.schema import Hyperparameters, SPECIAL_TOKENS
from source.core.tensorclass import DiscreteField, ContinuousField, EntityField

def mock(params: Hyperparameters, **overrides: float|int) -> TensorDict:
    
    params.param.update(**overrides)
    
    data = {}

    N = params.batch_size
    L = params.n_context

    is_padded = torch.arange(L).expand(N, L).lt(torch.randint(1, L, [N]).unsqueeze(-1)).bool()

    for field in params.fields:
        match field.dtype:
            case "continuous":
                indicators = torch.randint(0, len(SPECIAL_TOKENS), (N, L))
                padded = torch.where(is_padded, getattr(SPECIAL_TOKENS, "PAD_"), indicators)
                is_empty = padded.eq(getattr(SPECIAL_TOKENS, "VAL_")).long()
                values = torch.rand(N, L).mul(is_empty)
                data[field.name] = ContinuousField(content=values, lookup=padded, batch_size=[N])

            case "discrete":
                values = torch.randint(0, field.n_levels + len(SPECIAL_TOKENS), (N, L))
                padded = torch.where(is_padded, getattr(SPECIAL_TOKENS, "PAD_"), values)
                data[field.name] = DiscreteField(lookup=padded, batch_size=[N])

            case "entity":
                values = torch.randint(0, L + len(SPECIAL_TOKENS), (N, L))
                padded = torch.where(is_padded, getattr(SPECIAL_TOKENS, "PAD_"), values)
                data[field.name] = EntityField(lookup=padded, batch_size=[N])

    return TensorDict(data, batch_size=N)
