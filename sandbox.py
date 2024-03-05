from source.core.iterops import mock
from source.core.schema import Field, Hyperparameters
from source.core.architecture import SequenceModule

if (__name__ == '__main__'):

    fields: list[Field] = [
        Field("is_online", "discrete", n_levels=2),
        Field("is_foreign", "discrete", n_levels=2),
        Field("merchant_category", "discrete", n_levels=100),
        Field("transaction_type", "discrete", n_levels=100),
    ]
    
    params = Hyperparameters(fields=fields)
    
    module = SequenceModule(
        datapath='.',
        params=params,
        digests={},
    )
    
    obs = mock(params)
    
    print(module.forward(obs))