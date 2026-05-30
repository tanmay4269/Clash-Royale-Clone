from rl.networks.base import BaseActorCritic
from rl.networks.botnet import BotNet
from rl.networks.deep_sets import DeepSetsActorCritic
from rl.networks.transformer import TransformerActorCritic


NETWORK_REGISTRY = {
    'deep_sets': DeepSetsActorCritic,
    'transformer': TransformerActorCritic,
}


import inspect

def make_network(network_type, **kwargs):
    if network_type not in NETWORK_REGISTRY:
        raise ValueError(f"Unknown network type: {network_type}. Options: {list(NETWORK_REGISTRY.keys())}")
    
    net_cls = NETWORK_REGISTRY[network_type]
    sig = inspect.signature(net_cls.__init__)
    valid_params = {
        name for name, param in sig.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
    if not has_kwargs:
        kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
    return net_cls(**kwargs)
