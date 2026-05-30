from rl.networks.base import BaseActorCritic
from rl.networks.botnet import BotNet
from rl.networks.deep_sets import DeepSetsActorCritic
from rl.networks.transformer import TransformerActorCritic


NETWORK_REGISTRY = {
    'deep_sets': DeepSetsActorCritic,
    'transformer': TransformerActorCritic,
}


def make_network(network_type, **kwargs):
    if network_type not in NETWORK_REGISTRY:
        raise ValueError(f"Unknown network type: {network_type}. Options: {list(NETWORK_REGISTRY.keys())}")
    return NETWORK_REGISTRY[network_type](**kwargs)
