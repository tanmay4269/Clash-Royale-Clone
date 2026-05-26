from rl.networks.base import BaseActorCritic
from rl.networks.botnet import BotNet
from rl.networks.deep_sets_baseline import DeepSetsBaseline
from rl.networks.deep_sets import DeepSetsActorCritic
from rl.networks.pointer import PointerActorCritic
from rl.networks.attention import AttentionActorCritic
from rl.networks.transformer import TransformerActorCritic
from rl.networks.autoregressive import AutoregressiveActorCritic


NETWORK_REGISTRY = {
    'deep_sets_baseline': DeepSetsBaseline,
    'deep_sets': DeepSetsActorCritic,
    'pointer': PointerActorCritic,
    'attention': AttentionActorCritic,
    'transformer': TransformerActorCritic,
    'autoregressive': AutoregressiveActorCritic,
}


def make_network(network_type, **kwargs):
    if network_type not in NETWORK_REGISTRY:
        raise ValueError(f"Unknown network type: {network_type}. Options: {list(NETWORK_REGISTRY.keys())}")
    return NETWORK_REGISTRY[network_type](**kwargs)
