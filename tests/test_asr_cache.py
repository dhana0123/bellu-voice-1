import torch

from bellu.perception.asr import _patch_dynamic_cache_layers


def test_dynamic_cache_layers_shim():
    _patch_dynamic_cache_layers()
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    if not hasattr(cache, "key_cache"):
        return  # new transformers already expose .layers on instances

    key = torch.zeros(1, 2, 3, 4)
    val = torch.ones(1, 2, 3, 4)
    cache.key_cache = [key]
    cache.value_cache = [val]

    assert cache.layers[0].keys is key
    assert cache.layers[0].values is val
