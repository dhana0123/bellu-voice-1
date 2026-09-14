from bellu.perception.asr import _patch_dynamic_cache_layers


def test_dynamic_cache_layers_is_assignable():
    _patch_dynamic_cache_layers()
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    cache.layers = [object()]
    assert len(cache.layers) == 1


def test_dynamic_cache_layers_from_key_cache():
    _patch_dynamic_cache_layers()
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    if "key_cache" not in cache.__dict__ and not hasattr(cache, "key_cache"):
        return

    import torch

    key = torch.zeros(1, 2, 3, 4)
    val = torch.ones(1, 2, 3, 4)
    cache.__dict__["key_cache"] = [key]
    cache.__dict__["value_cache"] = [val]
    cache.__dict__.pop("layers", None)

    assert cache.layers[0].keys is key
    assert cache.layers[0].values is val
