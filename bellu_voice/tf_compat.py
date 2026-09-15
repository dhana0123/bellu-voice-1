"""Make Parler-TTS work on transformers 5.x (needed for Sarvam-30B)."""

from __future__ import annotations

import logging

logger = logging.getLogger("bellu_voice.tf_compat")

_PATCHED = False


def patch_transformers_for_parler() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _patch_isin_mps_friendly()
    _patch_config_diff_dict()
    _PATCHED = True


def patch_parler_class(cls) -> None:
    """Transformers 5 calls tie_weights(recompute_mapping=False); Parler does not accept that."""
    orig = cls.tie_weights

    def tie_weights(self, *args, **kwargs):
        kwargs.pop("recompute_mapping", None)
        try:
            return orig(self, *args, **kwargs)
        except TypeError:
            return orig(self)

    cls.tie_weights = tie_weights
    logger.info("patched %s.tie_weights for transformers 5", cls.__name__)


def _patch_isin_mps_friendly() -> None:
    import torch
    import transformers.pytorch_utils as pu

    if getattr(pu, "isin_mps_friendly", None) is not None:
        return

    def isin_mps_friendly(elements: torch.Tensor, test_elements) -> torch.Tensor:
        if not isinstance(test_elements, torch.Tensor):
            test_elements = torch.tensor(test_elements, device=elements.device)
        return torch.isin(elements, test_elements)

    pu.isin_mps_friendly = isin_mps_friendly
    logger.info("patched transformers.pytorch_utils.isin_mps_friendly for Parler")


def _patch_config_diff_dict() -> None:
    """Transformers 5 logs configs via empty ``Config()``, which Parler forbids."""
    from transformers.configuration_utils import PretrainedConfig

    orig = PretrainedConfig.to_diff_dict

    def to_diff_dict(self):
        try:
            return orig(self)
        except (ValueError, TypeError):
            return self.to_dict() if hasattr(self, "to_dict") else {}

    PretrainedConfig.to_diff_dict = to_diff_dict
    logger.info("patched PretrainedConfig.to_diff_dict for Parler")
