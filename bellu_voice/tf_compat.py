"""Make Parler-TTS importable on transformers 5.x (Sarvam-30B needs 4.57+/5)."""

from __future__ import annotations

import logging

logger = logging.getLogger("bellu_voice.tf_compat")


def patch_transformers_for_parler() -> None:
    """Parler still imports `isin_mps_friendly`, removed after transformers 5.0."""
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
