"""
Hugging Face-style logits processor for grammar-constrained decoding.

This helper is dependency-light and works with either PyTorch tensors or
NumPy arrays at runtime. If passed a PyTorch tensor, it masks in-place and
returns the same tensor, mirroring transformers' LogitsProcessor behavior.

Usage with transformers (example):

    from transformers import LogitsProcessorList
    from stolcke_pcfg.hf_logits import GrammarConstrainedLogitsProcessor

    proc = GrammarConstrainedLogitsProcessor(adapter)
    processors = LogitsProcessorList([proc])
    outputs = model.generate(..., logits_processor=processors)

Note: Terminals must align to tokenizer boundaries. Use the adapter's
next_token_filter to restrict to one-token terminals if needed.
"""

from __future__ import annotations

from typing import Any

from .constrained_adapter import ConstrainedDecoderAdapter


class GrammarConstrainedLogitsProcessor:
    """Mask disallowed token IDs in logits using a grammar adapter.

    - Works with torch.Tensor or NumPy ndarray scores.
    - For torch, performs in-place assignment and returns the same tensor.
    - For NumPy, returns a new array with masked values.
    """

    def __init__(
        self,
        adapter: ConstrainedDecoderAdapter,
        *,
        disallowed_value: float = -1e30,
    ) -> None:
        self.adapter = adapter
        self.disallowed_value = disallowed_value

    def __call__(self, input_ids: Any, scores: Any) -> Any:  # signature mirrors HF processors
        # Determine vocab size from scores' last dimension
        vocab_size = int(scores.shape[-1])
        mask_list = self.adapter.allowed_token_mask(vocab_size)

        # Torch path
        is_torch = False
        try:  # lazy import
            import torch  # noqa: F401

            is_torch = hasattr(scores, "__class__") and scores.__class__.__module__.startswith(
                "torch"
            )
        except Exception:  # torch not available
            is_torch = False

        if is_torch:
            import torch

            mask = torch.tensor(mask_list, dtype=torch.bool, device=scores.device)
            # Broadcast to batch dimension; HF uses shape (batch, vocab)
            if scores.dim() == 1:
                scores[~mask] = self.disallowed_value
            else:
                scores[:, ~mask] = self.disallowed_value
            return scores

        # NumPy path (or other array-like supporting boolean indexing)
        try:
            import numpy as np

            mask = np.asarray(mask_list, dtype=bool)
            masked = scores.copy()
            if masked.ndim == 1:
                masked[~mask] = self.disallowed_value
            else:
                masked[:, ~mask] = self.disallowed_value
            return masked
        except Exception:
            # Fallback: operate on Python lists
            if isinstance(scores, list):
                if scores and isinstance(scores[0], list):
                    return [
                        [
                            v if m else self.disallowed_value
                            for v, m in zip(row, mask_list, strict=False)
                        ]
                        for row in scores
                    ]
                pairs = zip(scores, mask_list, strict=False)
                return [v if m else self.disallowed_value for v, m in pairs]
            raise TypeError(
                "Unsupported scores type for GrammarConstrainedLogitsProcessor"
            ) from None


def make_hf_logits_processor(
    adapter: ConstrainedDecoderAdapter, *, disallowed_value: float = -1e30
) -> GrammarConstrainedLogitsProcessor:
    """Convenience factory returning a GrammarConstrainedLogitsProcessor."""
    return GrammarConstrainedLogitsProcessor(adapter, disallowed_value=disallowed_value)
