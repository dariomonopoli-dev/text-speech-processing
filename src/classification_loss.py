"""Normalize classification loss over all examples in an accumulation step."""

import torch
from transformers.modeling_outputs import SequenceClassifierOutput


def compute_classification_loss(
    outputs: SequenceClassifierOutput,
    labels: torch.Tensor,
    num_items_in_batch: int | torch.Tensor | None = None,
) -> torch.Tensor:
    """Use the full accumulated example count, including uneven final batches."""
    loss = torch.nn.functional.cross_entropy(
        outputs.logits, labels.to(outputs.logits.device), reduction="sum"
    )
    num_examples = labels.numel() if num_items_in_batch is None else num_items_in_batch
    return loss / num_examples
