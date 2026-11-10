import copy
from pathlib import Path

import pytest
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    XmodConfig,
    XmodForSequenceClassification,
)

from classification_loss import compute_classification_loss


@pytest.mark.parametrize("model_type", ["bert", "xmod"])
@pytest.mark.parametrize("batch_sizes", [(2, 2, 2, 2), (2, 2, 1)])
def test_accumulated_gradients_match_a_full_batch(
    tmp_path: Path, model_type: str, batch_sizes: tuple[int, ...]
) -> None:
    torch.manual_seed(42)
    config = dict(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout=0.0,
        num_labels=2,
    )
    if model_type == "bert":
        model = BertForSequenceClassification(BertConfig(**config))
    else:
        model = XmodForSequenceClassification(
            XmodConfig(**config, languages=["de_CH", "gsw"], default_language="de_CH")
        )
    reference = copy.deepcopy(model)
    num_examples = sum(batch_sizes)
    inputs = {
        "input_ids": torch.tensor([[0, 5 + i, 2] for i in range(num_examples)]),
        "attention_mask": torch.ones(num_examples, 3, dtype=torch.long),
        "labels": torch.tensor([i % 2 for i in range(num_examples)]),
    }
    reference_loss = reference(**inputs).loss
    reference_loss.backward()
    expected = torch.cat(
        [
            parameter.grad.flatten()
            for parameter in reference.parameters()
            if parameter.grad is not None
        ]
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            gradient_accumulation_steps=len(batch_sizes),
            report_to="none",
        ),
        compute_loss_func=compute_classification_loss,
    )
    batches = []
    start = 0
    for size in batch_sizes:
        batches.append(
            {key: value[start : start + size] for key, value in inputs.items()}
        )
        start += size
    trainer.current_gradient_accumulation_steps = len(batches)
    count = trainer._get_num_items_in_batch(batches, torch.device("cpu"))
    losses = [trainer.training_step(model, batch, count).item() for batch in batches]
    actual = torch.cat(
        [
            parameter.grad.flatten()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
    )

    assert sum(losses) == pytest.approx(reference_loss.item(), abs=1e-6)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=2e-5)
