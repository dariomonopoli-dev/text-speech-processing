from pathlib import Path

import pandas as pd
import pytest
import torch
from transformers import XmodConfig, XmodForSequenceClassification

from predict_swissbert import load_eval_set, predict_batches


def test_eval_set_preserves_ids_text_and_order(tmp_path: Path) -> None:
    path_input = tmp_path / "input.csv"
    pd.DataFrame(
        {"sentence_id": ["002", "001"], "text": [" sali ", "hoi"], "label": [1, 0]}
    ).to_csv(path_input, index=False)

    assert load_eval_set(path_input).to_dict("list") == {
        "sentence_id": ["002", "001"],
        "text": [" sali ", "hoi"],
    }


@pytest.mark.parametrize(
    "rows, message",
    [
        ({"sentence_id": [1, 1], "text": ["a", "b"]}, "duplicate"),
        ({"sentence_id": [1, None], "text": ["a", "b"]}, "missing"),
        ({"sentence_id": [1], "text": [" "]}, "blank"),
        ({"sentence_id": [], "text": []}, "no sentences"),
    ],
)
def test_eval_set_rejects_invalid_rows(
    tmp_path: Path, rows: dict, message: str
) -> None:
    path_input = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(path_input, index=False)

    with pytest.raises(ValueError, match=message):
        load_eval_set(path_input)


def test_batched_xmod_inference_uses_gsw_and_preserves_weights() -> None:
    model = XmodForSequenceClassification(
        XmodConfig(
            vocab_size=20,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            languages=["de_CH", "gsw"],
            default_language="de_CH",
            num_labels=2,
        )
    )
    model.set_default_language("gsw")
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    adapter_batch_sizes = []
    handles = [
        module.register_forward_hook(
            lambda module, inputs, output: adapter_batch_sizes.append(inputs[0].shape[0])
        )
        for name, module in model.named_modules()
        if name.endswith("adapter_modules.gsw")
    ]

    def tokenize(texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[0, 5 + len(text), 2] for text in texts])}

    texts = ["a", "bb", "ccc"]
    batched = predict_batches(model, tokenize, texts, 2, 128, torch.device("cpu"))
    single = predict_batches(model, tokenize, texts, 1, 128, torch.device("cpu"))
    assert sum(adapter_batch_sizes) == 2 * len(texts)

    adapter_batch_sizes.clear()
    model.set_default_language("de_CH")
    predict_batches(model, tokenize, texts, 2, 128, torch.device("cpu"))
    assert sum(adapter_batch_sizes) == 0
    for handle in handles:
        handle.remove()

    assert batched.shape == (3, 2)
    assert torch.allclose(batched.sum(dim=-1), torch.ones(3))
    assert torch.allclose(batched, single, atol=1e-6)
    assert all(
        torch.equal(weights[key], value) for key, value in model.state_dict().items()
    )
