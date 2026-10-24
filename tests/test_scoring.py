import pandas as pd
import pytest

from scoring import add_hate_scores, compute_hate_scores


def fake_classifier(texts, **kwargs):
    return [
        [{"label": "LABEL_0", "score": 0.25}, {"label": "LABEL_1", "score": 0.75}]
        for _ in texts
    ]


def test_hate_scores_are_the_probability_of_the_hate_label():
    scores = compute_hate_scores(["eins", "zwei"], classifier=fake_classifier)

    assert scores.tolist() == [0.75, 0.75]
    assert scores.name == "hate_score"


def test_hate_scores_reject_output_without_the_hate_label():
    def classifier(texts, **kwargs):
        return [[{"label": "LABEL_0", "score": 1.0}] for _ in texts]

    with pytest.raises(ValueError, match="hate label exactly once"):
        compute_hate_scores(["eins"], classifier=classifier)


def test_candidate_scores_are_added_in_batches():
    batch_sizes = []

    def classifier(texts, **kwargs):
        batch_sizes.append(len(texts))
        return fake_classifier(texts)

    candidates = pd.DataFrame(
        {"sentence_id": ["1", "2", "3"], "text": ["eins", "zwei", "drü"]}
    )

    scored = add_hate_scores(candidates, classifier, batch_size=2)

    assert scored["hate_score"].tolist() == [0.75, 0.75, 0.75]
    assert batch_sizes == [2, 1]
    assert "hate_score" not in candidates


def test_candidate_scoring_rejects_blank_text():
    candidates = pd.DataFrame({"sentence_id": ["1"], "text": [" "]})

    with pytest.raises(ValueError, match="non-empty string"):
        add_hate_scores(candidates, fake_classifier, batch_size=2)
