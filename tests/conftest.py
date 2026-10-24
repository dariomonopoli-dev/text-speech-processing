import numpy as np
import pandas as pd
import pytest

STRATUM_BOUNDS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
NUM_PER_STRATUM = 200


@pytest.fixture
def scored_candidates() -> pd.DataFrame:
    scores = np.concatenate(
        [
            np.linspace(low, high, NUM_PER_STRATUM, endpoint=False)
            for low, high in STRATUM_BOUNDS
        ]
    )
    sentence_ids = [str(100000 + i) for i in range(len(scores))]
    return pd.DataFrame(
        {
            "text": [f"satz nummer {i} isch e chli länger" for i in range(len(scores))],
            "sentence_id": pd.array(sentence_ids, dtype="string"),
            "hate_score": scores,
        }
    )
