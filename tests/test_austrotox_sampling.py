import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from austrotox_sampling import (  # noqa: E402
    FIXED_STRATUM_TARGETS,
    assign_austrotox_strata,
    audit_translations,
    sample_fixed_strata,
    score_with_gahd,
)


class AustroToxSamplingTests(unittest.TestCase):
    def test_assigns_non_overlapping_paper_boundaries(self):
        scores = pd.Series([0.0, 0.2999, 0.3, 0.5, 0.7, 0.9, 1.0])

        strata = assign_austrotox_strata(scores)

        self.assertEqual(
            strata.tolist(),
            [
                "0.0-0.3",
                "0.0-0.3",
                "0.3-0.5",
                "0.5-0.7",
                "0.7-0.9",
                "0.9-1.0",
                "0.9-1.0",
            ],
        )

    def test_rejects_scores_outside_probability_range(self):
        for scores in (
            pd.Series([-0.01]),
            pd.Series([1.01]),
            pd.Series([float("nan")]),
        ):
            with self.subTest(scores=scores.tolist()):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    assign_austrotox_strata(scores)

    def test_samples_fixed_scaled_austrotox_allocation_deterministically(self):
        self.assertEqual(
            FIXED_STRATUM_TARGETS,
            {
                "0.0-0.3": 134,
                "0.3-0.5": 134,
                "0.5-0.7": 134,
                "0.7-0.9": 164,
                "0.9-1.0": 134,
            },
        )
        rows = []
        for stratum, score in (
            ("0.0-0.3", 0.1),
            ("0.3-0.5", 0.4),
            ("0.5-0.7", 0.6),
            ("0.7-0.9", 0.8),
            ("0.9-1.0", 0.95),
        ):
            rows.extend(
                {
                    "sentence_id": f"{stratum}-{i}",
                    "text": f"Juden Frauen {stratum}-{i}",
                    "hate_score": score,
                }
                for i in range(200)
            )
        corpus = pd.DataFrame(rows)

        sample = sample_fixed_strata(corpus, random_state=42)
        repeated_sample = sample_fixed_strata(corpus, random_state=42)

        self.assertEqual(len(sample), 700)
        self.assertEqual(sample["sentence_id"].nunique(), 700)
        pd.testing.assert_frame_equal(sample, repeated_sample)
        self.assertEqual(
            sample["austrotox_stratum"].value_counts(sort=False).to_dict(),
            FIXED_STRATUM_TARGETS,
        )
        self.assertTrue(sample["text"].str.contains("Juden Frauen", regex=False).all())

    def test_reports_every_undersized_stratum_without_rebalancing(self):
        corpus = pd.DataFrame(
            [
                {
                    "sentence_id": f"row-{index}",
                    "text": f"text-{index}",
                    "hate_score": score,
                }
                for index, score in enumerate([0.1, 0.4, 0.6, 0.8, 0.95])
            ]
        )

        with self.assertRaises(ValueError) as context:
            sample_fixed_strata(corpus, random_state=42)

        message = str(context.exception)
        for stratum, target in FIXED_STRATUM_TARGETS.items():
            self.assertIn(f"{stratum}: available=1, required={target}", message)

    def test_rejects_duplicate_sentence_ids(self):
        corpus = pd.DataFrame(
            {
                "sentence_id": ["duplicate", "duplicate"],
                "text": ["first", "second"],
                "hate_score": [0.1, 0.4],
            }
        )

        with self.assertRaisesRegex(ValueError, "sentence_id values must be unique"):
            sample_fixed_strata(corpus)

    def test_extracts_hate_probability_by_label_not_output_order(self):
        class DeterministicClassifier:
            def __call__(self, texts, **kwargs):
                return [
                    [
                        {"label": "HATE", "score": 0.75},
                        {"label": "NON_HATE", "score": 0.25},
                    ],
                    [
                        {"label": "NON_HATE", "score": 0.9},
                        {"label": "HATE", "score": 0.1},
                    ],
                ]

        scores = score_with_gahd(
            ["first", "second"], classifier=DeterministicClassifier()
        )

        self.assertEqual(scores.tolist(), [0.75, 0.1])

    def test_flags_translation_failures_without_filtering_identity_terms(self):
        source = [
            "D Jude und d Fraue gönd hei.",
            "Das isch en normale Satz.",
            "No en normale Satz.",
            "Churz.",
        ]
        translations = [
            "Die Frauen gehen nach Hause.",
            "Hinweis zur Übersetzung: Das ist ein normaler Satz.",
            None,
            "Dies ist eine unangemessen lange Übersetzung mit vielen zusätzlichen "
            "Wörtern, die im Quelltext nicht vorkommen.",
        ]

        audit = audit_translations(source, translations)

        self.assertEqual(
            audit[
                ["missing", "meta_commentary", "extreme_length", "has_anomaly"]
            ].to_dict("records"),
            [
                {
                    "missing": False,
                    "meta_commentary": False,
                    "extreme_length": False,
                    "has_anomaly": False,
                },
                {
                    "missing": False,
                    "meta_commentary": True,
                    "extreme_length": False,
                    "has_anomaly": True,
                },
                {
                    "missing": True,
                    "meta_commentary": False,
                    "extreme_length": False,
                    "has_anomaly": True,
                },
                {
                    "missing": False,
                    "meta_commentary": False,
                    "extreme_length": True,
                    "has_anomaly": True,
                },
            ],
        )

    def test_empty_translation_audit_keeps_its_schema(self):
        audit = audit_translations([], [])

        self.assertEqual(
            audit.columns.tolist(),
            [
                "missing",
                "meta_commentary",
                "multiple_paragraphs",
                "length_ratio",
                "extreme_length",
                "has_anomaly",
            ],
        )

    def test_rejects_classifier_output_that_cannot_align_with_input(self):
        class TooShortClassifier:
            def __call__(self, texts, **kwargs):
                return [[{"label": "HATE", "score": 0.75}]]

        with self.assertRaisesRegex(ValueError, "one prediction per input text"):
            score_with_gahd(["first", "second"], classifier=TooShortClassifier())

    def test_rejects_invalid_hate_probability(self):
        class InvalidScoreClassifier:
            def __call__(self, texts, **kwargs):
                return [[{"label": "HATE", "score": 1.5}]]

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            score_with_gahd(["first"], classifier=InvalidScoreClassifier())

    def test_rejects_malformed_classifier_records(self):
        cases = [
            ([{"label": "HATE", "score": 0.5}], "sequence of label-score records"),
            ([[]], "GAHD hate label exactly once"),
            (
                [[{"label": "HATE", "score": 0.5}, {"label": "HATE", "score": 0.5}]],
                "GAHD hate label exactly once",
            ),
            ([[{"label": "HATE", "score": "invalid"}]], "numeric score"),
            ([[{"label": "HATE", "score": float("nan")}]], "between 0 and 1"),
            ([[{"label": "HATE", "score": float("inf")}]], "between 0 and 1"),
        ]

        for predictions, message in cases:

            class MalformedClassifier:
                def __call__(self, texts, **kwargs):
                    return predictions

            with self.subTest(predictions=predictions):
                with self.assertRaisesRegex(ValueError, message):
                    score_with_gahd(["first"], classifier=MalformedClassifier())

    def test_extracts_gahd_hate_probability_from_label_1(self):
        class GahdClassifier:
            def __call__(self, texts, **kwargs):
                return [
                    [
                        {"label": "LABEL_0", "score": 0.2},
                        {"label": "LABEL_1", "score": 0.8},
                    ]
                ]

        scores = score_with_gahd(["example"], classifier=GahdClassifier())

        self.assertEqual(scores.tolist(), [0.8])


if __name__ == "__main__":
    unittest.main()
