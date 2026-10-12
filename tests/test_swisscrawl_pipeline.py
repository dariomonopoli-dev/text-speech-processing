import json
import inspect
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import checkpoint_store  # noqa: E402
import artifact_io  # noqa: E402
from artifact_io import atomic_write_csv, atomic_write_json  # noqa: E402
from checkpoint_store import SQLiteCheckpointStore  # noqa: E402
from swisscrawl_pipeline import (  # noqa: E402
    build_annotation_frames,
    build_gahd_classifier,
    build_batch_translation_prompt,
    corpus_fingerprint,
    restore_batch_translation,
    score_candidate_pool,
    select_master_sample,
    translate_selected_batch,
    verify_artifacts,
)
import build_swisscrawl_testset as cli  # noqa: E402


class BatchTranslationTests(unittest.TestCase):
    class FakeMessages:
        def __init__(self, outcomes):
            self.outcomes = list(outcomes)
            self.calls = []

        class Stream:
            def __init__(self, outcome):
                self.outcome = outcome

            def __enter__(self):
                if isinstance(self.outcome, BaseException):
                    raise self.outcome
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def get_final_message(self):
                return self.outcome

        def stream(self, **kwargs):
            self.calls.append(kwargs)
            outcome = self.outcomes.pop(0)
            return self.Stream(outcome)

    @staticmethod
    def response(text, input_tokens=100, output_tokens=50, stop_reason="end_turn"):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason=stop_reason,
            usage=SimpleNamespace(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

    def test_prompt_is_the_approved_few_shot_prompt_followed_by_all_rows(self):
        prompt = build_batch_translation_prompt(["Grüezi", "Du bisch en Idiot."])

        self.assertTrue(
            prompt.startswith(
                "Übersetze die folgenden Schweizerdeutschen Sätze ins Hochdeutsche. "
                "Behalte Bedeutung, Ton und alle Beleidigungen oder vulgären "
                "Ausdrücke wörtlich bei, schwäche nichts ab."
            )
        )
        self.assertIn(
            "Schweizerdeutsch: De Sportverein en Nüereschdoorf bietet", prompt
        )
        self.assertIn(
            "Schweizerdeutsch: In Deitige isch es üblich, die monatliche Fixköste "
            'für de "V-Pass bequem per Duuruftrag z regle.',
            prompt,
        )
        self.assertTrue(prompt.endswith("Sätze:\n\n1. Grüezi\n2. Du bisch en Idiot."))

    def test_one_api_request_translates_the_complete_batch_in_order(self):
        messages = self.FakeMessages(
            [self.response("1. Guten Tag\n2. Du bist ein Idiot.")]
        )
        client = SimpleNamespace(messages=messages)
        frame = pd.DataFrame(
            {"sentence_id": ["a", "b"], "text": ["Grüezi", "Du bisch en Idiot."]}
        )

        translated, report = translate_selected_batch(frame, client)

        self.assertEqual(len(messages.calls), 1)
        call = messages.calls[0]
        self.assertEqual(call["model"], "claude-haiku-5-5")
        self.assertEqual(call["max_tokens"], 128_000)
        self.assertEqual(call["output_config"], {"effort": "medium"})
        self.assertEqual(
            translated["translation"].tolist(),
            ["Guten Tag", "Du bist ein Idiot."],
        )
        self.assertEqual(report["row_count"], 2)
        self.assertEqual(report["input_tokens"], 100)
        self.assertEqual(report["output_tokens"], 50)

        restored = restore_batch_translation(
            frame.assign(translation="obsolete"), report
        )
        self.assertEqual(
            restored["translation"].tolist(),
            translated["translation"].tolist(),
        )

        changed = frame.copy()
        changed.loc[0, "text"] = "Anders"
        with self.assertRaisesRegex(ValueError, "checkpoint does not match"):
            restore_batch_translation(changed, report)

    def test_rejects_incomplete_or_out_of_order_numbered_output(self):
        for output in (
            "1. Guten Tag",
            "01. Guten Tag\n2. Hallo zusammen",
            "2. Hallo zusammen\n1. Guten Tag",
            "Hier sind die Übersetzungen:\n1. Guten Tag\n2. Hallo zusammen",
        ):
            with self.subTest(output=output):
                messages = self.FakeMessages([self.response(output)])
                frame = pd.DataFrame(
                    {"sentence_id": ["a", "b"], "text": ["Grüezi", "Hoi zäme"]}
                )
                with self.assertRaisesRegex(ValueError, "numbered translations"):
                    translate_selected_batch(frame, SimpleNamespace(messages=messages))

    def test_rejects_a_structurally_complete_truncated_response(self):
        messages = self.FakeMessages(
            [
                self.response(
                    "1. Guten Tag\n2. Hallo zusam",
                    stop_reason="max_tokens",
                )
            ]
        )
        frame = pd.DataFrame(
            {"sentence_id": ["a", "b"], "text": ["Grüezi", "Hoi zäme"]}
        )

        with self.assertRaisesRegex(ValueError, "stop_reason"):
            translate_selected_batch(frame, SimpleNamespace(messages=messages))

    def test_checkpoints_invalid_paid_response_and_accumulates_on_resume(self):
        frame = pd.DataFrame(
            {"sentence_id": ["a", "b"], "text": ["Grüezi", "Hoi zäme"]}
        )
        checkpoints = []
        invalid_messages = self.FakeMessages(
            [self.response("1. Guten Tag", input_tokens=100, output_tokens=20)]
        )

        with self.assertRaisesRegex(ValueError, "numbered translations"):
            translate_selected_batch(
                frame,
                SimpleNamespace(messages=invalid_messages),
                checkpoint_callback=checkpoints.append,
            )

        self.assertEqual(checkpoints[-1]["status"], "invalid")
        self.assertEqual(checkpoints[-1]["raw_response"], "1. Guten Tag")
        valid_messages = self.FakeMessages(
            [
                self.response(
                    "1. Guten Tag\n2. Hallo zusammen",
                    input_tokens=100,
                    output_tokens=30,
                )
            ]
        )
        _, resumed_report = translate_selected_batch(
            frame,
            SimpleNamespace(messages=valid_messages),
            prior_report=checkpoints[-1],
            checkpoint_callback=checkpoints.append,
        )

        self.assertEqual(resumed_report["status"], "success")
        self.assertEqual(resumed_report["attempts"], 2)
        self.assertEqual(resumed_report["input_tokens"], 200)
        self.assertEqual(resumed_report["output_tokens"], 50)
        self.assertEqual(len(resumed_report["attempt_history"]), 2)

    def test_rejects_unvalidated_or_successful_prior_report(self):
        frame = pd.DataFrame(
            {"sentence_id": ["a", "b"], "text": ["Grüezi", "Hoi zäme"]}
        )
        messages = self.FakeMessages([])
        other_frame = frame.assign(text=["Anders", "Hoi zäme"])
        other_messages = self.FakeMessages(
            [self.response("1. Anders\n2. Hallo zusammen")]
        )
        _, other_report = translate_selected_batch(
            other_frame, SimpleNamespace(messages=other_messages)
        )

        with self.assertRaisesRegex(ValueError, "checkpoint does not match"):
            translate_selected_batch(
                frame,
                SimpleNamespace(messages=messages),
                prior_report=other_report,
            )

        valid_messages = self.FakeMessages(
            [self.response("1. Guten Tag\n2. Hallo zusammen")]
        )
        _, successful = translate_selected_batch(
            frame, SimpleNamespace(messages=valid_messages)
        )
        with self.assertRaisesRegex(ValueError, "already successful"):
            translate_selected_batch(
                frame,
                SimpleNamespace(messages=messages),
                prior_report=successful,
            )


class CorpusFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_stable_and_sensitive_to_content_and_order(self):
        frame = pd.DataFrame(
            {
                "sentence_id": ["a", "b"],
                "text": ["Grüezi", "Hoi"],
                "source": ["swisscrawl", "swisscrawl"],
            }
        )

        fingerprint = corpus_fingerprint(frame)

        self.assertEqual(fingerprint, corpus_fingerprint(frame.copy()))
        self.assertNotEqual(
            fingerprint,
            corpus_fingerprint(frame.assign(text=["Anders", "Hoi"])),
        )
        self.assertNotEqual(
            fingerprint,
            corpus_fingerprint(frame.iloc[::-1].reset_index(drop=True)),
        )


class SQLiteCheckpointStoreTests(unittest.TestCase):
    def test_checkpoint_store_is_an_abstract_contract(self):
        contract = getattr(checkpoint_store, "CheckpointStore", None)

        self.assertIsNotNone(contract)
        self.assertTrue(inspect.isabstract(contract))

    def test_sqlite_checkpoint_store_implements_contract(self):
        store_type = getattr(checkpoint_store, "SQLiteCheckpointStore", None)

        self.assertIsNotNone(store_type)
        self.assertTrue(issubclass(store_type, checkpoint_store.CheckpointStore))
        self.assertFalse(inspect.isabstract(store_type))

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "checkpoint.sqlite3"
        self.configuration = {
            "score_model": "gahd",
            "score_revision": "pinned-revision",
        }

    def open_store(self, corpus_hash="corpus-a", configuration=None):
        store = SQLiteCheckpointStore(
            self.path,
            corpus_hash=corpus_hash,
            configuration=configuration or self.configuration,
        )
        self.addCleanup(store.close)
        return store

    def test_committed_rows_resume_and_upsert_idempotently(self):
        store = self.open_store()
        store.save_scores(
            [
                {"sentence_id": "a", "hate_score": 0.1},
                {"sentence_id": "b", "hate_score": 0.8},
            ]
        )
        store.save_scores([{"sentence_id": "a", "hate_score": 0.2}])
        store.close()

        resumed = self.open_store()

        self.assertEqual(resumed.completed_score_ids(), {"a", "b"})
        self.assertEqual(resumed.load_scores(), {"a": 0.2, "b": 0.8})

    def test_score_batch_rolls_back_if_any_row_is_invalid(self):
        store = self.open_store()

        with self.assertRaises(sqlite3.IntegrityError):
            store.save_scores(
                [
                    {"sentence_id": "valid", "hate_score": 0.5},
                    {"sentence_id": "invalid", "hate_score": 1.5},
                ]
            )

        self.assertEqual(store.completed_score_ids(), set())

    def test_rejects_checkpoint_from_a_different_corpus(self):
        self.open_store().close()

        with self.assertRaisesRegex(ValueError, "corpus fingerprint"):
            SQLiteCheckpointStore(
                self.path,
                corpus_hash="corpus-b",
                configuration=self.configuration,
            )

    def test_rejects_checkpoint_from_a_different_model_revision(self):
        self.open_store().close()
        changed = dict(self.configuration, score_revision="other-revision")

        with self.assertRaisesRegex(ValueError, "configuration"):
            SQLiteCheckpointStore(
                self.path,
                corpus_hash="corpus-a",
                configuration=changed,
            )

    def test_migrates_matching_legacy_configuration_to_scoring_keys(self):
        legacy = dict(self.configuration, translation_model="legacy-haiku")
        self.open_store(configuration=legacy).close()

        resumed = self.open_store(configuration=self.configuration)

        self.assertEqual(resumed.completed_score_ids(), set())
        with sqlite3.connect(self.path) as connection:
            stored = dict(connection.execute("SELECT key, value FROM metadata"))
        self.assertEqual(json.loads(stored["configuration"]), self.configuration)


class AtomicExportTests(unittest.TestCase):
    def test_xlsx_export_preserves_formula_prefixes_as_text(self):
        writer = getattr(artifact_io, "atomic_write_xlsx", None)
        self.assertIsNotNone(writer)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "annotations.xlsx"
            values = ["=cmd()", "+SUM(1,1)", "-danger", "@name"]
            writer(pd.DataFrame({"text": values}), target)

            from openpyxl import load_workbook

            sheet = load_workbook(target, data_only=False).active
            cells = [sheet.cell(row=index, column=1) for index in range(2, 6)]
            self.assertEqual([cell.value for cell in cells], values)
            self.assertTrue(all(cell.data_type == "s" for cell in cells))

    def test_workbook_verification_rejects_formula_cells(self):
        verify = getattr(artifact_io, "verify_annotation_workbooks", None)
        self.assertIsNotNone(verify)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dario.xlsx"
            frame = pd.DataFrame(
                {
                    "sentence_id": ["1"],
                    "text": ["Hoi"],
                    "label": [""],
                    "comment": [""],
                }
            )
            artifact_io.atomic_write_xlsx(frame, target)
            verify({"dario": frame}, {"dario": target})

            from openpyxl import load_workbook

            workbook = load_workbook(target)
            workbook.active["B2"] = "=cmd()"
            workbook.save(target)
            with self.assertRaisesRegex(ValueError, "formula"):
                verify({"dario": frame}, {"dario": target})

    def test_atomic_csv_is_utf8_bom_encoded_for_spreadsheet_apps(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "annotations.csv"

            atomic_write_csv(pd.DataFrame({"text": ["dänk"]}), target)

            self.assertTrue(target.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_atomic_csv_and_json_replace_existing_files_without_temp_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "artifact.csv"
            json_path = root / "artifact.json"
            csv_path.write_text("stale\n", encoding="utf-8")
            json_path.write_text('{"stale": true}\n', encoding="utf-8")

            atomic_write_csv(pd.DataFrame({"value": ["Grüezi"]}), csv_path)
            atomic_write_json({"status": "fresh", "count": 1}, json_path)

            pd.testing.assert_frame_equal(
                pd.read_csv(csv_path), pd.DataFrame({"value": ["Grüezi"]})
            )
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8")),
                {"status": "fresh", "count": 1},
            )
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["artifact.csv", "artifact.json"],
            )


class CandidateScoringTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.store = SQLiteCheckpointStore(
            Path(self.temporary_directory.name) / "checkpoint.sqlite3",
            corpus_hash="scoring-corpus",
            configuration={"score_revision": "pinned-revision"},
        )
        self.addCleanup(self.store.close)

    def test_scores_missing_rows_in_batches_and_preserves_source_order(self):
        frame = pd.DataFrame(
            {
                "sentence_id": ["a", "b", "c", "d", "e"],
                "text": ["alpha", "bravo", "charlie", "delta", "echo"],
                "source": ["swisscrawl"] * 5,
            }
        )
        self.store.save_scores([{"sentence_id": "b", "hate_score": 0.22}])
        score_by_text = {
            "alpha": 0.11,
            "charlie": 0.33,
            "delta": 0.44,
            "echo": 0.55,
        }

        class RecordingClassifier:
            def __init__(self):
                self.calls = []

            def __call__(self, texts, **kwargs):
                self.calls.append(list(texts))
                return [
                    [
                        {"label": "NON_HATE", "score": 1 - score_by_text[text]},
                        {"label": "HATE", "score": score_by_text[text]},
                    ]
                    for text in texts
                ]

        classifier = RecordingClassifier()

        scored = score_candidate_pool(
            frame,
            store=self.store,
            classifier=classifier,
            batch_size=2,
        )

        self.assertEqual(classifier.calls, [["alpha", "charlie"], ["delta", "echo"]])
        self.assertEqual(scored["sentence_id"].tolist(), frame["sentence_id"].tolist())
        self.assertEqual(scored["source"].tolist(), frame["source"].tolist())
        self.assertEqual(scored["hate_score"].tolist(), [0.11, 0.22, 0.33, 0.44, 0.55])
        self.assertEqual(
            self.store.completed_score_ids(),
            {"a", "b", "c", "d", "e"},
        )

    def test_resume_with_every_row_complete_makes_no_classifier_calls(self):
        frame = pd.DataFrame({"sentence_id": ["a"], "text": ["alpha"]})
        self.store.save_scores([{"sentence_id": "a", "hate_score": 0.4}])

        class FailingClassifier:
            def __call__(self, texts, **kwargs):
                raise AssertionError("completed rows must not be scored again")

        scored = score_candidate_pool(frame, self.store, FailingClassifier(), 8)

        self.assertEqual(scored["hate_score"].tolist(), [0.4])

    def test_rejects_duplicate_ids_and_empty_text_before_classification(self):
        cases = [
            (
                pd.DataFrame({"sentence_id": ["a", "a"], "text": ["first", "second"]}),
                "sentence_id values must be unique",
            ),
            (
                pd.DataFrame({"sentence_id": ["a"], "text": ["  "]}),
                "non-empty string",
            ),
        ]

        for frame, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    score_candidate_pool(frame, self.store, object(), 2)

    def test_rejects_classifier_cardinality_mismatch_without_saving_batch(self):
        frame = pd.DataFrame({"sentence_id": ["a", "b"], "text": ["alpha", "bravo"]})

        class TooShortClassifier:
            def __call__(self, texts, **kwargs):
                return [[{"label": "HATE", "score": 0.5}]]

        with self.assertRaisesRegex(ValueError, "one prediction per input text"):
            score_candidate_pool(frame, self.store, TooShortClassifier(), 2)

        self.assertEqual(self.store.completed_score_ids(), set())

    def test_builds_classifier_with_pinned_revision_and_explicit_device(self):
        sentinel = object()

        with patch(
            "swisscrawl_pipeline._create_transformers_pipeline",
            return_value=sentinel,
        ) as transformer_pipeline:
            classifier = build_gahd_classifier(device="cpu")

        self.assertIs(classifier, sentinel)
        transformer_pipeline.assert_called_once_with(
            model="jagoldz/gahd",
            revision="90c6e4f50d109c42bebdeb0f43fb5c3e619612fc",
            device="cpu",
        )


class MasterSampleTests(unittest.TestCase):
    @staticmethod
    def scored_fixture(rows_per_stratum=200):
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
                    "sentence_id": f"{stratum}-{index}",
                    "text": f"text {stratum} {index}",
                    "source_url": f"https://example.test/{stratum}/{index}",
                    "hate_score": score,
                }
                for index in range(rows_per_stratum)
            )
        return pd.DataFrame(rows)

    def test_selects_exact_sample_and_builds_reproducible_diagnostic_report(self):
        scored = self.scored_fixture()

        sample, report = select_master_sample(scored, random_state=42)
        repeated, repeated_report = select_master_sample(scored, random_state=42)

        pd.testing.assert_frame_equal(sample, repeated)
        self.assertEqual(report, repeated_report)
        self.assertEqual(len(sample), 700)
        self.assertEqual(sample["sentence_id"].nunique(), 700)
        self.assertEqual(
            sample.columns.tolist(),
            [
                "sentence_id",
                "text",
                "source_url",
                "hate_score",
                "austrotox_stratum",
            ],
        )
        expected_targets = {
            "0.0-0.3": 134,
            "0.3-0.5": 134,
            "0.5-0.7": 134,
            "0.7-0.9": 164,
            "0.9-1.0": 134,
        }
        self.assertEqual(report["source_counts"], dict.fromkeys(expected_targets, 200))
        self.assertEqual(report["target_counts"], expected_targets)
        self.assertEqual(report["selected_counts"], expected_targets)
        self.assertEqual(report["sample_size"], 700)
        self.assertEqual(report["random_seed"], 42)
        self.assertEqual(
            report["score_model"],
            "jagoldz/gahd",
        )
        self.assertEqual(
            report["score_revision"],
            "90c6e4f50d109c42bebdeb0f43fb5c3e619612fc",
        )
        self.assertEqual(
            report["score_summaries"]["0.7-0.9"],
            {"min": 0.8, "median": 0.8, "max": 0.8},
        )

    def test_selection_fails_with_all_undersized_strata_in_diagnostic(self):
        scored = self.scored_fixture(rows_per_stratum=1)

        with self.assertRaises(ValueError) as context:
            select_master_sample(scored)

        message = str(context.exception)
        for stratum, target in {
            "0.0-0.3": 134,
            "0.3-0.5": 134,
            "0.5-0.7": 134,
            "0.7-0.9": 164,
            "0.9-1.0": 134,
        }.items():
            self.assertIn(f"{stratum}: available=1, required={target}", message)


class AnnotationExportTests(unittest.TestCase):
    def setUp(self):
        scored = MasterSampleTests.scored_fixture()
        self.master, _ = select_master_sample(scored, random_state=42)
        self.master = self.master.assign(
            translation="Standard German",
            translation_status="success",
            translation_attempts=1,
            translation_input_tokens=10,
            translation_output_tokens=3,
            translation_estimated_cost_usd=0.0000025,
            translation_error=None,
            translation_missing=False,
            translation_meta_commentary=False,
            translation_multiple_paragraphs=False,
            translation_length_ratio=1.0,
            translation_extreme_length=False,
            translation_has_anomaly=False,
        )
        self.annotations = build_annotation_frames(self.master, random_state=42)

    def test_builds_shared_and_private_assignments_without_model_leakage(self):
        self.assertEqual(set(self.annotations), {"dario", "marjorie", "yannis"})
        id_sets = {}
        text_by_id = self.master.set_index("sentence_id")["text"]
        for annotator, frame in self.annotations.items():
            with self.subTest(annotator=annotator):
                self.assertEqual(len(frame), 300)
                self.assertEqual(frame["sentence_id"].nunique(), 300)
                self.assertEqual(
                    frame.columns.tolist(),
                    ["sentence_id", "text", "label", "comment"],
                )
                self.assertTrue(frame["label"].eq("").all())
                self.assertTrue(frame["comment"].eq("").all())
                self.assertEqual(
                    frame.set_index("sentence_id")["text"].to_dict(),
                    text_by_id.loc[frame["sentence_id"]].to_dict(),
                )
                id_sets[annotator] = set(frame["sentence_id"])

        shared = set.intersection(*id_sets.values())
        self.assertEqual(len(shared), 100)
        private = {name: ids - shared for name, ids in id_sets.items()}
        self.assertEqual(
            {name: len(ids) for name, ids in private.items()},
            dict.fromkeys(id_sets, 200),
        )
        self.assertFalse(private["dario"] & private["marjorie"])
        self.assertFalse(private["dario"] & private["yannis"])
        self.assertFalse(private["marjorie"] & private["yannis"])

    def test_verifier_accepts_valid_artifacts_and_reports_overlap(self):
        report = verify_artifacts(self.master, self.annotations)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["master_rows"], 700)
        self.assertEqual(report["shared_count"], 100)
        self.assertEqual(
            report["annotator_rows"], {"dario": 300, "marjorie": 300, "yannis": 300}
        )
        self.assertEqual(
            report["private_counts"], {"dario": 200, "marjorie": 200, "yannis": 200}
        )

    def test_verifier_rejects_untranslated_or_nonterminal_master(self):
        untranslated = self.master.drop(
            columns=[
                column for column in self.master if column.startswith("translation_")
            ]
            + ["translation"]
        )
        with self.assertRaisesRegex(ValueError, "translation columns"):
            verify_artifacts(untranslated, self.annotations)

        nonterminal = self.master.copy()
        nonterminal.loc[0, "translation_status"] = "retrying_api"
        with self.assertRaisesRegex(ValueError, "terminal translation status"):
            verify_artifacts(nonterminal, self.annotations)

    def test_verifier_rejects_each_integrity_violation_specifically(self):
        cases = []

        duplicate_master = self.master.copy()
        duplicate_master.loc[1, "sentence_id"] = duplicate_master.loc[0, "sentence_id"]
        cases.append(
            (
                duplicate_master,
                self.annotations,
                "master sentence_id values must be unique",
            )
        )

        wrong_stratum = self.master.copy()
        wrong_stratum.loc[0, "austrotox_stratum"] = (
            "0.0-0.3"
            if wrong_stratum.loc[0, "austrotox_stratum"] != "0.0-0.3"
            else "0.3-0.5"
        )
        cases.append((wrong_stratum, self.annotations, "master stratum counts"))

        swapped_strata = self.master.copy()
        first = swapped_strata.index[swapped_strata["austrotox_stratum"] == "0.0-0.3"][
            0
        ]
        second = swapped_strata.index[swapped_strata["austrotox_stratum"] == "0.9-1.0"][
            0
        ]
        swapped_strata.loc[[first, second], "austrotox_stratum"] = swapped_strata.loc[
            [second, first], "austrotox_stratum"
        ].to_numpy()
        cases.append(
            (
                swapped_strata,
                self.annotations,
                "austrotox_stratum does not match hate_score",
            )
        )

        leaked = {name: frame.copy() for name, frame in self.annotations.items()}
        leaked["dario"]["hate_score"] = 0.5
        cases.append((self.master, leaked, "annotation columns"))

        changed_text = {name: frame.copy() for name, frame in self.annotations.items()}
        changed_text["dario"].loc[0, "text"] = "changed"
        cases.append((self.master, changed_text, "original Swiss German text"))

        id_sets = {
            name: set(frame["sentence_id"]) for name, frame in self.annotations.items()
        }
        shared = set.intersection(*id_sets.values())
        missing_shared = {
            name: frame.copy() for name, frame in self.annotations.items()
        }
        shared_id = next(iter(shared))
        replacement_id = next(iter(id_sets["marjorie"] - id_sets["dario"]))
        replacement_text = self.master.set_index("sentence_id").loc[
            replacement_id, "text"
        ]
        row_index = missing_shared["dario"].index[
            missing_shared["dario"]["sentence_id"] == shared_id
        ][0]
        missing_shared["dario"].loc[row_index, ["sentence_id", "text"]] = [
            replacement_id,
            replacement_text,
        ]
        cases.append((self.master, missing_shared, "shared overlap"))

        private_overlap = {
            name: frame.copy() for name, frame in self.annotations.items()
        }
        dario_private_id = next(iter(id_sets["dario"] - shared))
        marjorie_private_id = next(iter(id_sets["marjorie"] - shared))
        dario_private_text = self.master.set_index("sentence_id").loc[
            dario_private_id, "text"
        ]
        row_index = private_overlap["marjorie"].index[
            private_overlap["marjorie"]["sentence_id"] == marjorie_private_id
        ][0]
        private_overlap["marjorie"].loc[row_index, ["sentence_id", "text"]] = [
            dario_private_id,
            dario_private_text,
        ]
        cases.append((self.master, private_overlap, "private assignments overlap"))

        for master, annotations, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    verify_artifacts(master, annotations)


class PipelineCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.candidate = self.root / "candidate.csv"
        self.scored = self.root / "scored.csv"
        self.master = self.root / "master.csv"
        self.report = self.root / "report.json"
        self.checkpoint = self.root / "checkpoint.sqlite3"
        self.verification = self.root / "verification.json"
        pd.DataFrame({"sentence_id": ["1"], "text": ["Hoi"]}).to_csv(
            self.candidate, index=False
        )
        pd.DataFrame(
            {"sentence_id": ["1"], "text": ["Hoi"], "hate_score": [0.1]}
        ).to_csv(self.scored, index=False)
        pd.DataFrame({"sentence_id": ["1"], "text": ["Hoi"]}).to_csv(
            self.master, index=False
        )

    def arguments(self, command):
        return [
            command,
            "--candidate",
            str(self.candidate),
            "--scored",
            str(self.scored),
            "--master",
            str(self.master),
            "--sampling-report",
            str(self.report),
            "--checkpoint",
            str(self.checkpoint),
            "--verification-report",
            str(self.verification),
            "--output-dir",
            str(self.root),
        ]

    def test_help_and_unknown_subcommand_have_standard_argparse_behavior(self):
        with self.assertRaises(SystemExit) as help_exit, redirect_stdout(
            StringIO()
        ) as output:
            cli.main(["--help"])
        self.assertEqual(help_exit.exception.code, 0)
        self.assertIn(
            "{score,sample,translate,annotations,verify,all}", output.getvalue()
        )

        with self.assertRaises(SystemExit) as unknown_exit, redirect_stderr(StringIO()):
            cli.main(["unknown"])
        self.assertEqual(unknown_exit.exception.code, 2)

    def test_missing_stage_input_fails_before_handler_invocation(self):
        calls = []
        arguments = self.arguments("score")
        arguments[arguments.index("--candidate") + 1] = str(self.root / "missing.csv")

        with redirect_stderr(StringIO()) as error:
            exit_code = cli.main(
                arguments,
                stage_handlers={"score": lambda args: calls.append(args)},
                environ={},
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(calls, [])
        self.assertIn("does not exist", error.getvalue())

    def test_api_key_is_required_only_for_translate_and_all(self):
        for command in ("translate", "all"):
            calls = []
            with self.subTest(command=command), redirect_stderr(StringIO()) as error:
                exit_code = cli.main(
                    self.arguments(command) + ["--resume"],
                    stage_handlers={command: lambda args: calls.append(args)},
                    environ={},
                )
            self.assertEqual(exit_code, 1)
            self.assertEqual(calls, [])
            self.assertIn("ANTHROPIC_API_KEY", error.getvalue())

        calls = []
        exit_code = cli.main(
            self.arguments("score") + ["--resume"],
            stage_handlers={"score": lambda args: calls.append(args)},
            environ={},
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(calls), 1)

    def test_all_reuses_successful_translation_checkpoint_without_api_key(self):
        translation_checkpoint = self.root / "translation.json"
        master = pd.read_csv(self.master, dtype={"sentence_id": "string"})
        translation_checkpoint.write_text(
            json.dumps(
                {
                    "status": "success",
                    "translation_model": "claude-haiku-5-5",
                    "translation_prompt_version": (
                        "swiss-german-standard-few-shot-batch-v2"
                    ),
                    "row_count": 1,
                    "source_fingerprint": corpus_fingerprint(
                        master[["sentence_id", "text"]].reset_index(drop=True)
                    ),
                }
            ),
            encoding="utf-8",
        )
        calls = []

        exit_code = cli.main(
            self.arguments("all")
            + ["--translation-checkpoint", str(translation_checkpoint), "--resume"],
            stage_handlers={"all": lambda args: calls.append(args)},
            environ={},
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(calls), 1)

    def test_existing_output_is_preserved_until_resume_is_explicit(self):
        calls = []
        with redirect_stderr(StringIO()) as error:
            exit_code = cli.main(
                self.arguments("score"),
                stage_handlers={"score": lambda args: calls.append(args)},
                environ=os.environ,
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(calls, [])
        self.assertIn("already exists", error.getvalue())

        resumed = cli.main(
            self.arguments("score") + ["--resume"],
            stage_handlers={"score": lambda args: calls.append(args)},
            environ=os.environ,
        )
        self.assertEqual(resumed, 0)
        self.assertEqual(len(calls), 1)

    def test_dispatches_requested_stage_through_injected_handler(self):
        self.master.unlink()
        calls = []

        exit_code = cli.main(
            self.arguments("sample"),
            stage_handlers={"sample": lambda args: calls.append(args.command)},
            environ={},
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, ["sample"])

    def test_annotation_resume_preserves_existing_assignment_files(self):
        dario_path = self.root / "annotation_dario_stratified.csv"
        original = "sentence_id,text,label,comment\nkeep,Original,,\n"
        dario_path.write_text(original, encoding="utf-8")
        fake = pd.DataFrame(
            {"sentence_id": ["new"], "text": ["Neu"], "label": [""], "comment": [""]}
        )
        arguments = cli.build_parser().parse_args(
            self.arguments("annotations") + ["--resume"]
        )

        with patch(
            "build_swisscrawl_testset.build_annotation_frames",
            return_value={name: fake for name in ("dario", "marjorie", "yannis")},
        ):
            cli._run_annotations(arguments)

        self.assertEqual(dario_path.read_text(encoding="utf-8"), original)
        self.assertTrue((self.root / "annotation_marjorie_stratified.csv").exists())
        self.assertTrue((self.root / "annotation_yannis_stratified.csv").exists())


if __name__ == "__main__":
    unittest.main()
