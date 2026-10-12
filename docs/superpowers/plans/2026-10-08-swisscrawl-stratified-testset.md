# SwissCrawl Stratified Test-Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, hate-enriched 700-sentence SwissCrawl test set, translate only those 700 sentences with Claude Haiku 5.5, and export leakage-free annotation files.

**Architecture:** Keep score assignment, fixed-stratum sampling, translation auditing, and annotation splitting as pure functions in focused modules. Use a SQLite checkpoint store for resumable GAHD-model scoring and Haiku translation, then export final CSV and JSON artifacts atomically. Expose the stages through a command-line entry point so expensive work can resume without repeating completed rows.

**Tech Stack:** Python 3.10, pandas, NumPy, PyTorch, Hugging Face Transformers, Anthropic Python SDK, SQLite, unittest

**Spec:** `docs/superpowers/specs/2026-10-08-swisscrawl-stratified-testset-design.md`

## Global Constraints

- Input is only `data/swisscrawl_filtered.csv`.
- Score all 209,487 Swiss German candidates directly with the official GAHD classifier at pinned revision `90c6e4f50d109c42bebdeb0f43fb5c3e619612fc`.
- Use the GAHD classifier only as a sampling signal, never as a gold label.
- Select exactly `134/134/134/164/134` rows from the five ordered score strata with seed `42`.
- Call `claude-haiku-5-5` only for the selected 700 sentences.
- Human annotation files contain original Swiss German only, never translation, score, or stratum.
- Preserve existing CSV files and notebook changes; write new `*_stratified` artifacts.
- Never print or persist `ANTHROPIC_API_KEY`.
- Use TDD for every production behavior and run the complete unittest suite after every task.
- Commit only files owned by the task and never add an agent co-author. If `.git` remains read-only, record the intended commit without staging unrelated work.

## Review Focus

- A source stratum with fewer rows than its fixed quota must stop sampling with counts, not silently rebalance.
- A resumed checkpoint created from a different corpus or model revision must be rejected through a corpus/configuration fingerprint.
- An interrupted score or translation batch must leave already committed rows readable and must not duplicate completed API calls.
- Refusals, rate limits, malformed content, and persistent translation anomalies must end in explicit statuses without removing the original selected sentence.
- Annotation exports must have exactly 100 shared IDs, disjoint 200-ID private sets, and no model-derived columns.

---

## File Structure

- Modify `code/austrotox_sampling.py`: fixed five-stratum allocation, validated GAHD-model scoring, and translation audit primitives.
- Create `code/swisscrawl_pipeline.py`: checkpoint store, corpus fingerprinting, scoring, translation, sampling reports, annotation splits, atomic artifact exports, and verification.
- Create `code/build_swisscrawl_testset.py`: command-line stage orchestration and preflight checks.
- Modify `tests/test_austrotox_sampling.py`: fixed allocation and malformed model-output coverage.
- Create `tests/test_swisscrawl_pipeline.py`: checkpoint, resume, reports, translation retry, annotation, and verification tests.
- Generate `data/swisscrawl_scored_gahd.csv`, `data/master_sample_stratified.csv`, `data/stratified_sampling_report.json`, `data/stratified_verification_report.json`, and three `annotation_*_stratified.csv` files during the real run.

### Task 1: Fixed AustroTox-Adapted Allocation

**Files:**
- Modify: `code/austrotox_sampling.py`
- Modify: `tests/test_austrotox_sampling.py`

**Interfaces:**
- Produces: `FIXED_STRATUM_TARGETS: dict[str, int]`
- Produces: `sample_fixed_strata(corpus: pd.DataFrame, score_column: str = "hate_score", random_state: int = 42) -> pd.DataFrame`
- Retains: `assign_austrotox_strata(scores: Iterable[float]) -> pd.Series`

- [ ] **Step 1: Write failing fixed-allocation tests**

Add tests asserting exactly `134/134/134/164/134`, exactly 700 unique rows, deterministic repeated output with seed `42`, retention of identity terms, and an explicit error naming every undersized stratum.

- [ ] **Step 2: Run the targeted tests and verify RED**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_austrotox_sampling.py' -v`

Expected: FAIL because `sample_fixed_strata` and `FIXED_STRATUM_TARGETS` do not exist.

- [ ] **Step 3: Implement the fixed sampler**

Implement the interfaces above. Assign strata first, validate all five available counts before sampling, sample each stratum independently with seed `42`, concatenate, shuffle deterministically, and reject duplicate `sentence_id` values.

- [ ] **Step 4: Run targeted and complete tests**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/austrotox_sampling.py tests/test_austrotox_sampling.py
git commit -m "feat: add fixed AustroTox sampling allocation"
```

### Task 2: Transactional Checkpoint Store

**Files:**
- Create: `code/swisscrawl_pipeline.py`
- Create: `tests/test_swisscrawl_pipeline.py`

**Interfaces:**
- Produces: `corpus_fingerprint(frame: pd.DataFrame) -> str`
- Produces: `PipelineStore(path: Path, corpus_hash: str, configuration: Mapping[str, str])`
- Produces methods: `completed_score_ids()`, `save_scores(rows)`, `completed_translation_ids()`, `save_translation(result)`, and `close()`
- Produces: `atomic_write_csv(frame: pd.DataFrame, path: Path) -> None`
- Produces: `atomic_write_json(payload: Mapping[str, Any], path: Path) -> None`

- [ ] **Step 1: Write failing checkpoint tests**

Use temporary directories and real SQLite. Test committed-row resume, idempotent upserts, rollback after an injected transaction failure, corpus fingerprint mismatch, model revision mismatch, and atomic replacement of existing CSV/JSON artifacts.

- [ ] **Step 2: Run the checkpoint tests and verify RED**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_swisscrawl_pipeline.py' -v`

Expected: FAIL because the store and export functions do not exist.

- [ ] **Step 3: Implement the store and atomic exports**

Use SQLite primary keys on `sentence_id`, explicit transactions, metadata rows for fingerprint/configuration, and `Path.replace()` from a sibling temporary file for final artifacts. Do not store secrets.

- [ ] **Step 4: Run targeted and complete tests**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests PASS and no temporary files remain.

- [ ] **Step 5: Commit**

```bash
git add code/swisscrawl_pipeline.py tests/test_swisscrawl_pipeline.py
git commit -m "feat: add resumable pipeline checkpoints"
```

### Task 3: Resumable GAHD Corpus Scoring

**Files:**
- Modify: `code/swisscrawl_pipeline.py`
- Modify: `tests/test_swisscrawl_pipeline.py`

**Interfaces:**
- Consumes: `PipelineStore`, `score_with_gahd`
- Produces: `score_candidate_pool(frame: pd.DataFrame, store: PipelineStore, classifier, batch_size: int) -> pd.DataFrame`
- Produces: `build_gahd_classifier(device: str | None = None)`

- [ ] **Step 1: Write failing scoring tests**

Test batching, source-order preservation, skipping already scored IDs on resume, exact score-to-ID alignment, duplicate-ID rejection, empty-text rejection, and rejection of a classifier result with the wrong cardinality.

- [ ] **Step 2: Run the scoring tests and verify RED**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_swisscrawl_pipeline.py' -v`

Expected: FAIL because scoring orchestration does not exist.

- [ ] **Step 3: Implement scoring orchestration**

Load the pinned model revision, select MPS when available and otherwise CPU, process only missing IDs in bounded batches, commit each completed batch, merge scores back by `sentence_id`, and export `data/swisscrawl_scored_gahd.csv` atomically only after all rows are scored.

- [ ] **Step 4: Run targeted and complete tests**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/swisscrawl_pipeline.py tests/test_swisscrawl_pipeline.py
git commit -m "feat: add resumable SwissCrawl hate scoring"
```

### Task 4: Sampling Artifact and Diagnostic Report

**Files:**
- Modify: `code/swisscrawl_pipeline.py`
- Modify: `tests/test_swisscrawl_pipeline.py`

**Interfaces:**
- Consumes: `sample_fixed_strata`, atomic export functions
- Produces: `select_master_sample(scored: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, dict[str, Any]]`

- [ ] **Step 1: Write failing report tests**

Assert exact source, target, and selected counts for all strata; score min/median/max summaries; seed and pinned-model metadata; 700 unique selected IDs; deterministic output; and failure on undersized strata.

- [ ] **Step 2: Run the report tests and verify RED**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_swisscrawl_pipeline.py' -v`

Expected: FAIL because the selection/report interface does not exist.

- [ ] **Step 3: Implement selection and report generation**

Generate `data/master_sample_stratified.csv` and `data/stratified_sampling_report.json` atomically. Preserve all SwissCrawl provenance columns and add only `hate_score` and `austrotox_stratum` at this stage.

- [ ] **Step 4: Run targeted and complete tests**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/swisscrawl_pipeline.py tests/test_swisscrawl_pipeline.py
git commit -m "feat: add stratified sample report"
```

### Task 5: Haiku Batch Translation and Audit

**Files:**
- Modify: `code/swisscrawl_pipeline.py`
- Modify: `tests/test_swisscrawl_pipeline.py`

**Interfaces:**
- Consumes: `audit_translations`
- Produces: `build_batch_translation_prompt(texts)` with the approved four examples
- Produces: `translate_selected_batch(frame, client, max_api_attempts)`
- Produces: `restore_batch_translation(frame, report)`

- [ ] **Step 1: Write failing translation tests**

Use a deterministic fake Anthropic boundary. Test one request containing every
numbered sentence, the exact approved prompt, strict numbered-output parsing,
transient error backoff, batch token accounting, and checkpoint restoration.

- [ ] **Step 2: Run the translation tests and verify RED**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_swisscrawl_pipeline.py' -v`

Expected: FAIL because the translation interfaces do not exist.

- [ ] **Step 3: Implement prompt and translation state machine**

Use `claude-haiku-5-5`, `output_config={"effort": "medium"}`, streaming, and
`max_tokens=128000`. Send all 700 rows in one prompt, retry the unchanged prompt
only after failures, and accept output only when lines 1 through 700 are complete
and ordered.

- [ ] **Step 4: Merge audited translation fields into the master artifact**

Add `translation`, batch attempt count, status, and anomaly flags to
`data/master_sample_stratified.csv`. Store raw output, batch token totals, and
estimated cost in `data/checkpoints/haiku_batch_translation.json`.

- [ ] **Step 5: Run targeted and complete tests**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add code/swisscrawl_pipeline.py tests/test_swisscrawl_pipeline.py
git commit -m "feat: translate and audit selected SwissCrawl sentences"
```

### Task 6: Leakage-Free Annotation Exports and Verifier

**Files:**
- Modify: `code/swisscrawl_pipeline.py`
- Modify: `tests/test_swisscrawl_pipeline.py`

**Interfaces:**
- Produces: `build_annotation_frames(master: pd.DataFrame, annotators: Sequence[str] = ("dario", "marjorie", "yannis"), shared_count: int = 100, random_state: int = 42) -> dict[str, pd.DataFrame]`
- Produces: `verify_artifacts(master: pd.DataFrame, annotations: Mapping[str, pd.DataFrame]) -> dict[str, Any]`

- [ ] **Step 1: Write failing annotation tests**

Assert 300 rows per annotator, the same 100 shared IDs in all three files, disjoint private sets of 200, exact original-text preservation, empty label/comment values, and exact column list `sentence_id,text,label,comment`.

- [ ] **Step 2: Write failing verifier tests**

Mutate one fixture at a time to introduce duplicate IDs, wrong stratum counts, leaked score/translation columns, changed original text, missing shared IDs, and private-set overlap. Assert a specific failure for each mutation.

- [ ] **Step 3: Run annotation tests and verify RED**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_swisscrawl_pipeline.py' -v`

Expected: FAIL because annotation and verification interfaces do not exist.

- [ ] **Step 4: Implement annotation splits and verification**

Export the three `annotation_*_stratified.csv` files and `data/stratified_verification_report.json` atomically. Do not overwrite legacy annotation files.

- [ ] **Step 5: Run targeted and complete tests**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add code/swisscrawl_pipeline.py tests/test_swisscrawl_pipeline.py
git commit -m "feat: export verified annotation assignments"
```

### Task 7: Command-Line Orchestration and Preflight

**Files:**
- Create: `code/build_swisscrawl_testset.py`
- Modify: `tests/test_swisscrawl_pipeline.py`

**Interfaces:**
- Produces subcommands: `score`, `sample`, `translate`, `annotations`, `verify`, and `all`
- Default checkpoint: `data/checkpoints/swisscrawl_testset.sqlite3`

- [ ] **Step 1: Write failing CLI tests**

Test help output, unknown subcommand failure, missing input failure, missing API key failure only for `translate` and `all`, preservation of existing outputs unless `--resume` is supplied, and stage invocation with dependency injection.

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_swisscrawl_pipeline.py' -v`

Expected: FAIL because the CLI does not exist.

- [ ] **Step 3: Implement the CLI**

Load `.env` without printing values, perform stage-specific preflight checks, show row counts and checkpoint progress, and return nonzero on any failed invariant. `--resume` must reuse a matching checkpoint and reject mismatched metadata.

- [ ] **Step 4: Run static and complete verification**

Run:

```bash
black --check code/austrotox_sampling.py code/swisscrawl_pipeline.py code/build_swisscrawl_testset.py tests/
.venv/bin/python -m compileall -q code tests
.venv/bin/python -m unittest discover -s tests -v
```

Expected: formatter clean, compilation exit 0, all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/build_swisscrawl_testset.py code/swisscrawl_pipeline.py tests/test_swisscrawl_pipeline.py
git commit -m "feat: add SwissCrawl test-set pipeline CLI"
```

### Task 8: Execute the Pipeline on SwissCrawl

**Files:**
- Generate: `data/swisscrawl_scored_gahd.csv`
- Generate: `data/master_sample_stratified.csv`
- Generate: `data/stratified_sampling_report.json`
- Generate: `data/annotation_dario_stratified.csv`
- Generate: `data/annotation_marjorie_stratified.csv`
- Generate: `data/annotation_yannis_stratified.csv`
- Generate: `data/stratified_verification_report.json`

**Interfaces:**
- Consumes: CLI from Task 7 and `ANTHROPIC_API_KEY` from the environment
- Produces: final verified data artifacts and verification summary

- [ ] **Step 1: Run preflight without API work**

Run: `.venv/bin/python code/build_swisscrawl_testset.py score --resume`

Expected: exactly 209,487 input rows, pinned model metadata, progress resumed from checkpoint, and final scored CSV after completion.

- [ ] **Step 2: Generate the fixed stratified sample**

Run: `.venv/bin/python code/build_swisscrawl_testset.py sample --resume`

Expected: exactly 700 rows with counts `134/134/134/164/134` and a JSON report with seed `42`.

- [ ] **Step 3: Verify API credentials without exposing them**

Run: `.venv/bin/python code/build_swisscrawl_testset.py translate --resume`

Expected when configured: send one prompt containing all 700 selected rows or
reuse the validated batch checkpoint. Expected when unavailable: stop before the
first request with a message naming `ANTHROPIC_API_KEY`.

- [ ] **Step 4: Resume translation until all 700 rows have terminal statuses**

Repeat the same `translate --resume` command after transient interruptions. A
completed checkpoint must be reused without another API request.

- [ ] **Step 5: Generate annotation files and verify all artifacts**

Run:

```bash
.venv/bin/python code/build_swisscrawl_testset.py annotations --resume
.venv/bin/python code/build_swisscrawl_testset.py verify --resume
```

Expected: three 300-row annotation files, 100 shared IDs, disjoint private sets, exact original text, no model-derived annotation columns, and zero unresolved structural verification failures.

- [ ] **Step 6: Record operational results**

Report scoring duration, device, source and selected stratum counts, translation calls, retries, unresolved anomalies, token totals, and estimated cost. Do not commit raw data unless the repository's data policy explicitly requires it.
