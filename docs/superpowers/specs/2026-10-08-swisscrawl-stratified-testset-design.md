# SwissCrawl Stratified Test-Set Pipeline

Date: 2026-10-08

## Objective

Build a reproducible 700-sentence Swiss German test set enriched for hateful
content while limiting Claude Haiku normalisation to the final 700 selected
sentences. Human annotators must always label the original Swiss German text,
never the normalised translation.

The pipeline adapts the score strata reported by AustroTox. It does not claim
to reproduce AustroTox exactly because AustroTox used Perspective API scores,
whereas this project uses official GAHD classifier scores on Swiss German candidate
sentences.

## Methodological Interpretation

The official GAHD classifier's hate probability is used only as a ranking and
sampling signal.
It is not a gold label, a calibrated estimate of prevalence, or a substitute
for manual annotation. Because the GAHD classifier is applied directly to Swiss German,
its scores may be less reliable than scores on Standard German.

The resulting test set is deliberately enriched for likely hateful content.
It is not representative of the natural prevalence of hateful content in
SwissCrawl. Reported class prevalence and aggregate evaluation metrics must be
interpreted in that context.

## Inputs

- `data/swisscrawl_filtered.csv`
- Required columns: `sentence_id`, `text`, and the existing provenance columns
- GAHD model: `jagoldz/gahd`
- Pinned GAHD revision: `90c6e4f50d109c42bebdeb0f43fb5c3e619612fc`
- Claude model: `claude-haiku-5-5`
- Anthropic credentials supplied through `ANTHROPIC_API_KEY`

## Data Flow

### 1. Score the complete candidate pool

Run the pinned GAHD model over the original Swiss German `text` column.
Request all class probabilities and store the probability associated with the
class `1` (`LABEL_1`) as `hate_score`.

Scoring must:

- preserve a one-to-one mapping between `sentence_id` and score;
- reject malformed classifier output rather than silently misalign rows;
- process the corpus in batches;
- write resumable checkpoints;
- record model name, model revision, batch size, and execution timestamp.

Output:

- `data/swisscrawl_scored_gahd.csv`

### 2. Assign score strata

Assign each valid score to one non-overlapping interval:

- `[0.0, 0.3)`
- `[0.3, 0.5)`
- `[0.5, 0.7)`
- `[0.7, 0.9)`
- `[0.9, 1.0]`

Store the interval in `austrotox_stratum`. Produce a diagnostic table containing
the size of every source stratum before sampling.

### 3. Select exactly 700 sentences

Use a fixed random seed of `42`. The project uses the five AustroTox score
intervals and deliberately allocates more examples to the `0.7-0.9` interval.
The following 700-row allocation is specific to this study and is not claimed
to come from the AustroTox source code:

| Score stratum | Target count |
| --- | ---: |
| `[0.0, 0.3)` | 134 |
| `[0.3, 0.5)` | 134 |
| `[0.5, 0.7)` | 134 |
| `[0.7, 0.9)` | 164 |
| `[0.9, 1.0]` | 134 |
| Total | 700 |

The pipeline must fail with an explicit diagnostic if any source stratum has
fewer rows than its target count. It must not silently move quota between
strata because that would change the documented sampling distribution.

Before translation, export a stratification report containing source counts,
target counts, selected counts, score summaries, and the random seed.

Output:

- `data/master_sample_stratified.csv`
- `data/stratified_sampling_report.json`

### 4. Normalise only the selected 700 sentences

Translate the selected Swiss German sentences to Standard German with Claude
Haiku 5.5. Send all 700 numbered sentences in one request using the approved
German four-example prompt. The prompt explicitly requires literal preservation
of meaning, tone, insults, and vulgar expressions and exactly one numbered
translation per line.

Translation must:

- make one streaming request containing the complete selected set;
- require response numbers `1` through `700` exactly once and in order;
- retry transient API and rate-limit failures with exponential backoff;
- save the complete raw response atomically before replacing the master CSV;
- reuse a matching batch checkpoint after interruption;
- reject incomplete, reordered, or explanatory output;
- record token usage and estimated cost at batch level.

The checkpoint is bound to the ordered `sentence_id` and source-text fingerprint,
model identifier, and prompt version. It cannot be reused for a changed sample.

### 5. Audit and retry anomalous translations

Audit each translation for:

- missing or empty output;
- model commentary or translation notes;
- multiple paragraphs;
- extreme source-to-translation length ratios.

The audit deliberately contains no blocklist for identity terms, profanity, or
offensive vocabulary. Content such as `Juden`, `Frauen`, insults, and vulgarity
must remain eligible.

Do not issue sentence-level repair prompts because they would violate the fixed
batch protocol. Do not replace a selected Swiss German sentence merely because
its translation remains anomalous. Retain it and mark its translation status for
manual review.

The final master sample contains:

- original Swiss German `text`;
- Standard German `translation`;
- `hate_score`;
- `austrotox_stratum`;
- batch attempt count;
- anomaly flags and final translation status;
- original SwissCrawl provenance columns.

### 6. Produce annotation files

Create a deterministic shared overlap of 100 sentences from the final master
sample. Shuffle the remaining 600 sentences with the fixed seed and divide them
into three disjoint sets of 200.

Each annotator receives:

- the same 100 overlap sentences;
- one exclusive set of 200 sentences;
- 300 sentences in total.

Annotation files expose only:

- `sentence_id`;
- original Swiss German `text`;
- empty `label` column;
- empty `comment` column.

They must not expose the translation, GAHD model score, or stratum, because
those fields could bias annotators.

Outputs:

- `data/annotation_dario_stratified.csv`
- `data/annotation_marjorie_stratified.csv`
- `data/annotation_yannis_stratified.csv`
- matching `.xlsx` workbooks with all text cells stored explicitly as strings

The CSV files are canonical machine-readable artifacts. Annotators should use
the XLSX workbooks, which prevent spreadsheet applications from treating forum
text beginning with formula characters as executable formulas.

Existing annotation files must not be overwritten.

## Reproducibility and Safety

- Use seed `42` for sampling and overlap selection.
- Pin the GAHD model revision.
- Record the Claude model identifier and prompt version.
- Write checkpoints atomically through a temporary file followed by replacement.
- Never print or persist the Anthropic API key.
- Stop before API work when `ANTHROPIC_API_KEY` is unavailable.
- Preserve all existing data files and notebook changes.

## Verification

Automated tests must cover:

- exact boundary assignment for all five strata;
- exact target allocation `134/134/134/164/134`;
- deterministic sampling under seed `42`;
- failure on undersized strata;
- one-to-one classifier score alignment;
- checkpoint resume behavior;
- anomaly detection and retry limits;
- annotation split sizes and overlap identities;
- absence of translations, scores, and strata from annotation files.

Before delivery, verify:

- exactly 700 unique `sentence_id` values in the master sample;
- exact selected count in every stratum;
- no overlap among the three exclusive annotation subsets;
- exactly 100 shared rows appearing in all three annotation files;
- 300 rows in each annotation file;
- all annotation text exactly matches the original SwissCrawl text;
- all unresolved translation anomalies are explicitly reported.

## Paper Description

The paper must describe the procedure as an adaptation of AustroTox, not an
exact reproduction. It must state that the GAHD classifier scores the original Swiss
German candidate pool, that Haiku normalises only the selected 700 sentences,
that model scores are used solely for sampling, and that manual annotation is
performed on the original Swiss German text.

