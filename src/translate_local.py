"""Translate Swiss German evaluation sentences into Standard German with a local model."""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from get_loggers import get_logger
from pipeline_io import (
    PipelineError,
    check_inputs_exist,
    check_outputs_absent,
    check_positive,
    load_csv,
    write_csv,
)
from translation import (
    API_USAGE_COLUMNS,
    add_translations,
    flag_unchanged,
    parse_translation_response,
    to_translation_prompt,
)

logger = get_logger("translate")

PROMPT_VERSION = "swiss-german-standard-few-shot-local-v1"
SOURCE_COLUMNS = ["sentence_id", "text"]


def check_sentences(sentences: pd.DataFrame, path_sentences: Path) -> None:
    missing_columns = sorted(set(SOURCE_COLUMNS) - set(sentences.columns))
    if missing_columns:
        raise KeyError(f"{path_sentences} lacks columns {missing_columns}")
    if sentences["sentence_id"].isna().any() or sentences["sentence_id"].duplicated().any():
        raise ValueError(f"{path_sentences} needs unique, non-null sentence_id values.")


def translate_batch(
    texts: Sequence[str], generate: Callable[[str], str]
) -> list[str | None]:
    """Translate texts in one prompt, splitting the batch if the reply cannot be parsed."""
    try:
        return list(parse_translation_response(generate(to_translation_prompt(texts)), len(texts)))
    except ValueError as error:
        logger.warning(f"Unparsable reply for {len(texts)} sentences: {error}")
    if len(texts) == 1:
        return [None]
    middle = len(texts) // 2
    return translate_batch(texts[:middle], generate) + translate_batch(
        texts[middle:], generate
    )


def load_checkpoint(path_checkpoint: Path, model_name: str) -> dict[str, str | None]:
    """Load finished translations by sentence_id and refuse another model's records."""
    if not path_checkpoint.is_file():
        return {}
    translations: dict[str, str | None] = {}
    with path_checkpoint.open(encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line)
            if record["model_name"] != model_name or record["prompt_version"] != PROMPT_VERSION:
                raise PipelineError(
                    f"Checkpoint {path_checkpoint} was written with "
                    f"{record['model_name']} / {record['prompt_version']}, "
                    f"not {model_name} / {PROMPT_VERSION}."
                )
            translations[record["sentence_id"]] = record["translation"]
    return translations


def append_checkpoint(
    path_checkpoint: Path,
    sentence_ids: Sequence[str],
    translations: Sequence[str | None],
    model_name: str,
) -> None:
    path_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with path_checkpoint.open("a", encoding="utf-8") as fout:
        for sentence_id, translation in zip(sentence_ids, translations, strict=True):
            record = {
                "sentence_id": sentence_id,
                "translation": translation,
                "model_name": model_name,
                "prompt_version": PROMPT_VERSION,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()


def translate_sentences(
    sentences: pd.DataFrame,
    generate: Callable[[str], str],
    model_name: str,
    batch_size: int,
    path_checkpoint: Path,
) -> pd.DataFrame:
    """Translate every sentence not yet in the checkpoint and return the audited frame."""
    done = load_checkpoint(path_checkpoint, model_name)
    pending = sentences[~sentences["sentence_id"].isin(done)]
    logger.info(f"{len(done)} sentences in the checkpoint, {len(pending)} to translate.")
    for start in range(0, len(pending), batch_size):
        batch = pending.iloc[start : start + batch_size]
        translations = translate_batch(batch["text"].tolist(), generate)
        append_checkpoint(path_checkpoint, batch["sentence_id"].tolist(), translations, model_name)
        done.update(zip(batch["sentence_id"], translations, strict=True))
        logger.info(f"Translated {min(start + batch_size, len(pending))}/{len(pending)}.")

    translated = add_translations(
        sentences[SOURCE_COLUMNS].reset_index(drop=True),
        [done[sentence_id] for sentence_id in sentences["sentence_id"]],
        attempts=1,
    )
    translated = flag_unchanged(translated.drop(columns=list(API_USAGE_COLUMNS)))
    translated["translation_model"] = model_name
    translated["translation_prompt_version"] = PROMPT_VERSION
    return translated


def make_generate(model_name: str, max_tokens: int) -> Callable[[str], str]:
    """Load an MLX model and return a greedy prompt-to-reply function."""
    from mlx_lm import generate, load

    model, tokenizer = load(model_name)

    def generate_reply(prompt: str) -> str:
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )
        return generate(model, tokenizer, prompt=chat_prompt, max_tokens=max_tokens)

    return generate_reply


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), default=str))
    check_inputs_exist([args.path_sentences])
    check_positive("batch_size", args.batch_size)
    check_positive("max_tokens", args.max_tokens)
    check_outputs_absent([args.path_output], args.resume)
    if not args.resume and args.path_checkpoint.exists():
        raise PipelineError(
            f"Checkpoint {args.path_checkpoint} exists; pass --resume to reuse it."
        )

    sentences = load_csv(args.path_sentences)
    check_sentences(sentences, args.path_sentences)

    logger.info(f"Start translation with {args.model_name}.")
    translated = translate_sentences(
        sentences,
        make_generate(args.model_name, args.max_tokens),
        args.model_name,
        args.batch_size,
        args.path_checkpoint,
    )
    logger.info("Finished translation.")

    num_anomalies = int(translated["translation_has_anomaly"].sum())
    if num_anomalies:
        logger.warning(f"{num_anomalies} translations are flagged for manual review.")
    write_csv(translated, args.path_output)
    logger.info(f"wrote {len(translated)} rows to {args.path_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate Swiss German sentences with a local MLX model."
    )
    # data
    parser.add_argument(
        "--path_sentences",
        type=Path,
        default=Path("data/eval_sentences.csv"),
        help="CSV with the columns sentence_id and text.",
    )
    parser.add_argument(
        "--path_output",
        type=Path,
        required=True,
        help="CSV that receives the sentences with their translations.",
    )
    parser.add_argument(
        "--path_checkpoint",
        type=Path,
        required=True,
        help="JSONL file of finished translations, reused with --resume.",
    )
    # model
    parser.add_argument(
        "--model_name",
        required=True,
        help="Hugging Face id of an MLX-compatible model, e.g. an Apertus instruct model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Sentences per prompt."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens per reply; must fit the whole translated batch.",
    )
    # saving
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the checkpoint and overwrite the output file.",
    )
    cmd_args = parser.parse_args()
    try:
        main(cmd_args)
    except PipelineError as error:
        logger.error(str(error))
        sys.exit(1)
