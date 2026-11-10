"""Keep only the best checkpoint of a run, chosen by eval_f1-macro on the dev split."""

import argparse
import json
import os
import re
import shutil

from get_loggers import get_logger

CHECKPOINT_REGEX = re.compile(r"checkpoint-(\d+)")
BEST_METRIC = "eval_f1-macro"

cp_logger = get_logger("checkpoint")


def get_checkpoints(path_run_dir: str) -> list[int]:
    """Return the sorted step numbers of all checkpoints in a run directory."""
    step_nums = []
    for name in os.listdir(path_run_dir):
        match = re.fullmatch(CHECKPOINT_REGEX, name)
        if match:
            step_nums.append(int(match.group(1)))
    return sorted(step_nums)


def get_best_checkpoint(path_run_dir: str, checkpoints: list[int]) -> int:
    """Return the step with the highest dev macro F1, the first one on ties."""
    path_train_state = os.path.join(
        path_run_dir, f"checkpoint-{checkpoints[-1]}", "trainer_state.json"
    )
    with open(path_train_state) as fin:
        log_history = json.load(fin)["log_history"]
    checkpoint_eval = {
        item["step"]: item[BEST_METRIC] for item in log_history if BEST_METRIC in item
    }
    if not checkpoint_eval:
        raise ValueError(f"No '{BEST_METRIC}' entries in {path_train_state}.")
    return max(checkpoint_eval, key=checkpoint_eval.get)


def get_checkpoint_path(path_run_dir: str) -> str:
    """Return the path of the single checkpoint left in a run directory."""
    if not os.path.isdir(path_run_dir):
        raise FileNotFoundError(f"Run directory '{path_run_dir}' does not exist.")
    checkpoints = get_checkpoints(path_run_dir)
    if len(checkpoints) != 1:
        raise ValueError(
            f"Expected exactly one checkpoint in '{path_run_dir}', found {checkpoints}."
        )
    return os.path.join(path_run_dir, f"checkpoint-{checkpoints[0]}")


def delete_non_best_checkpoints(
    path_run_dir: str, checkpoints: list[int], best_checkpoint: int
) -> None:
    """Remove every checkpoint directory except the best one."""
    for checkpoint in checkpoints:
        if checkpoint != best_checkpoint:
            path_run = os.path.realpath(path_run_dir)
            path_checkpoint = os.path.realpath(
                os.path.join(path_run, f"checkpoint-{checkpoint}")
            )
            if os.path.dirname(path_checkpoint) != path_run:
                raise ValueError(f"Checkpoint resolves outside run: {path_checkpoint}")
            cp_logger.info(f"Delete checkpoint: {path_checkpoint}")
            shutil.rmtree(path_checkpoint)


def main(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.path_run_dir):
        raise FileNotFoundError(f"Run directory '{args.path_run_dir}' does not exist.")
    checkpoints = get_checkpoints(args.path_run_dir)
    if not checkpoints:
        raise ValueError(f"No checkpoints found in '{args.path_run_dir}'.")
    best_checkpoint = get_best_checkpoint(args.path_run_dir, checkpoints)
    if best_checkpoint not in checkpoints:
        raise ValueError(
            f"Best step {best_checkpoint} has no checkpoint in '{args.path_run_dir}'."
        )
    cp_logger.info(f"Checkpoints available: {checkpoints}")
    cp_logger.info(f"Best checkpoint: {best_checkpoint}, delete other checkpoints.")
    delete_non_best_checkpoints(args.path_run_dir, checkpoints, best_checkpoint)
    cp_logger.info("Other checkpoints deleted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path_run_dir",
        required=True,
        help="Path to a run directory containing checkpoint-<step> directories.",
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
