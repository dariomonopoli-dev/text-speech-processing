"""Run the train and predict steps of one JSON config."""

import argparse
import json
import os
from types import ModuleType

import predict
import train
from get_loggers import get_logger

STEPS = {"train": train, "predict": predict}

run_logger = get_logger("run")


def load_config(path_config: str) -> dict[str, dict]:
    if not os.path.isfile(path_config):
        raise FileNotFoundError(f"Config '{path_config}' does not exist.")
    with open(path_config) as fin:
        return json.load(fin)


def get_step_dests(step: ModuleType) -> set[str]:
    return {action.dest for action in step.get_argparser()._actions} - {"help"}


def check_config_section(key: str, section: dict, step: ModuleType) -> None:
    """Refuse a section whose keys differ from the argparse dests of its step."""
    expected_keys = get_step_dests(step)
    missing_keys = sorted(expected_keys - section.keys())
    if missing_keys:
        raise KeyError(f"Config section '{key}' is missing the keys {missing_keys}.")
    unknown_keys = sorted(section.keys() - expected_keys)
    if unknown_keys:
        raise ValueError(f"Config section '{key}' has unknown keys {unknown_keys}.")


def main(args: argparse.Namespace) -> None:
    config = load_config(args.path_config)
    for step_name, step in STEPS.items():
        for key in config:
            if key.startswith(step_name):
                check_config_section(key, config[key], step)
    for step_name, step in STEPS.items():
        for key in config:
            if key.startswith(step_name):
                run_logger.info(f"Start {key}.")
                step.main(argparse.Namespace(**config[key]))
                run_logger.info(f"Finished {key}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path_config",
        required=True,
        help='Path to a JSON file whose "train" and "predict*" sections hold the '
        "arguments of train.py and predict.py.",
    )
    cmd_args = parser.parse_args()
    try:
        main(cmd_args)
    except Exception:
        run_logger.exception(f"Run failed for config {cmd_args.path_config}.")
        raise
