"""Shared experiment paths, configuration, and index validation."""

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAME_FEATURE_COUNT = 84


@dataclass(frozen=True)
class ExperimentConfig:
    method: str
    window_length: int
    codeword_count: int
    random_state: int


def validate_experiment(experiment: int) -> None:
    if isinstance(experiment, bool) or not isinstance(experiment, int):
        raise ValueError("実験番号は整数で指定してください")
    if experiment < 1:
        raise ValueError("実験番号は1以上で指定してください")


def validate_index(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name}は整数で指定してください")
    if value < 1:
        raise ValueError(f"{name}は1以上で指定してください")


def experiment_directory(experiment: int) -> Path:
    validate_experiment(experiment)
    return PROJECT_ROOT / "experiments" / f"exp{experiment:02d}"


def load_experiment_config(experiment: int) -> ExperimentConfig:
    config_path = experiment_directory(experiment) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"実験設定ファイルが見つかりません: {config_path}"
        )

    try:
        with config_path.open(encoding="utf-8") as config_file:
            config_data = json.load(config_file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"実験設定ファイルをJSONとして読み込めません: {config_path}"
        ) from error

    if not isinstance(config_data, dict):
        raise ValueError(
            f"{config_path} はJSONオブジェクトである必要があります"
        )

    method = config_data.get("method")
    window_length = config_data.get("L")
    codeword_count = config_data.get("D")
    random_state = config_data.get("seed")

    if not isinstance(method, str):
        raise ValueError(f"{config_path} の method は文字列で指定してください")
    if (
        isinstance(window_length, bool)
        or not isinstance(window_length, int)
        or window_length < 2
        or window_length % 2
    ):
        raise ValueError(
            f"{config_path} の L は2以上の偶数で指定してください"
        )
    if (
        isinstance(codeword_count, bool)
        or not isinstance(codeword_count, int)
        or codeword_count < 1
    ):
        raise ValueError(
            f"{config_path} の D は1以上の整数で指定してください"
        )
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise ValueError(f"{config_path} の seed は整数で指定してください")

    return ExperimentConfig(
        method=method,
        window_length=window_length,
        codeword_count=codeword_count,
        random_state=random_state,
    )
