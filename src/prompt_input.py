"""Prompt display and keyboard input handling."""

import argparse
import json
import msvcrt
import sys
import time
from pathlib import Path

_render_initialized = False
_last_prompt_line_count = 0
_last_answer_line_count = 0


def load_prompts() -> list[str]:
    prompt_path = Path(__file__).with_name("prompts.json")

    with prompt_path.open(encoding="utf-8") as file:
        data = json.load(file)

    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not all(
        isinstance(prompt, str) for prompt in prompts
    ):
        raise ValueError('prompts.json の "prompts" は文字列の配列である必要があります。')

    if not prompts:
        raise ValueError("prompts.json にプロンプトがありません。")

    return prompts



def select_prompt(prompts: list[str], prompt_index: int) -> str:
    if prompt_index < 1 or prompt_index > len(prompts):
        available_indices = ", ".join(str(i) for i in range(1, len(prompts) + 1))
        raise ValueError(
            f"指定されたプロンプトのインデックスが無効です: {prompt_index} "
            f"(選択可能: {available_indices})"
        )

    return prompts[prompt_index - 1]


def flash_background() -> None:
    """Flash the terminal background briefly to signal an incorrect input."""
    if not sys.stdout.isatty():
        return

    print("\033[1;1H\033[48;5;52m          \033[0m", end="", flush=True)
    time.sleep(0.04)
    print("\033[1;1H\033[42m          \033[0m", end="", flush=True)


def render_screen(prompt: str, answer: str, cursor: int) -> None:
    global _render_initialized
    global _last_prompt_line_count
    global _last_answer_line_count

    output = []
    if not _render_initialized:
        output.append("\033[2J\033[H")
        _render_initialized = True
    else:
        output.append("\033[H")

    status_color = "42" if answer == prompt[:len(answer)] else "41"
    prompt_lines = prompt.split("\n")
    answer_lines = answer.split("\n")
    prompt_line_count = len(prompt_lines)

    # Draw each fixed screen row independently and erase its old content first.
    output.append(f"\033[1;1H\033[2K\033[{status_color}m          \033[0m")
    output.append("\033[2;1H\033[2K" + "-" * 70)
    old_separator_row = 3 + _last_prompt_line_count
    separator_row = 3 + prompt_line_count
    for row in range(separator_row, old_separator_row + 1):
        output.append(f"\033[{row};1H\033[2K")
    for row, line in enumerate(prompt_lines, start=3):
        output.append(f"\033[{row};1H\033[2K{line}")

    output.append(f"\033[{separator_row};1H\033[2K" + "-" * 70)

    answer_start_row = separator_row + 1
    stale_answer_line_count = max(_last_answer_line_count, len(answer_lines))
    for row in range(answer_start_row, answer_start_row + stale_answer_line_count):
        output.append(f"\033[{row};1H\033[2K")
    for row, line in enumerate(answer_lines, start=answer_start_row):
        output.append(f"\033[{row};1H{line}")

    input_start_row = 5 + prompt.count("\n")
    answer_before_cursor = answer[:cursor]
    cursor_row = input_start_row + answer_before_cursor.count("\n")
    cursor_column = len(answer_before_cursor.rsplit("\n", 1)[-1]) + 1
    output.append(f"\033[{cursor_row};{cursor_column}H")

    _last_prompt_line_count = prompt_line_count
    _last_answer_line_count = len(answer_lines)
    print("".join(output), end="", flush=True)


def handle_enter(
    prompt: str,
    answer: str,
    cursor: int,
) -> tuple[str, int, str]:
    """Handle Enter and report whether to insert newline, finish, or ignore."""
    if cursor < len(prompt) and prompt[cursor] == "\n":
        answer = answer[:cursor] + "\n" + answer[cursor:]
        return answer, cursor + 1, "insert_newline"

    if cursor == len(answer) == len(prompt):
        return answer, cursor, "finish"

    return answer, cursor, "ignored"


def read_answer(prompt: str, on_key=None) -> str | None:
    answer = ""
    cursor = 0
    render_screen(prompt, answer, cursor)

    while True:
        character = msvcrt.getwch()
        if on_key is not None:
            on_key()

        if character == "\x1b":
            render_screen(prompt, answer, cursor)
            print()
            return None

        if character in ("\r", "\n"):
            answer, cursor, action = handle_enter(prompt, answer, cursor)
            if action == "insert_newline":
                render_screen(prompt, answer, cursor)
                continue
            if action == "finish":
                render_screen(prompt, answer, cursor)
                print()
                return answer

            flash_background()
            render_screen(prompt, answer, cursor)
            continue

        if character in ("\x00", "\xe0"):
            key = msvcrt.getwch()
            if key == "K":
                cursor = max(0, cursor - 1)
            elif key == "M":
                cursor = min(len(answer), cursor + 1)
            elif key == "G":
                cursor = answer.rfind("\n", 0, cursor) + 1
            elif key == "O":
                next_newline = answer.find("\n", cursor)
                cursor = len(answer) if next_newline == -1 else next_newline
            elif key == "S" and cursor < len(answer):
                answer = answer[:cursor] + answer[cursor + 1:]
            render_screen(prompt, answer, cursor)
            continue

        if character == "\x08":
            if cursor > 0:
                answer = answer[:cursor - 1] + answer[cursor:]
                cursor -= 1
                render_screen(prompt, answer, cursor)
            continue

        was_correct = answer == prompt[:len(answer)]
        answer = answer[:cursor] + character + answer[cursor:]
        cursor += 1
        is_correct = answer == prompt[:len(answer)]

        if was_correct and not is_correct:
            flash_background()
        render_screen(prompt, answer, cursor)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="コマンドラインで指定したプロンプトを表示します。"
    )
    parser.add_argument(
        "prompt_index", 
        type=int, 
        default=1,
        help="プロンプトのインデックス"
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    try:
        prompt = select_prompt(load_prompts(), args.prompt_index)
    except ValueError as error:
        raise SystemExit(f"エラー: {error}") from error

    answer = read_answer(prompt)

    if answer is None:
        print("Escキーで入力を終了しました。")
        return

    if answer == prompt:
        print("入力が一致しました。")
    else:
        print("入力が一致しませんでした。")


if __name__ == "__main__":
    main()