"""The AI file-edit helper: prompt building, reply parsing, style and sanity checks."""
from __future__ import annotations

from editing import assistant


def test_prompt_contains_instruction_path_and_the_whole_file():
    original = "lr = 2e-4\n" * 500  # far longer than a chat message would keep
    messages = assistant.build_edit_messages("configs/train.py", original, "  set the learning rate\nto 1e-4 ")
    body = messages[0]["content"]
    assert "Instruction: set the learning rate to 1e-4" in body
    assert "File: configs/train.py" in body
    assert original in body, "the file must never be truncated"


def test_extracts_the_largest_block_and_the_explanation():
    reply = (
        "Lowered the learning rate. For example `x = 1` or:\n```\nlr=1\n```\n"
        "Full file:\n```python\nimport os\nlr = 1e-4\nepochs = 3\n```\n"
    )
    result = assistant.extract_file_from_reply(reply)
    assert result["content"] == "import os\nlr = 1e-4\nepochs = 3\n"
    assert result["explanation"].startswith("Lowered the learning rate.")


def test_reply_without_a_code_block_is_rejected():
    assert assistant.extract_file_from_reply("Sure, change lr to 1e-4.") is None
    assert assistant.extract_file_from_reply("") is None
    assert assistant.extract_file_from_reply(None) is None


def test_original_line_endings_and_final_newline_are_kept():
    assert assistant.match_original_style("a\r\nb\r\n", "a\nc\n") == "a\r\nc\r\n"
    assert assistant.match_original_style("a\nb", "a\nc\n") == "a\nc"
    assert assistant.match_original_style("a\nb\n", "a\nc") == "a\nc\n"


def test_assess_flags_unchanged_and_suspiciously_short_results():
    big = "x = 1\n" * 200
    assert assistant.assess(big, big)["unchanged"] is True
    assert assistant.assess(big, big + "y = 2\n")["unchanged"] is False
    assert assistant.assess(big, "x = 1\n# ...rest unchanged\n")["suspectTruncation"] is True
    assert assistant.assess("a = 1\n", "a = 2\n")["suspectTruncation"] is False
