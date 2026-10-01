from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "native/edgellm_voice_worker/qwen3_asr_worker.cpp"
NJSON_CANDIDATES = (
    Path(os.environ["NLOHMANN_JSON_INCLUDE"]) if os.environ.get("NLOHMANN_JSON_INCLUDE") else None,
    ROOT / "3rdParty/nlohmannJson/include",
)


def _build_helper_harness(tmp_path: Path) -> Path:
    njson = next((path for path in NJSON_CANDIDATES if path and (path / "nlohmann/json.hpp").exists()), None)
    if njson is None:
        raise RuntimeError("nlohmann/json.hpp is required; set NLOHMANN_JSON_INCLUDE for the candidate build")
    source = WORKER.read_text()
    match = re.search(r"bool assistantScaffoldEnabled\(\).*?\n}\n\n// ===========", source, re.S)
    assert match, "production helper block not found"
    helper = match.group(0).rsplit("\n\n// ===========", 1)[0]
    harness = tmp_path / "assistant_scaffold_harness.cpp"
    harness.write_text(
        '#include <cstdlib>\n#include <iostream>\n#include <string>\n'
        '#include <nlohmann/json.hpp>\nusing Json = nlohmann::json;\n'
        f"{helper}\n"
        "int main() { std::string line; while (std::getline(std::cin, line)) "
        "{ Json input = Json::parse(line); applyAssistantScaffold(input); "
        "std::cout << input.dump() << '\\n'; } }\n"
    )
    binary = tmp_path / "assistant_scaffold_harness"
    subprocess.run(
        ["g++", "-std=c++17", "-I", str(njson), str(harness), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


def _run(binary: Path, payloads: list[dict], enabled: bool) -> list[dict]:
    env = os.environ.copy()
    if enabled:
        env["OVS_ASR_ASSISTANT_SCAFFOLD"] = "1"
    else:
        env.pop("OVS_ASR_ASSISTANT_SCAFFOLD", None)
    result = subprocess.run(
        [str(binary)],
        input="".join(json.dumps(p) + "\n" for p in payloads),
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )
    return [json.loads(line) for line in result.stdout.splitlines()]


def _audio_payload() -> dict:
    return {
        "requests": [{"reference": "keep", "messages": [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}]}]}],
        "language": "auto",
        "other": {"keep": True},
    }


def test_cpp_helper_request_conversion(tmp_path):
    binary = _build_helper_harness(tmp_path)
    payload = _audio_payload()
    assert _run(binary, [payload], enabled=False)[0] == payload
    on = _run(binary, [payload], enabled=True)[0]
    assert on["requests"][0]["messages"][-1] == {"role": "assistant", "content": "language None<asr_text>"}
    assert on["apply_chat_template"] is True and on["add_generation_prompt"] is False
    assert on["language"] == "auto" and on["other"] == {"keep": True}


def test_cpp_helper_preserves_prefix_non_audio_and_mixed_batches(tmp_path):
    binary = _build_helper_harness(tmp_path)
    prefixed = _audio_payload()
    prefixed["requests"][0]["messages"].append({"role": "assistant", "content": "prefix"})
    warmup = _audio_payload()
    warmup.update({"apply_chat_template": True, "add_generation_prompt": True, "_ovs_asr_prefix_warmup": True})
    non_audio = {"requests": [{"messages": [{"role": "user", "content": "hello"}]}]}
    mixed = {"requests": [_audio_payload()["requests"][0], non_audio["requests"][0]]}
    out = _run(binary, [prefixed, warmup, non_audio, mixed], enabled=True)
    assert out[0] == prefixed
    warmup_expected = dict(warmup)
    warmup_expected.pop("_ovs_asr_prefix_warmup")
    assert out[1] == warmup_expected
    assert out[2] == non_audio and out[3] == mixed


def test_cpp_helper_changes_all_eligible_multi_requests(tmp_path):
    binary = _build_helper_harness(tmp_path)
    first = _audio_payload()["requests"][0]
    second = {"reference": "two", "messages": [{"role": "user", "content": [{"type": "audio", "audio": "y.wav"}]}]}
    out = _run(binary, [{"requests": [first, second], "language": "auto"}], enabled=True)[0]
    assert all(r["messages"][-1]["content"] == "language None<asr_text>" for r in out["requests"])
