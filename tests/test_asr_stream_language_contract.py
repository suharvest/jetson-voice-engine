from __future__ import annotations

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
    njson = next((p for p in NJSON_CANDIDATES if p and (p / "nlohmann/json.hpp").exists()), None)
    if njson is None:
        raise RuntimeError("nlohmann/json.hpp is required; set NLOHMANN_JSON_INCLUDE")
    source = WORKER.read_text()
    start = source.index("std::optional<std::string> canonicalWorkerLanguage")
    end = source.index("bool assistantScaffoldEnabled()", start)
    helper = source[start:end]
    harness = tmp_path / "stream_language_harness.cpp"
    harness.write_text(
        "#include <cctype>\n#include <iostream>\n#include <optional>\n#include <stdexcept>\n"
        "#include <string>\n#include <unordered_map>\n#include <nlohmann/json.hpp>\n"
        "using Json = nlohmann::json;\n" + helper + "\n"
        "int main() {\n"
        "  std::cout << *canonicalWorkerLanguage(\"en-US\") << '\\n';\n"
        "  std::cout << (canonicalWorkerLanguage(\"auto\").has_value() ? \"bad\" : \"auto\") << '\\n';\n"
        "  try { canonicalWorkerLanguage(\"xx\"); std::cout << \"bad-invalid\\n\"; }"
        " catch (std::invalid_argument const&) { std::cout << \"invalid-before-lane\\n\"; }\n"
        "  Json forced = {{\"requests\", {{{\"messages\", {{{\"role\",\"user\"},{\"content\",{{{\"type\",\"audio\"}}}}}}}}}}};\n"
        "  applyWorkerLanguagePrompt(forced, canonicalWorkerLanguage(\"zh\"));\n"
        "  std::cout << forced.dump() << '\\n';\n"
        "  Json automatic = forced; automatic[\"requests\"][0][\"messages\"].erase(1);"
        "  automatic.erase(\"apply_chat_template\"); automatic.erase(\"add_generation_prompt\");\n"
        "  auto before = automatic.dump(); applyWorkerLanguagePrompt(automatic, std::nullopt);"
        "  std::cout << (before == automatic.dump() ? \"auto-json-same\" : \"bad-auto-json\") << '\\n';\n"
        "  Json continuation = automatic; continuation[\"requests\"][0][\"messages\"].push_back({{\"role\",\"assistant\"},{\"content\",\"prefix\"}});"
        "  try { applyWorkerLanguagePrompt(continuation, canonicalWorkerLanguage(\"en\")); std::cout << \"bad-prefix\"; }"
        " catch (std::invalid_argument const&) { std::cout << \"prefix-unique-assistant\\n\"; }\n"
        "  std::cout << prefixWarmupDecoded(\"hello\", canonicalWorkerLanguage(\"en\")) << '\\n';\n"
        "  std::cout << prefixWarmupDecoded(\"hello\", std::nullopt) << '\\n';\n"
        "}\n"
    )
    binary = tmp_path / "stream_language_harness"
    subprocess.run(["g++", "-std=c++17", "-I", str(njson), str(harness), "-o", str(binary)], check=True)
    return binary


def test_cpp_stream_language_contract(tmp_path):
    binary = _build_helper_harness(tmp_path)
    out = subprocess.run([str(binary)], text=True, capture_output=True, check=True).stdout.splitlines()
    print("HARNESS_OUTPUT_BEGIN")
    print("\n".join(out))
    print("HARNESS_OUTPUT_END")
    assert out[:3] == ["English", "auto", "invalid-before-lane"]
    forced = __import__("json").loads(out[3])
    messages = forced["requests"][0]["messages"]
    assert messages[-1] == {"role": "assistant", "content": "language Chinese<asr_text>"}
    assert forced["apply_chat_template"] is True
    assert forced["add_generation_prompt"] is False
    assert out[4:] == ["auto-json-same", "prefix-unique-assistant", "language English<asr_text>hello", "hello"]


def test_language_validation_and_builders_are_ordered_before_lane_or_decode():
    source = WORKER.read_text()
    begin = source.index("void handleBegin(Json const& input)")
    lane = source.index("gLaneMgr->acquire", begin)
    validation = source.index("canonicalWorkerLanguage", begin)
    assert validation < lane
    handle = source[begin:lane]
    assert "catch (std::invalid_argument const& e)" in handle
    assert "invalid_force_language" in handle
    finalize = source.index("Json buildFinalizeRequest", source.index("Json buildFinalizeRequest", begin) + 1)
    finalize_end = source.index("\n}", finalize)
    assert "applyWorkerLanguagePrompt(req, st.requestedLanguage)" in source[finalize:finalize_end]
    prefix = source.index("Json buildPrefixRequest", begin)
    prefix_end = source.index("\n}", prefix)
    assert "if (prefix.empty()) applyWorkerLanguagePrompt(req, st.requestedLanguage);" in source[prefix:prefix_end]
