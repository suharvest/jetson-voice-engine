# CMake integration notes for AudioVadSplitter (Phase 2)

**Do not apply during Phase 1.** Task #24 (worker source unification) is
modifying `native/edgellm_voice_worker/CMakeLists.txt` concurrently;
landing the additions below would create a merge conflict. After #24
settles, apply this diff.

## Proposed additions to `native/edgellm_voice_worker/CMakeLists.txt`

```cmake
# --- AudioVadSplitter (energy-based audio chunk splitter) ---------------
add_library(audioVadSplit STATIC
    audio_vad_split.cpp
)
target_include_directories(audioVadSplit PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(audioVadSplit PUBLIC cxx_std_17)

# Standalone test binary (WAV in → boundaries JSON out + sanity checks)
add_executable(test_audio_vad_split tests/test_audio_vad_split.cpp)
target_link_libraries(test_audio_vad_split PRIVATE audioVadSplit)

# Link into the worker (Phase 2 — worker source consumes AudioVadSplitter)
target_link_libraries(qwen3_asr_worker PRIVATE audioVadSplit)
```

## Standalone build (Phase 1, no CMake)

Already verified on macOS arm64 with:

```bash
cd native/edgellm_voice_worker
clang++ -std=c++17 -O2 -Wall -Wextra \
    -o build_vad/test_audio_vad_split \
    audio_vad_split.cpp tests/test_audio_vad_split.cpp
```

Linux/Jetson equivalent (Phase 2 host build):
```bash
g++ -std=c++17 -O2 -Wall -Wextra \
    -o build/test_audio_vad_split \
    audio_vad_split.cpp tests/test_audio_vad_split.cpp
```

No external deps (no nlohmann/json, no kissfft) — pure stdlib, by design.

## Verification post-integration

After applying the diff and rebuilding:

```bash
cd native/edgellm_voice_worker/build
./test_audio_vad_split ../../../docs/audio-evidence/zh-long-04-2026-05-13.wav 5.5 2.0 /tmp/vad.json
python3 ../../../scripts/test_audio_vad_parity.py \
    --wav ../../../docs/audio-evidence/zh-long-04-2026-05-13.wav \
    --binary ./test_audio_vad_split \
    --max-chunk-sec 5.5 --search-expand-sec 2.0
```

Expected: `PASS: boundaries match within tolerance` (max diff = 0 samples
on zh-long-04 — measured Phase 1, see `vad-aligned-segmentation-design.md`).
