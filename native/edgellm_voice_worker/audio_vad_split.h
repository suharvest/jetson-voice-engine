// SPDX-License-Identifier: MIT
//
// AudioVadSplitter — energy-based audio splitting at low-energy (~silence)
// boundaries. C++ port of `split_audio_into_chunks` from
// Qwen3-ASR/qwen_asr/inference/utils.py:246.
//
// Given a mono float32 PCM buffer and a target max chunk duration, returns
// a list of chunks whose concatenation reproduces the input sample-for-sample.
// Boundary positions are chosen near the target cut by argmin over a sliding
// 100ms absolute-amplitude sum window within ±search_expand_sec of the cut.
// Short tail chunks are zero-padded to min_chunk_sec (mirrors Python's
// MIN_ASR_INPUT_SECONDS=0.5 padding).
//
// Used in Phase 2 by the worker auto-segment path to avoid mid-word cuts
// that confuse the semantic dedup boundary detector.

#pragma once

#include <cstdint>
#include <vector>

namespace edgellm_voice_worker {

struct AudioChunk {
    std::vector<float> samples; // float32 PCM mono
    double offset_sec;          // start offset (sec) in the original buffer
};

class AudioVadSplitter {
public:
    explicit AudioVadSplitter(std::int32_t sample_rate = 16000,
                              double max_chunk_sec = 13.0,
                              double search_expand_sec = 2.0,
                              double min_window_ms = 100.0,
                              double min_chunk_sec = 0.5);

    // Split audio into chunks at low-energy boundaries.
    //
    // Concatenation of returned `samples` equals the input exactly (no
    // overlap, no gap) *before* min_chunk_sec tail padding. After padding,
    // a chunk's `samples.size()` may exceed (boundary - start); the
    // semantic offset_sec / next-chunk start are tracked from the pre-pad
    // boundary so timestamps remain correct.
    std::vector<AudioChunk> split(std::vector<float> const& pcm) const;

    // Boundary positions in original buffer (sample indices). Includes
    // the trailing total_len as last entry. Useful for parity testing.
    std::vector<std::int64_t> boundaries(std::vector<float> const& pcm) const;

    std::int32_t sample_rate() const { return sample_rate_; }
    double max_chunk_sec() const { return max_chunk_sec_; }
    double search_expand_sec() const { return search_expand_sec_; }
    double min_window_ms() const { return min_window_ms_; }
    double min_chunk_sec() const { return min_chunk_sec_; }

private:
    std::int32_t sample_rate_;
    double max_chunk_sec_;
    double search_expand_sec_;
    double min_window_ms_;
    double min_chunk_sec_;
};

} // namespace edgellm_voice_worker
