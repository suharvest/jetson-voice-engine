// SPDX-License-Identifier: MIT
//
// AudioVadSplitter implementation — see audio_vad_split.h.
//
// Mirror of qwen_asr/inference/utils.py:246 split_audio_into_chunks.
// Operation-for-operation parity is required for the boundary search:
//   - cut = start + max_len
//   - left = max(start, cut - expand), right = min(total_len, cut + expand)
//   - if right - left <= win: boundary = cut
//   - else:
//       seg_abs = |seg|
//       window_sums = convolve(seg_abs, ones(win), 'valid')   # length = right-left-win+1
//       min_pos = argmin(window_sums)
//       local = seg_abs[min_pos : min_pos+win]
//       inner = argmin(local)
//       boundary = left + min_pos + inner
//   - boundary = clamp(boundary, start+1, total_len)
//
// Argmin semantics: numpy.argmin returns the *first* minimum on ties.
// std::min_element matches this (returns iterator to first equal min).

#include "audio_vad_split.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace edgellm_voice_worker {

namespace {

// Sliding-window absolute-amplitude sums equivalent to
// numpy.convolve(seg_abs, np.ones(win), mode='valid').
// Output length: (seg.size() - win + 1). Uses an online running sum to
// avoid an O(N*W) cost.
std::vector<float> windowSumsAbs(float const* seg, std::size_t n, std::size_t win)
{
    if (n < win) return {};
    std::vector<float> out(n - win + 1);
    double running = 0.0;
    for (std::size_t i = 0; i < win; ++i) {
        running += std::fabs(seg[i]);
    }
    out[0] = static_cast<float>(running);
    for (std::size_t i = win; i < n; ++i) {
        running += std::fabs(seg[i]);
        running -= std::fabs(seg[i - win]);
        // Clamp accumulated numerical drift at zero (sums of |x| are >= 0).
        if (running < 0.0) running = 0.0;
        out[i - win + 1] = static_cast<float>(running);
    }
    return out;
}

// numpy.argmin with first-minimum tie-break.
std::size_t argminFirst(std::vector<float> const& v)
{
    std::size_t best = 0;
    float bestv = v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i] < bestv) {
            bestv = v[i];
            best = i;
        }
    }
    return best;
}

std::size_t argminFirstAbs(float const* p, std::size_t n)
{
    std::size_t best = 0;
    float bestv = std::fabs(p[0]);
    for (std::size_t i = 1; i < n; ++i) {
        float a = std::fabs(p[i]);
        if (a < bestv) {
            bestv = a;
            best = i;
        }
    }
    return best;
}

} // namespace

AudioVadSplitter::AudioVadSplitter(std::int32_t sample_rate,
                                   double max_chunk_sec,
                                   double search_expand_sec,
                                   double min_window_ms,
                                   double min_chunk_sec)
    : sample_rate_(sample_rate),
      max_chunk_sec_(max_chunk_sec),
      search_expand_sec_(search_expand_sec),
      min_window_ms_(min_window_ms),
      min_chunk_sec_(min_chunk_sec)
{}

std::vector<std::int64_t> AudioVadSplitter::boundaries(std::vector<float> const& pcm) const
{
    std::vector<std::int64_t> out;
    const std::int64_t total_len = static_cast<std::int64_t>(pcm.size());
    const double total_sec = static_cast<double>(total_len) / static_cast<double>(sample_rate_);

    if (total_sec <= max_chunk_sec_) {
        out.push_back(total_len);
        return out;
    }

    const std::int64_t max_len = static_cast<std::int64_t>(max_chunk_sec_ * sample_rate_);
    const std::int64_t expand = static_cast<std::int64_t>(search_expand_sec_ * sample_rate_);
    std::int64_t win = static_cast<std::int64_t>((min_window_ms_ / 1000.0) * sample_rate_);
    if (win < 4) win = 4;

    std::int64_t start = 0;

    while ((total_len - start) > max_len) {
        std::int64_t cut = start + max_len;
        std::int64_t left = std::max<std::int64_t>(start, cut - expand);
        std::int64_t right = std::min<std::int64_t>(total_len, cut + expand);

        std::int64_t boundary;
        if (right - left <= win) {
            boundary = cut;
        } else {
            std::size_t seg_n = static_cast<std::size_t>(right - left);
            float const* seg = pcm.data() + left;
            auto sums = windowSumsAbs(seg, seg_n, static_cast<std::size_t>(win));
            std::size_t min_pos = argminFirst(sums);
            std::size_t inner = argminFirstAbs(seg + min_pos, static_cast<std::size_t>(win));
            boundary = left + static_cast<std::int64_t>(min_pos) + static_cast<std::int64_t>(inner);
        }

        if (boundary < start + 1) boundary = start + 1;
        if (boundary > total_len) boundary = total_len;

        out.push_back(boundary);
        start = boundary;
    }

    out.push_back(total_len);
    return out;
}

std::vector<AudioChunk> AudioVadSplitter::split(std::vector<float> const& pcm) const
{
    std::vector<AudioChunk> chunks;
    auto bnds = boundaries(pcm);
    const std::int64_t total_len = static_cast<std::int64_t>(pcm.size());

    std::int64_t start = 0;
    double offset_sec = 0.0;
    for (std::int64_t b : bnds) {
        if (b < start) b = start; // defensive
        if (b > total_len) b = total_len;
        AudioChunk c;
        c.offset_sec = offset_sec;
        c.samples.assign(pcm.begin() + start, pcm.begin() + b);
        offset_sec += static_cast<double>(b - start) / static_cast<double>(sample_rate_);
        start = b;
        chunks.push_back(std::move(c));
    }

    // Zero-pad short chunks to min_chunk_sec (mirrors Python MIN_ASR_INPUT_SECONDS).
    const std::size_t min_len = static_cast<std::size_t>(min_chunk_sec_ * sample_rate_);
    for (auto& c : chunks) {
        if (c.samples.size() < min_len) {
            c.samples.resize(min_len, 0.0f);
        }
    }

    return chunks;
}

} // namespace edgellm_voice_worker
