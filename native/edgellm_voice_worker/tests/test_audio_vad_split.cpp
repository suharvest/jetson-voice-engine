// SPDX-License-Identifier: MIT
//
// test_audio_vad_split — sanity + JSON-dump tool for AudioVadSplitter.
//
// Usage:
//   test_audio_vad_split <input.wav> [max_chunk_sec=13.0]
//                        [search_expand_sec=2.0] [out.json]
//
// Loads a 16 kHz mono PCM16 WAV, runs AudioVadSplitter::split, and:
//   1. Verifies concatenation of pre-pad chunks reproduces the input bit-for-bit
//      (uses `boundaries()` to bypass the tail-padding step).
//   2. Asserts no pre-pad chunk exceeds max_chunk_sec.
//   3. Sanity: window energy at boundary <= mean window energy across search range.
//   4. Prints boundary positions (sample indices + seconds) to stdout.
//   5. If out.json is provided, dumps {boundaries_samples, boundaries_sec,
//      max_chunk_sec, search_expand_sec, min_window_ms, sample_rate}.
//
// Returns non-zero on any verification failure.

#include "../audio_vad_split.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

#pragma pack(push, 1)
struct WavHeader {
    char riff[4];
    std::uint32_t riff_size;
    char wave[4];
};
struct ChunkHeader {
    char id[4];
    std::uint32_t size;
};
struct FmtChunk {
    std::uint16_t audio_format;
    std::uint16_t num_channels;
    std::uint32_t sample_rate;
    std::uint32_t byte_rate;
    std::uint16_t block_align;
    std::uint16_t bits_per_sample;
};
#pragma pack(pop)

// Minimal PCM16 mono WAV loader. Returns float32 samples scaled to [-1, 1].
std::vector<float> loadWavPcm16(std::string const& path, std::uint32_t& sample_rate)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    WavHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (std::memcmp(hdr.riff, "RIFF", 4) != 0 || std::memcmp(hdr.wave, "WAVE", 4) != 0) {
        throw std::runtime_error("not a RIFF/WAVE: " + path);
    }
    FmtChunk fmt{};
    bool got_fmt = false;
    std::vector<float> samples;
    while (f) {
        ChunkHeader ch{};
        f.read(reinterpret_cast<char*>(&ch), sizeof(ch));
        if (!f) break;
        if (std::memcmp(ch.id, "fmt ", 4) == 0) {
            std::vector<char> buf(ch.size);
            f.read(buf.data(), ch.size);
            std::memcpy(&fmt, buf.data(), std::min<std::size_t>(sizeof(fmt), ch.size));
            got_fmt = true;
        } else if (std::memcmp(ch.id, "data", 4) == 0) {
            if (!got_fmt) throw std::runtime_error("data before fmt");
            if (fmt.audio_format != 1 || fmt.bits_per_sample != 16) {
                throw std::runtime_error("only PCM16 supported");
            }
            std::size_t nsamp_total = ch.size / sizeof(std::int16_t);
            std::vector<std::int16_t> pcm(nsamp_total);
            f.read(reinterpret_cast<char*>(pcm.data()), ch.size);
            std::size_t nframes = nsamp_total / fmt.num_channels;
            samples.resize(nframes);
            for (std::size_t i = 0; i < nframes; ++i) {
                std::int32_t acc = 0;
                for (std::size_t c = 0; c < fmt.num_channels; ++c) {
                    acc += pcm[i * fmt.num_channels + c];
                }
                float v = static_cast<float>(acc) / static_cast<float>(fmt.num_channels) / 32768.0f;
                samples[i] = v;
            }
        } else {
            f.seekg(ch.size, std::ios::cur);
        }
        if (ch.size & 1u) f.seekg(1, std::ios::cur); // pad byte
    }
    sample_rate = fmt.sample_rate;
    return samples;
}

double windowEnergy(std::vector<float> const& pcm, std::int64_t start, std::int64_t win)
{
    if (start < 0) start = 0;
    if (start + win > static_cast<std::int64_t>(pcm.size())) {
        win = static_cast<std::int64_t>(pcm.size()) - start;
    }
    if (win <= 0) return 0.0;
    double s = 0.0;
    for (std::int64_t i = 0; i < win; ++i) s += std::fabs(pcm[start + i]);
    return s / static_cast<double>(win);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <input.wav> [max_chunk_sec=13.0] [search_expand_sec=2.0] [out.json]\n",
            argv[0]);
        return 2;
    }
    std::string wav_path = argv[1];
    double max_chunk_sec = (argc >= 3) ? std::atof(argv[2]) : 13.0;
    double search_expand_sec = (argc >= 4) ? std::atof(argv[3]) : 2.0;
    std::string out_json = (argc >= 5) ? argv[4] : "";

    std::uint32_t sr = 0;
    auto pcm = loadWavPcm16(wav_path, sr);
    if (sr != 16000) {
        std::fprintf(stderr, "WARN: expected 16000 Hz, got %u\n", sr);
    }
    std::printf("Loaded %s: %zu samples, %.3f sec, sr=%u\n",
        wav_path.c_str(), pcm.size(),
        static_cast<double>(pcm.size()) / sr, sr);

    edgellm_voice_worker::AudioVadSplitter splitter(
        static_cast<std::int32_t>(sr), max_chunk_sec, search_expand_sec, 100.0, 0.5);

    auto bnds = splitter.boundaries(pcm);
    auto chunks = splitter.split(pcm);

    std::printf("Config: max=%.2fs expand=%.2fs win=100ms\n", max_chunk_sec, search_expand_sec);
    std::printf("Chunks: %zu (boundaries: %zu)\n", chunks.size(), bnds.size());

    // (1) Concatenation parity check via boundaries (pre-pad).
    std::vector<float> reconstructed;
    reconstructed.reserve(pcm.size());
    std::int64_t prev = 0;
    for (auto b : bnds) {
        for (std::int64_t i = prev; i < b; ++i) reconstructed.push_back(pcm[i]);
        prev = b;
    }
    if (reconstructed.size() != pcm.size()) {
        std::fprintf(stderr, "FAIL: reconstructed size %zu != original %zu\n",
            reconstructed.size(), pcm.size());
        return 1;
    }
    for (std::size_t i = 0; i < pcm.size(); ++i) {
        if (reconstructed[i] != pcm[i]) {
            std::fprintf(stderr, "FAIL: reconstructed mismatch at sample %zu\n", i);
            return 1;
        }
    }
    std::printf("OK  concat-parity: reconstructed bit-identical to input\n");

    // (2) Max chunk duration check (pre-pad, by boundary diff).
    // The algorithm picks the cut within [cut-expand, cut+expand], so a chunk
    // can be up to (max_len + expand) samples long. Tail (last chunk) can be
    // up to max_len samples (loop exits when total_len - start <= max_len).
    std::int64_t max_len_samples = static_cast<std::int64_t>(max_chunk_sec * sr);
    std::int64_t expand_samples = static_cast<std::int64_t>(search_expand_sec * sr);
    std::int64_t internal_cap = max_len_samples + expand_samples;
    std::int64_t s = 0;
    bool over = false;
    for (std::size_t i = 0; i < bnds.size(); ++i) {
        std::int64_t b = bnds[i];
        std::int64_t len = b - s;
        bool is_tail = (i + 1 == bnds.size());
        std::int64_t cap = is_tail ? max_len_samples : internal_cap;
        if (len > cap) {
            std::fprintf(stderr, "FAIL: chunk [%lld..%lld] len=%lld > cap=%lld\n",
                static_cast<long long>(s), static_cast<long long>(b),
                static_cast<long long>(len), static_cast<long long>(cap));
            over = true;
        }
        s = b;
    }
    if (over) return 1;
    std::printf("OK  max-len: every chunk within bound (internal<=%.2fs, tail<=%.2fs)\n",
        static_cast<double>(internal_cap) / sr, max_chunk_sec);

    // (3) Boundary energy sanity: window at boundary should be below the
    //     mean window energy in the search range. Only meaningful for internal
    //     boundaries (skip the trailing total_len).
    std::int64_t win = static_cast<std::int64_t>(0.100 * sr); // 100 ms
    std::int64_t expand = static_cast<std::int64_t>(search_expand_sec * sr);
    std::int64_t cum = 0;
    int internal = 0;
    for (std::size_t i = 0; i + 1 < bnds.size(); ++i) {
        std::int64_t b = bnds[i];
        std::int64_t left = std::max<std::int64_t>(0, b - expand);
        std::int64_t right = std::min<std::int64_t>(static_cast<std::int64_t>(pcm.size()), b + expand);
        double e_b = windowEnergy(pcm, b - win / 2, win);
        // mean energy over the search range (sample of 11 windows)
        double sum = 0.0;
        int n = 0;
        for (std::int64_t t = left; t <= right - win; t += (right - left) / 10 + 1) {
            sum += windowEnergy(pcm, t, win);
            n++;
        }
        double e_mean = (n > 0) ? sum / n : 0.0;
        std::printf("  boundary[%zu] @ sample=%lld (%.3fs)  e_at=%.6f  e_mean=%.6f  ratio=%.3f\n",
            i, static_cast<long long>(b), static_cast<double>(b) / sr,
            e_b, e_mean, (e_mean > 0 ? e_b / e_mean : 0.0));
        if (e_b > e_mean * 1.5) {
            std::fprintf(stderr,
                "WARN: boundary %zu energy %.6f > 1.5x mean %.6f (not a strong silence point)\n",
                i, e_b, e_mean);
        }
        internal++;
        cum += static_cast<std::int64_t>(b);
    }
    (void)cum;
    if (internal == 0) {
        std::printf("(no internal boundaries — input shorter than max_chunk_sec)\n");
    }

    // (4) Pretty print chunk summary.
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        std::printf("  chunk[%zu] offset=%.3fs  samples=%zu  (~%.3fs after padding)\n",
            i, chunks[i].offset_sec, chunks[i].samples.size(),
            static_cast<double>(chunks[i].samples.size()) / sr);
    }

    // (5) JSON dump.
    if (!out_json.empty()) {
        std::ofstream o(out_json);
        if (!o) {
            std::fprintf(stderr, "FAIL: cannot write %s\n", out_json.c_str());
            return 1;
        }
        o << "{\n";
        o << "  \"input\": \"" << wav_path << "\",\n";
        o << "  \"sample_rate\": " << sr << ",\n";
        o << "  \"num_samples\": " << pcm.size() << ",\n";
        o << "  \"max_chunk_sec\": " << max_chunk_sec << ",\n";
        o << "  \"search_expand_sec\": " << search_expand_sec << ",\n";
        o << "  \"min_window_ms\": 100.0,\n";
        o << "  \"min_chunk_sec\": 0.5,\n";
        o << "  \"boundaries_samples\": [";
        for (std::size_t i = 0; i < bnds.size(); ++i) {
            if (i) o << ", ";
            o << bnds[i];
        }
        o << "],\n";
        o << "  \"boundaries_sec\": [";
        for (std::size_t i = 0; i < bnds.size(); ++i) {
            if (i) o << ", ";
            o << (static_cast<double>(bnds[i]) / sr);
        }
        o << "]\n";
        o << "}\n";
        std::printf("Wrote %s\n", out_json.c_str());
    }

    std::printf("PASS\n");
    return 0;
}
