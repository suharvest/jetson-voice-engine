/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audio_vad_split.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "common/trtUtils.h"
#include "mel_extractor.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "requestFileParser.h"
#include "runtime/llmInferenceSpecDecodeRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/slotPool.h"
#include "tokenizer/tokenizer.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <optional>
#include <poll.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

namespace
{
struct Args
{
    std::string engineDir;
    std::string multimodalEngineDir;
    std::string melSettingsPath;     //!< whisper_feature_extractor.json (optional, enables PCM input)
    std::string melFiltersPath;      //!< mel_filters.bin (optional, enables PCM input)
    int32_t maxSlots{4};             //!< ASR decoder slot-pool size (D1 concurrency). Default 4.
    bool debug{false};
};

// ---------------------------------------------------------------------------
// M3 streaming-ASR worker (design doc §15 v5.3 — step 2 mechanism + step 4
// max_input_len enforcement + transparent auto-segmentation + cleanup).
//
// The worker is event-driven on stdin. Each line is a JSON object:
//
//   {"event":"begin","id":<sid>,"sample_rate":16000,
//    "chunk_size_sec":0.5,"unfixed_chunk_num":2,"unfixed_token_num":5,
//    "force_language":null,"context":""}
//   {"event":"chunk","id":<sid>,"mel_path":<path>,"audio_sec":<float>,"last":false}
//   {"event":"end","id":<sid>}
//
// Per design §15.5.2, when projected per-hop input length would exceed the
// thinker engine's max_input_len, the worker transparently rotates the
// session: decodes the current mel as a segment, appends its text to a
// session-level fullText accumulator, emits a `segment_rotation` event
// (driver MUST trim audio_accum to last `carryover_sec` and continue), and
// resets internal hop state. The client sees exactly ONE `final` event at
// end-of-stream, carrying the concatenation of all segment texts.
//
// `audio_sec` on `chunk` events is REQUIRED for cumulative-mel callers that
// want auto-segmentation; when absent, behavior reduces to step 2 spike
// (no enforcement, hop-id increments forever).
//
// Lines with no `event` field hit the legacy one-shot handler — required
// for byte-equivalent backward compatibility with M2 callers.
//
// Slot-pool worker (D1 Step 3): up to --max_slots concurrent logical sessions,
// each routed by its `id` to an independent decoder slot (own runtime +
// IExecutionContext + KV state + CUDA stream, sharing one base engine). A
// `begin` for a NEW id when every slot is busy is refused with
// {"event":"error","error":"pool_saturated","status":4429}. A `begin` for an
// id that already owns a slot reuses it (idempotent re-begin). With
// --max_slots=1 the behaviour reduces to the prior single-session worker
// (one slot; a second concurrent id gets pool_saturated).
//
// NOTE: the worker drives a SINGLE stdin poll() loop (one OS thread), so events
// are processed serially even when several sessions are interleaved on the
// stream; the routing tables are still mutex-guarded for forward compatibility
// with a future thread-per-connection front-end.
// ---------------------------------------------------------------------------

// §15.5.1 — engine cap is max_input_len=256 (rebuilt P1, highperf-v2/
// asr_thinker_full_fp8embed). Per-hop input budget breakdown
// (audio_tokens + prompt_overhead + audio_bos/eos). 13 audio tokens/s is the
// Spike A measurement. Prompt overhead ~32 tokens for the highperf chat
// template. Reserve 8 tokens as safety margin.
constexpr int32_t kEngineMaxInputLen = 256;
constexpr int32_t kPromptOverheadTokens = 32;
constexpr int32_t kAudioBosEosTokens = 2;
constexpr int32_t kInputSafetyMargin = 8;
constexpr double kAudioTokensPerSec = 13.0;
//! Hard refuse if any single chunk's audio_sec exceeds this. Set above the
//! per-hop cap so auto-segmentation has a window to fire first. Engine math
//! at max_input_len=256: audio cap = (256 - 8 - 32 - 2) / 13 ≈ 16.5 s; we
//! hard-refuse at 15.0 s to leave ~1.5 s margin for prompt/jitter and reserve
//! the upper band for the auto-segment path.
constexpr double kSingleChunkHardLimitSec = 15.0;
constexpr double kCarryoverSec = 1.0;              //!< Segment boundary keeps last N seconds of audio.
constexpr int64_t kIdleTimeoutMs = 30000;          //!< Force-close session after this much inactivity.

//! P2 VAD-aligned auto-segment params (design doc §15.5.2 VAD revision).
//! Trigger BEFORE the hard 15 s refuse so VAD has a search window. When
//! triggered, the splitter looks for a low-energy boundary within
//! ±kVadSearchExpandSec of the target cut at kVadMaxChunkSec. If a clean
//! boundary is found inside [trigger, hard-limit], rotate at that boundary
//! and carry over the audio that follows it. Otherwise fall through to the
//! hard refuse path (preserves old behavior for adversarial inputs).
constexpr double kVadTriggerSec = 12.0;           //!< Earliest audio_sec at which VAD considers rotating.
constexpr double kVadMaxChunkSec = 13.0;          //!< Target cut point inside the VAD splitter (~2 s margin below hard limit).
constexpr double kVadSearchExpandSec = 2.0;       //!< Symmetric search window around the target cut.
constexpr double kVadMinChunkSec = 0.5;           //!< Pad short tail chunks to this length (mirrors MIN_ASR_INPUT_SECONDS).

inline int32_t projectInputTokens(double audioSec)
{
    auto const audioTokens = static_cast<int32_t>(std::ceil(audioSec * kAudioTokensPerSec));
    return audioTokens + kPromptOverheadTokens + kAudioBosEosTokens;
}

inline bool wouldOverflow(double audioSec)
{
    return projectInputTokens(audioSec) > (kEngineMaxInputLen - kInputSafetyMargin);
}

struct AsrSessionState
{
    std::string sessionId;                                   //!< Stable ID emitted by the client at begin.
    std::chrono::steady_clock::time_point lastActivity{};    //!< Updated on every chunk/end touching the slot.
    bool active{false};                                      //!< True between begin and end.

    // §15.1 streaming state — populated at begin.
    double sampleRate{16000.0};
    double chunkSizeSec{0.5};
    int32_t unfixedChunkNum{2};
    int32_t unfixedTokenNum{5};
    int32_t maxDecodeTokensPerHop{64};   //!< Max new tokens per streaming hop (mirrors official).
    std::string forceLanguage{};      //!< Empty = no force; e.g. "Chinese".
    std::string context{};            //!< System-prompt context.

    // Per-hop accumulator: precomputed mel frames stored as concatenated
    // fp16 bytes. Each chunk-event mel payload is appended verbatim. The
    // mel tensor shape is [1, mel_bins, T_total]; we track T_total.
    std::vector<uint8_t> melAccumBytes;
    int32_t melBins{128};            //!< Mel bin count; locked from first chunk.
    int32_t melFrames{0};            //!< Cumulative frame count of melAccumBytes.

    // Hop-trigger bookkeeping: number of mel frames at the time of last hop.
    int32_t melFramesAtLastHop{0};

    // Decoded text state.
    std::string rawDecoded{};        //!< Mirrors official state._raw_decoded.
    int32_t chunkId{0};              //!< Hop counter (mirrors official chunk_id).

    // Session-level accumulator for auto-segmentation (Step 4).
    std::string fullText{};

    // Audio seconds in the most-recent chunk's mel (driver-reported). Used
    // for max_input_len enforcement and the auto-segmentation projection.
    double lastAudioSec{0.0};
    // Number of internal segment rotations during this session (debug only).
    int32_t segmentCount{0};

    // P2 VAD-aligned auto-segment state.
    //
    // When the chunk event delivers PCM (`pcm_b64`), we accumulate the raw
    // float32 mono PCM so that at rotation time we can run AudioVadSplitter
    // to find a low-energy boundary and recommend a precise carry-over to
    // the driver. For mel_path-only sessions, pcmAccum stays empty and the
    // worker falls back to the fixed-carryover policy (kCarryoverSec).
    std::vector<float> pcmAccum;
};

//! Reset session to inactive state and clear all accumulators. Used by both
//! the normal end-of-session path and every error-cleanup path (§15.6 step 4).
void freeSession(AsrSessionState& session)
{
    session = AsrSessionState{};
}

// ===========================================================================
// ASR slot-pool (Phase D-1, Step 2 scaffolding).
//
// C-optimized concurrency layout:
//   * ONE shared audio encoder context, serialized by gEncoderMutex. Measured
//     N=4 encoder GPU utilization is only ~16%, so a single time-sliced
//     encoder context is the right trade-off for now (the encoder lives inside
//     each runtime today; gEncoderMutex is the conservative guard until
//     audio-runner multi-context safety is separately proven — spec §2 / R2).
//   * N decoder/"thinker" slots, each with its OWN
//     LLMInferenceSpecDecodeRuntime + IExecutionContext + KV/cache manager
//     state + runtime tensors + CUDA stream, all SHARING one deserialized base
//     ICudaEngine (weight memory paid once; fork shared-engine constructor).
//
// Step 2 builds the pool data structures + initialization ONLY. Request
// routing (begin/chunk/end -> slot) stays on the single-session path; Step 3
// wires acquire/release into dispatch.
//
// Reference: MossTtsNanoSlot + mPoolMutex/mSlots/acquirePoolSlot/releasePoolSlot
// in cpp/runtime/mossTtsNanoRuntime.{h,cpp} of the TensorRT-Edge-LLM fork.
// ===========================================================================

//! Per-slot decoder state. One AsrSlot == one logical ASR session capacity.
//! Owns an independent runtime + CUDA stream + the full per-session token/mel
//! state that used to live in the single `AsrSessionState session`. Two live
//! sessions therefore never share KV state, runtime tensors, or accumulators.
struct AsrSlot
{
    int32_t slotId{-1};
    std::atomic<bool> inUse{false};

    //! Per-slot decoder runtime. Slot 0 deserializes the base engine; slots
    //! 1..N-1 share slot 0's ICudaEngine via the fork shared-engine path.
    std::unique_ptr<rt::LLMInferenceSpecDecodeRuntime> thinker;

    //! Per-slot CUDA stream. Each slot's handleRequest passes THIS stream so
    //! independent execution contexts can (eventually) overlap on the GPU.
    cudaStream_t stream{nullptr};

    //! Per-slot streaming/inference state (moved out of the single session).
    AsrSessionState session{};

    AsrSlot() = default;
    // Non-copyable / non-movable: holds std::atomic + CUDA stream + unique runtime.
    AsrSlot(AsrSlot const&) = delete;
    AsrSlot& operator=(AsrSlot const&) = delete;
};

//! Number of decoder slots. Default 4; overridable via --max_slots CLI.
int32_t gMaxSlots{4};

//! D-2.5: the slot vector, capacity, sessionId→slot routing map (+its mutex),
//! CAS acquire, bind/unbind/lookup, saturation are now provided by the generic
//! SlotPool<AsrSlot> template (cpp/runtime/slotPool.h in the TRT-Edge-LLM fork,
//! included cross-repo via the existing runtime/ include path). What stays in
//! this worker: runtime/stream construction, the AsrSessionState reset (done
//! BEFORE release / on begin — RISK POINTs 2 & 3), the single poll thread.
namespace rt_slotpool = tensorrt_edge_llm::runtime;
std::unique_ptr<rt_slotpool::SlotPool<AsrSlot>> gPool;

//! Serializes the SHARED audio encoder path across slots (spec §2 / R2).
//! Held only around the encoder portion of a hop once Step 3 wires routing.
std::mutex gEncoderMutex;

//! Look up the slot index bound to @p sessionId. Returns -1 if no live mapping
//! exists (session was never begun, already ended, or timed out).
int32_t lookupSlot(std::string const& sessionId)
{
    return gPool->lookup(sessionId);
}

//! Bind sessionId → slotId. Caller must already hold a freshly acquired slot.
void bindSession(std::string const& sessionId, int32_t slotId)
{
    gPool->bind(sessionId, slotId);
}

//! Remove the sessionId → slot mapping (end / timeout / error cleanup).
void unbindSession(std::string const& sessionId)
{
    gPool->unbind(sessionId);
}

//! Release a slot back to the free list and reset reusable state.
//!
//! RISK POINT 3 — session reset MUST happen BEFORE the slot is marked free, so
//! a slot handed back out by a subsequent acquire never carries a prior
//! session's token/mel/KV accumulators. The generic SlotPool::release() clears
//! `inUse` ONLY, so this wrapper performs freeSession() FIRST, then release().
//! Ordering is preserved exactly as the legacy releaseSlot (freeSession →
//! mark-free).
//!
//! [decoder KV/runtime reset] The streaming path drives decode through
//! runStreamingHop → runtime.handleRequest with NO explicit
//! beginAsrSession/endAsrSession, so the fork runtime holds no per-session KV
//! state that survives between handleRequest calls — each hop rebuilds its
//! prefill from the prefix + mel. There is therefore no in-runtime KV residue
//! to clear here; the freeSession() below resets all worker-side accumulators.
//! [confirmed no residue]
void releaseSlot(int32_t slotId)
{
    AsrSlot* slot = gPool->get(slotId);
    if (slot == nullptr)
    {
        return;
    }
    freeSession(slot->session); // RISK POINT 3: reset BEFORE marking free
    gPool->release(slotId);     // clears inUse only
}

//! Build the N-slot decoder pool sharing one deserialized base ICudaEngine.
//!
//! Slot 0 takes the EXISTING path-based vanilla constructor (deserializes the
//! base engine once + loads its own multimodal/audio encoder + tokenizer +
//! embedding). Slots 1..N-1 use the fork shared-engine constructor overload,
//! passing slot 0's base engine via getBaseEngine() so the heavy LLM weights
//! are paid for ONCE while each slot keeps its own IExecutionContext, KV/cache
//! state, runtime tensors, shared-exec-context memory, and CUDA stream.
//!
//! @param engineDir            thinker/decoder engine dir (--engineDir)
//! @param multimodalEngineDir  multimodal/audio engine dir (--multimodalEngineDir)
//! @param loraWeightsMap       LoRA weights map (passed through to each runtime)
//! @param enableCudaGraph      capture decoding CUDA graph per slot when true
//!
//! NOTE (Step 2): called from main() to replace the single-runtime init. The
//! single-session dispatch path then drives slot 0 only (Step 3 fans out).
void initSlotPool(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string>& loraWeightsMap, bool enableCudaGraph)
{
    int32_t const n = std::max(1, gMaxSlots);
    gPool = std::make_unique<rt_slotpool::SlotPool<AsrSlot>>(n);
    auto& slots = gPool->slots();

    for (int32_t i = 0; i < n; ++i)
    {
        auto slot = std::make_unique<AsrSlot>();
        slot->slotId = i;
        slot->inUse.store(false);
        CUDA_CHECK(cudaStreamCreate(&slot->stream));

        if (i == 0)
        {
            // Slot 0: existing path-based vanilla constructor deserializes the
            // base engine once and loads multimodal/audio + tokenizer + embeds.
            slot->thinker = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(
                engineDir, multimodalEngineDir, loraWeightsMap, slot->stream);
        }
        else
        {
            // Slots 1..N-1: SHARE slot 0's deserialized base ICudaEngine. Each
            // still loads its own multimodal/audio runner, tokenizer, embedding
            // table, and allocates its own IExecutionContext + KV/cache + shared
            // execution-context memory — only base LLM weights are shared.
            std::shared_ptr<nvinfer1::ICudaEngine> sharedBase = slots[0]->thinker->getBaseEngine();
            slot->thinker = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(
                sharedBase, engineDir, multimodalEngineDir, loraWeightsMap, slot->stream);
        }

        if (enableCudaGraph && !slot->thinker->captureDecodingCUDAGraph(slot->stream))
        {
            LOG_WARNING("CUDA graph capture failed for ASR slot %d, proceeding without.", i);
        }
        slots.push_back(std::move(slot));
    }
}

//! Destroy all slot CUDA streams + runtimes. Called at worker shutdown so EOF
//! / OVS restart releases every slot's GPU resources (spec §6 / R4).
void destroySlotPool()
{
    if (!gPool)
    {
        return;
    }
    // Worker keeps ownership of runtime/stream teardown (OUT of the template).
    for (auto& slotPtr : gPool->slots())
    {
        if (slotPtr->stream != nullptr)
        {
            cudaStreamDestroy(slotPtr->stream);
            slotPtr->stream = nullptr;
        }
        slotPtr->thinker.reset();
    }
    gPool->clear();
    gPool.reset();
}

//! Global MelExtractor handle (M4 step 5).  Loaded once at startup if the
//! caller passed --melSettings / --melFilters (or set the matching env vars).
//! Null means PCM input is disabled and `pcm_b64` chunks are refused.
std::unique_ptr<MelExtractor> gMelExtractor;

//! Convenience wrapper: build the structured kv_capacity_exceeded error event
//! the design doc §12 milestone 2 calls for. M3 routes through this when the
//! runtime returns false with status kKvCapacityExceeded.
Json makeKvCapacityErrorEvent(std::string const& id, int32_t kvLength, int32_t cap)
{
    Json ev = {
        {"event", "error"},
        {"ok", false},
        {"error", "kv_capacity_exceeded"},
        {"kv_length", kvLength},
        {"cap", cap},
    };
    if (!id.empty())
    {
        ev["id"] = id;
    }
    return ev;
}

//! P2 — Explicit prefill_failed error emission. Replaces the prior silent-
//! empty-partial behavior when runtime->handleRequest returns false in the
//! streaming hop path. Surfaces enough diagnostic state (hop_id, audio_sec,
//! kv state, engine cap) for the driver / operator to understand whether
//! the failure is a true engine cap breach (use a bigger thinker) or an
//! upstream input issue.
Json makePrefillFailedEvent(std::string const& id, int32_t hopId, double audioSec,
    rt::LLMInferenceSpecDecodeRuntime const& runtime)
{
    Json ev = {
        {"event", "error"},
        {"ok", false},
        {"error", "prefill_failed"},
        {"hop_id", hopId},
        {"audio_sec", audioSec},
        {"kv_length", runtime.getLastObservedKvLength()},
        {"max_kv_capacity", runtime.getMaxKvCacheCapacity()},
        {"hint", "audio_sec or accumulated token count likely exceeds engine max_input_len; "
                 "see docs/plans/p1-thinker-rebuild-results.md for the rebuild path"},
    };
    if (!id.empty()) {
        ev["id"] = id;
    }
    return ev;
}

//! Maps a runtime AppendPrefillStatus to the structured worker-side JSON event.
std::optional<Json> mapAppendStatusToErrorEvent(
    rt::LLMInferenceSpecDecodeRuntime const& runtime, std::string const& id)
{
    using Status = rt::LLMInferenceSpecDecodeRuntime::AppendPrefillStatus;
    auto const status = runtime.getLastAppendStatus();
    switch (status)
    {
    case Status::kOk: return std::nullopt;
    case Status::kKvCapacityExceeded:
        return makeKvCapacityErrorEvent(id, runtime.getLastObservedKvLength(), runtime.getMaxKvCacheCapacity());
    case Status::kChunkTooLong:
    case Status::kPreconditionFailed:
    case Status::kPrefillFailed:
    default: return std::nullopt;
    }
}

enum OptionId : int
{
    HELP = 1000,
    ENGINE_DIR,
    MULTIMODAL_ENGINE_DIR,
    MEL_SETTINGS,
    MEL_FILTERS,
    MAX_SLOTS,
    DEBUG,
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " --engineDir=<path> --multimodalEngineDir=<path>"
              << " [--melSettings=<json>] [--melFilters=<bin>] [--max_slots=<N>] [--debug]\n\n"
              << "Reads llm_inference-compatible JSON lines from stdin and writes JSON lines to stdout.\n"
              << "Pass --melSettings + --melFilters (or set EDGE_LLM_ASR_MEL_SETTINGS/EDGE_LLM_ASR_MEL_FILTERS)\n"
              << "to enable PCM-input streaming via `pcm_b64` chunk events.\n"
              << "--max_slots=<N> sets the ASR decoder slot-pool size (default 4; D1 concurrency).\n";
}

bool parseArgs(Args& args, int argc, char** argv)
{
    static struct option options[] = {{"help", no_argument, 0, HELP},
        {"engineDir", required_argument, 0, ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, MULTIMODAL_ENGINE_DIR},
        {"melSettings", required_argument, 0, MEL_SETTINGS},
        {"melFilters", required_argument, 0, MEL_FILTERS},
        {"max_slots", required_argument, 0, MAX_SLOTS},
        {"debug", no_argument, 0, DEBUG},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case HELP: printUsage(argv[0]); std::exit(EXIT_SUCCESS);
        case ENGINE_DIR: args.engineDir = optarg; break;
        case MULTIMODAL_ENGINE_DIR: args.multimodalEngineDir = optarg; break;
        case MEL_SETTINGS: args.melSettingsPath = optarg; break;
        case MEL_FILTERS: args.melFiltersPath = optarg; break;
        case MAX_SLOTS:
        {
            int const v = std::atoi(optarg);
            args.maxSlots = (v >= 1) ? v : 1; // clamp to >=1; 1 == single-session behavior
            break;
        }
        case DEBUG: args.debug = true; break;
        default: return false;
        }
    }
    // Env-var fallbacks let the worker auto-locate the assets shipped under
    // deploy/audio_preprocessing/ without forcing every caller to wire flags.
    if (args.melSettingsPath.empty())
    {
        if (char const* p = std::getenv("EDGE_LLM_ASR_MEL_SETTINGS")) args.melSettingsPath = p;
    }
    if (args.melFiltersPath.empty())
    {
        if (char const* p = std::getenv("EDGE_LLM_ASR_MEL_FILTERS")) args.melFiltersPath = p;
    }
    // D1 slot-pool size env fallback (matches the OVS EDGE_LLM_ASR_MAX_CONCURRENT
    // convention in spec §4). CLI --max_slots takes precedence when given.
    if (args.maxSlots == 4)
    {
        if (char const* p = std::getenv("EDGE_LLM_ASR_MAX_CONCURRENT"))
        {
            int const v = std::atoi(p);
            if (v >= 1) args.maxSlots = v;
        }
    }

    return !args.engineDir.empty() && !args.multimodalEngineDir.empty();
}

// ---------------------------------------------------------------------------
// PCM input support (M4 step 5).  Adds a `pcm_b64` field to the chunk event:
// raw float32 16 kHz mono PCM, base64-encoded, MelExtractor produces the mel,
// we write it as a temp safetensors file and let the existing runHop path
// consume it. Backward compat: `mel_path` chunks still work verbatim.
// ---------------------------------------------------------------------------
std::vector<uint8_t> base64Decode(std::string const& in)
{
    static int8_t kT[256];
    static bool kInit = []() {
        for (auto& v : kT) v = -1;
        char const* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) kT[static_cast<unsigned char>(a[i])] = static_cast<int8_t>(i);
        return true;
    }();
    (void)kInit;
    std::vector<uint8_t> out;
    out.reserve(in.size() * 3 / 4 + 4);
    int32_t v = 0;
    int32_t bits = 0;
    for (char c : in)
    {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
        int8_t const x = kT[static_cast<unsigned char>(c)];
        if (x < 0) continue;
        v = (v << 6) | x;
        bits += 6;
        if (bits >= 8)
        {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((v >> bits) & 0xFF));
        }
    }
    return out;
}

//! Write a single fp16 tensor as a safetensors file with key "mel".
//! Layout matches scripts/test_streaming_worker.py::write_safetensors and the
//! mel-tensor consumer in TensorRT-Edge-LLM examples/utils/requestFileParser.
void writeMelSafetensors(std::vector<float> const& mel_f32,
                         int32_t n_mels, int32_t n_frames,
                         std::filesystem::path const& out_path)
{
    // Convert to fp16 to match the on-disk format the worker's mel reader uses
    // (other writers in this repo emit fp16). Use a minimal round-to-nearest
    // fp32→fp16 conversion sufficient for log-mel range (-1.0..1.0 after
    // post_normalize) — full IEEE 754 fp16 with subnormals + NaN/Inf.
    auto f32_to_f16 = [](float f) -> uint16_t {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(x));
        uint32_t const sign = (x >> 16) & 0x8000u;
        int32_t const exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
        uint32_t const mant = x & 0x7FFFFFu;
        if (exp <= 0)
        {
            // Subnormal or underflow.
            if (exp < -10) return static_cast<uint16_t>(sign);
            uint32_t const m = (mant | 0x800000u) >> (1 - exp);
            uint32_t const rounded = (m + 0x1000u) >> 13;
            return static_cast<uint16_t>(sign | rounded);
        }
        if (exp >= 31)
        {
            // Overflow / Inf / NaN.
            if (((x >> 23) & 0xFF) == 0xFF && mant != 0) return static_cast<uint16_t>(sign | 0x7E00u);
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        uint32_t const m = mant >> 13;
        uint32_t const r = mant & 0x1FFFu;
        uint16_t out = static_cast<uint16_t>(sign | (exp << 10) | m);
        if (r > 0x1000u || (r == 0x1000u && (m & 1))) ++out;
        return out;
    };

    int32_t const batch = 1;
    size_t const elem_count = static_cast<size_t>(batch) * n_mels * n_frames;
    if (elem_count != mel_f32.size())
    {
        throw std::runtime_error("writeMelSafetensors: tensor size mismatch");
    }
    std::vector<uint16_t> half(elem_count);
    for (size_t i = 0; i < elem_count; ++i) half[i] = f32_to_f16(mel_f32[i]);

    size_t const nbytes = half.size() * sizeof(uint16_t);
    Json header = Json{{"mel", Json{{"dtype", "F16"},
                                    {"shape", Json::array({batch, n_mels, n_frames})},
                                    {"data_offsets", Json::array({0, nbytes})}}}};
    std::string header_str = header.dump();
    while ((header_str.size() % 8) != 0) header_str.push_back(' ');

    std::ofstream f(out_path, std::ios::binary);
    if (!f) throw std::runtime_error("writeMelSafetensors: cannot open " + out_path.string());
    uint64_t const header_len = static_cast<uint64_t>(header_str.size());
    f.write(reinterpret_cast<char const*>(&header_len), sizeof(header_len));
    f.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
    f.write(reinterpret_cast<char const*>(half.data()), static_cast<std::streamsize>(nbytes));
}

std::filesystem::path writeTempInput(Json const& input, std::string const& id)
{
    std::string safeId = id.empty() ? "request" : id;
    for (auto& ch : safeId)
    {
        bool const ok = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_'
            || ch == '-';
        if (!ok)
        {
            ch = '_';
        }
    }
    auto path = std::filesystem::temp_directory_path()
        / ("qwen3_asr_worker_" + safeId + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
            + ".json");
    std::ofstream file(path);
    if (!file)
    {
        throw std::runtime_error("Failed to open temp input file: " + path.string());
    }
    file << input.dump();
    return path;
}

// ---------------------------------------------------------------------------
// SPIKE — replaced in step 3.
// Step 2 chunk handler: per-chunk full one-shot decode via existing
// handleRequest path. No prefix prompt. The test driver writes per-hop
// mel safetensors files containing the FULL audio accumulated so far
// (hop k = first 500*(k+1) ms). Worker just builds the legacy one-shot
// request JSON pointing at that mel and times the call.
// Step 3 will replace this with prefix-prompt rollback (§15.6 step 3).
// ---------------------------------------------------------------------------

//! Per-stage timing snapshot pulled from gTimer. Tracks the last-recorded
//! GPU time for each stage of interest. We capture cumulative entry counts
//! between hops so we can diff and isolate this-hop's contribution even
//! when stages report multiple runs per call (e.g. eagle iterations).
struct StageTimingSnapshot
{
    float encoderMs{0.0f};         //!< Sum of new entries for audio_encoder this hop.
    float prefillMs{0.0f};         //!< Sum of new entries for llm_prefill this hop.
    float decodeMs{0.0f};          //!< Sum of new entries for llm_generation this hop.
};

//! Cumulative counters across hops, used to diff per-stage timing slices.
struct StageTimingCounters
{
    size_t encoderEntries{0};
    size_t prefillEntries{0};
    size_t decodeEntries{0};
};

float sumNewEntries(std::string const& stageId, size_t& priorCount)
{
    auto data = gTimer.getTimingData(stageId);
    if (!data.has_value())
    {
        return 0.0f;
    }
    auto const& times = data->gpuTimesMs;
    float sumMs = 0.0f;
    for (size_t i = priorCount; i < times.size(); ++i)
    {
        sumMs += times[i];
    }
    priorCount = times.size();
    return sumMs;
}

StageTimingSnapshot captureStageDelta(StageTimingCounters& counters)
{
    StageTimingSnapshot snap;
    snap.encoderMs = sumNewEntries(metrics::StageNames::kAUDIO_ENCODER, counters.encoderEntries);
    snap.prefillMs = sumNewEntries(metrics::StageNames::kLLM_PREFILL, counters.prefillEntries);
    snap.decodeMs = sumNewEntries(metrics::StageNames::kLLM_GENERATION, counters.decodeEntries);
    return snap;
}

//! Build the legacy one-shot request JSON for a single mel file.
//! Mirrors the request layout in scripts/validate_qwen3_tts_quality_gate.py.
Json buildOneShotRequestForMel(std::string const& melPath, int32_t maxGenerateLength)
{
    Json msg = {
        {"role", "user"},
        {"content", Json::array({Json{{"type", "audio"}, {"audio", melPath}}})},
    };
    Json req = {
        {"messages", Json::array({msg})},
    };
    Json input = {
        {"requests", Json::array({req})},
        {"batch_size", 1},
        {"temperature", 1.0},
        {"top_p", 1.0},
        {"top_k", 1},
        {"max_generate_length", maxGenerateLength},
        {"apply_chat_template", true},
        {"add_generation_prompt", true},
    };
    return input;
}

//! Core: drive one handleRequest pass on a mel file, return (text, timings).
//! Used by handleChunk/handleEnd in spike. Returns std::nullopt on failure.
struct HopResult
{
    bool ok{false};
    std::string text;              //!< Legacy: full output from runHop (back-compat).
    std::string generatedText;     //!< Step 3.1: raw output (excludes prefix) from runStreamingHop.
    std::string rawDecoded;        //!< Step 3.1: prefix + generatedText.
    double totalMs{0.0};
    StageTimingSnapshot stages{};
};

// ---------------------------------------------------------------------------
// Step 3.1: prefix-prompt rollback chunk-and-confirm loop (§15.6 step 3).
// Mirrors qwen3_asr.py:728-746 (per-hop) / 809-816 (finish variant). Saves
// ~21ms/hop steady-state by skipping re-decode of confirmed prefix tokens.
// Requires EdgeLLM Method B fix (system-prompt KV cache mismatch fallback,
// streaming-asr/m1-append-prefill-embeds@0f618d6) to avoid duplication.
// ---------------------------------------------------------------------------
constexpr char const* kAssistantGenPrompt = "<|im_start|>assistant\n";
constexpr char const* kUtf8Replacement = "\xEF\xBF\xBD";  //!< U+FFFD as UTF-8.

//! Strip a leading "language X<asr_text>" / language prefix from raw output.
//! Splits on <asr_text> tag when present; otherwise trims "language Xxx ".
std::string parseAsrText(std::string const& raw)
{
    static constexpr char const* kAsrTextTag = "<asr_text>";
    auto tagPos = raw.find(kAsrTextTag);
    if (tagPos != std::string::npos)
    {
        return raw.substr(tagPos + std::strlen(kAsrTextTag));
    }
    if (raw.rfind("language ", 0) == 0)
    {
        auto nl = raw.find('\n');
        if (nl != std::string::npos)
        {
            return raw.substr(nl + 1);
        }
        auto sp1 = raw.find(' ', 9);
        if (sp1 != std::string::npos)
        {
            auto sp2 = raw.find(' ', sp1 + 1);
            if (sp2 != std::string::npos)
            {
                return raw.substr(sp2 + 1);
            }
        }
    }
    return raw;
}

//! Build prefix text from accumulated decoded output via tokenizer roll-back.
//! Mirrors qwen3_asr.py:728-746 (per-hop) and 809-816 (finish variant).
std::string computePrefix(AsrSessionState const& session, tokenizer::Tokenizer* tok, bool isFinish)
{
    if (tok == nullptr || session.rawDecoded.empty())
    {
        return "";
    }
    if (session.chunkId < session.unfixedChunkNum)
    {
        return "";
    }
    auto tokens = tok->encode(session.rawDecoded, /*addBos=*/false, /*addEos=*/false);
    if (tokens.empty())
    {
        return "";
    }
    int k = session.unfixedTokenNum;
    int const total = static_cast<int>(tokens.size());
    while (true)
    {
        int end = total - k;
        if (isFinish)
        {
            end = std::max(1, end);
        }
        else
        {
            end = std::max(0, end);
        }
        std::string prefix;
        if (end > 0)
        {
            std::vector<tokenizer::Rank> slice(tokens.begin(), tokens.begin() + end);
            prefix = tok->decode(slice, /*skipSpecialTokens=*/false);
        }
        if (prefix.find(kUtf8Replacement) == std::string::npos)
        {
            return prefix;
        }
        if (!isFinish && end == 0)
        {
            return "";
        }
        if (isFinish && end == 1)
        {
            return prefix;
        }
        ++k;
    }
}

//! Build LLMGenerationRequest from prefix + audio mel and invoke runtime.
//! Step 3.1 streaming hop: assistant-message-as-prefix trick. Requires
//! EdgeLLM Method B fix to function correctly across all hop indices.
HopResult runStreamingHop(std::string const& melPath, std::string const& prefix,
    AsrSessionState const& session, rt::LLMInferenceSpecDecodeRuntime& runtime,
    cudaStream_t stream, StageTimingCounters& stageCounters)
{
    HopResult result;
    auto const t0 = std::chrono::steady_clock::now();
    try
    {
        rt::LLMGenerationRequest::Request req;

        rt::Message sysMsg;
        sysMsg.role = "system";
        sysMsg.contents.push_back({"text", session.context});
        req.messages.push_back(sysMsg);

        rt::Message userMsg;
        userMsg.role = "user";
        rt::Message::MessageContent audioContent;
        audioContent.type = "audio";
        audioContent.content = melPath;
        userMsg.contents.push_back(audioContent);
        req.messages.push_back(userMsg);

        std::string assistantContent = kAssistantGenPrompt;
        if (!session.forceLanguage.empty())
        {
            assistantContent += "language " + session.forceLanguage + "<asr_text>";
        }
        assistantContent += prefix;
        rt::Message asstMsg;
        asstMsg.role = "assistant";
        asstMsg.contents.push_back({"text", assistantContent});
        req.messages.push_back(asstMsg);

        rt::audioUtils::AudioData audio;
        audio.melSpectrogramPath = melPath;
        audio.melSpectrogramFormat = "safetensors";
        req.audioBuffers.push_back(std::move(audio));

        rt::LLMGenerationRequest llmReq;
        llmReq.requests.push_back(std::move(req));
        llmReq.temperature = 1.0f;
        llmReq.topP = 1.0f;
        llmReq.topK = 1;
        llmReq.maxGenerateLength = session.maxDecodeTokensPerHop;
        llmReq.applyChatTemplate = true;
        llmReq.addGenerationPrompt = false;
        llmReq.enableThinking = false;

        rt::LLMGenerationResponse llmResponse;
        // Step 3 — SHARED encoder serialization (spec §2 / R2). The audio
        // encoder currently lives INSIDE each slot's runtime and is invoked as
        // part of handleRequest (multimodal prefill), so it cannot be isolated
        // and serialized on its own without a fork-side encoder split. Until
        // that split lands we conservatively hold gEncoderMutex across the
        // whole hop: this guarantees two slots never run their encoder passes
        // concurrently (correctness/safety) at the cost of also serializing the
        // decode portion. [needs follow-up fork-side encoder split to serialize
        // ONLY the encoder and let decoders overlap across slots/streams.]
        bool ok = false;
        {
            std::lock_guard<std::mutex> const encGuard(gEncoderMutex);
            ok = runtime.handleRequest(llmReq, llmResponse, stream);
        }
        result.ok = ok;
        if (ok && !llmResponse.outputTexts.empty())
        {
            result.generatedText = llmResponse.outputTexts[0];
        }
        result.rawDecoded = prefix + result.generatedText;
        result.text = result.rawDecoded;  //!< Back-compat for code reading hop.text.
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("runStreamingHop exception: %s", e.what());
        result.ok = false;
        result.generatedText = std::string("hop_exception: ") + e.what();
        result.rawDecoded = prefix + result.generatedText;
        result.text = result.rawDecoded;
    }
    result.totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    result.stages = captureStageDelta(stageCounters);
    return result;
}

HopResult runHop(std::string const& melPath, int32_t maxGenerateLength,
    rt::LLMInferenceSpecDecodeRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap,
    StageTimingCounters& stageCounters)
{
    HopResult result;
    auto const t0 = std::chrono::steady_clock::now();
    std::filesystem::path tempPath;
    try
    {
        Json input = buildOneShotRequestForMel(melPath, maxGenerateLength);
        tempPath = writeTempInput(input, "spike_hop");
        std::vector<rt::LLMGenerationRequest> batched;
        std::tie(loraWeightsMap, batched) = exampleUtils::parseRequestFile(tempPath, -1, -1);
        if (batched.empty())
        {
            throw std::runtime_error("parseRequestFile produced no requests");
        }
        rt::LLMGenerationResponse llmResponse;
        bool const ok = runtime.handleRequest(batched[0], llmResponse, stream);
        result.ok = ok;
        if (ok && !llmResponse.outputTexts.empty())
        {
            result.text = llmResponse.outputTexts[0];
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("runHop exception: %s", e.what());
        result.ok = false;
        result.text = std::string("runHop_exception: ") + e.what();
    }
    if (!tempPath.empty())
    {
        std::error_code ec;
        std::filesystem::remove(tempPath, ec);
    }
    result.totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    result.stages = captureStageDelta(stageCounters);
    return result;
}

//! Step 3 — slot-routed begin. Acquires a free slot for this sessionId (or
//! reuses the one already bound for an idempotent re-begin), binds the
//! sessionId→slotId map, and initialises that slot's AsrSessionState. Emits a
//! 4429 pool-saturated error when every slot is busy (replaces the old
//! single-session `session_already_active` guard).
void handleBegin(Json const& input)
{
    std::string const id = input.value("id", "");

    // RISK POINT 2 — idempotent begin: a `begin` for a sessionId that already
    // owns a slot reuses that slot (we do NOT acquire a second one); otherwise
    // a fresh free slot is claimed and bound. The check-then-acquire is atomic
    // (SlotPool::acquireOrExisting does lookup + CAS-claim + bind under the pool
    // mutex), matching the old single-session semantics where re-begin
    // re-initialised in place. Note: the legacy acquireSlot reset the slot's
    // session on claim, but handleBegin reinitialises `session` below on EVERY
    // begin (reuse or fresh), so the reset is preserved unconditionally.
    int32_t const slotId = gPool->acquireOrExisting(id);
    if (slotId < 0)
    {
        // Pool saturated: every decoder slot is in use. Surface a 4429-style
        // structured error so the front-end / session-limiter can back off
        // (replaces the legacy `session_already_active` single-session guard).
        Json ev = {{"event", "error"}, {"ok", false}, {"error", "pool_saturated"},
            {"status", 4429}, {"max_slots", gMaxSlots}};
        if (!id.empty())
        {
            ev["id"] = id;
        }
        std::cout << ev.dump() << std::endl;
        return;
    }

    AsrSessionState& session = gPool->get(slotId)->session;
    session = AsrSessionState{};
    session.sessionId = id;
    session.active = true;
    session.lastActivity = std::chrono::steady_clock::now();
    session.sampleRate = input.value("sample_rate", 16000.0);
    session.chunkSizeSec = input.value("chunk_size_sec", 0.5);
    session.unfixedChunkNum = input.value("unfixed_chunk_num", 2);
    session.unfixedTokenNum = input.value("unfixed_token_num", 5);
    session.context = input.value("context", std::string{});
    if (input.contains("force_language") && !input["force_language"].is_null())
    {
        session.forceLanguage = input["force_language"].get<std::string>();
    }
    Json ev = {{"event", "begin_ack"}, {"id", id}};
    std::cout << ev.dump() << std::endl;
}

//! Strip "language Chinese " / "language English " etc. prefixes that the
//! ASR model adds. Mirror of the spike test driver helper. Applied before
//! concatenating segment texts so the fullText output reads naturally.
std::string stripLanguagePrefix(std::string const& text)
{
    static char const* kPrefix = "language ";
    if (text.compare(0, std::strlen(kPrefix), kPrefix) != 0)
    {
        return text;
    }
    static char const* kLangs[] = {"Chinese", "English", "Cantonese", "Japanese", "Korean",
        "French", "German", "Italian", "Portuguese", "Russian", "Spanish"};
    for (auto const* lang : kLangs)
    {
        std::string const probe = std::string(kPrefix) + lang;
        if (text.compare(0, probe.size(), probe) == 0)
        {
            size_t off = probe.size();
            while (off < text.size() && (text[off] == ' ' || text[off] == '\t' || text[off] == '\n'))
            {
                ++off;
            }
            return text.substr(off);
        }
    }
    return text;
}

//! P1 — Auto-segment boundary dedup (design doc §15.5.2).
//!
//! At segment rotation, the driver carries over `kCarryoverSec` of audio
//! into the next segment to preserve context continuity. The first hop of
//! the new segment therefore re-transcribes that overlap region, causing
//! the head of newSegment to duplicate the tail of fullText (the LCS≤0.85
//! failure on long Chinese utterances).
//!
//! Strategy: find the longest k such that fullText's last k chars equal
//! newSegment's first k chars (UTF-8 byte-wise; safe because we only
//! return a clean cut when the boundary aligns on multi-byte sequence
//! boundaries — see the continuation-byte guard below). Return the
//! newSegment with that overlap trimmed.
//!
//! Tuned for Chinese (3 bytes/char). The overlap window is bounded by
//! 2 × kCarryoverSec × audio_tokens_per_sec × ~2 bytes/char ≈ 80 bytes
//! upper bound for typical fast-speech zh-CN. We scan up to min(64, half
//! the shorter string) for cost control.
std::string dedupAtBoundary(std::string const& fullText, std::string const& newSegment)
{
    if (fullText.empty() || newSegment.empty())
    {
        return newSegment;
    }
    auto const isUtf8Cont = [](unsigned char c) { return (c & 0xC0) == 0x80; };
    int const scanLimit = static_cast<int>(std::min<size_t>({fullText.size(), newSegment.size(), size_t{96}}));
    for (int k = scanLimit; k > 0; --k)
    {
        // Skip k that would split a UTF-8 sequence on either side.
        if (isUtf8Cont(static_cast<unsigned char>(newSegment[k - 1])))
        {
            // mid-sequence end on the newSegment side, only valid if next byte (newSegment[k]) is also continuation
            // simpler: just require the byte AT newSegment[k] (if exists) to not be a continuation byte → boundary
        }
        if (static_cast<size_t>(k) < newSegment.size()
            && isUtf8Cont(static_cast<unsigned char>(newSegment[k])))
        {
            continue;  //!< k cuts mid-UTF8 in newSegment; skip.
        }
        if (fullText.compare(fullText.size() - k, k, newSegment, 0, k) == 0)
        {
            return newSegment.substr(k);
        }
    }
    return newSegment;
}

void handleChunk(Json const& input,
    std::unordered_map<std::string, std::string>& loraWeightsMap,
    StageTimingCounters& stageCounters, int32_t maxGenerateLength)
{
    std::string const id = input.value("id", "");
    // Step 3 — route to the slot bound at begin. A missing mapping means the
    // session was never begun, already ended, or timed out.
    int32_t const slotId = lookupSlot(id);
    if (slotId < 0)
    {
        Json ev = {{"event", "error"}, {"ok", false}, {"error", "no_active_session"}};
        if (!id.empty())
        {
            ev["id"] = id;
        }
        std::cout << ev.dump() << std::endl;
        return;
    }
    AsrSlot& slot = *gPool->get(slotId);
    AsrSessionState& session = slot.session;
    rt::LLMInferenceSpecDecodeRuntime& runtime = *slot.thinker;
    cudaStream_t const stream = slot.stream;
    if (!session.active)
    {
        Json ev = {{"event", "error"}, {"ok", false}, {"error", "no_active_session"}};
        if (!id.empty())
        {
            ev["id"] = id;
        }
        std::cout << ev.dump() << std::endl;
        return;
    }

    // Step 3 — slot release guard. Every terminal path in this handler (error
    // cleanup AND the last=true final) calls freeSession(session), which clears
    // session.active. A partial/rotation hop returns with session.active still
    // true (the session continues on this slot). So at scope exit we release
    // the slot back to the pool + drop the routing mapping IFF the session
    // became inactive — i.e. exactly when a terminal path fired. This keeps the
    // ~7 existing freeSession() cleanup sites untouched while guaranteeing no
    // slot leak on any exit path.
    struct SlotReleaseGuard
    {
        AsrSessionState& s;
        int32_t sid;
        std::string id;
        ~SlotReleaseGuard()
        {
            if (!s.active)
            {
                releaseSlot(sid);     // resets slot.session + marks slot free
                unbindSession(id);    // drop sessionId → slot mapping
            }
        }
    } slotReleaseGuard{session, slotId, id};
    bool const hasMelPath = input.contains("mel_path") && input["mel_path"].is_string();
    bool const hasPcm = input.contains("pcm_b64") && input["pcm_b64"].is_string();
    if (!hasMelPath && !hasPcm)
    {
        Json ev = {{"event", "error"}, {"ok", false}, {"error", "chunk_missing_mel_path"}, {"id", id}};
        std::cout << ev.dump() << std::endl;
        freeSession(session);
        return;
    }
    if (hasPcm && !gMelExtractor)
    {
        Json ev = {{"event", "error"}, {"ok", false},
            {"error", "pcm_input_unsupported"}, {"id", id},
            {"hint", "worker started without --melSettings/--melFilters; pass them or set EDGE_LLM_ASR_MEL_{SETTINGS,FILTERS}"}};
        std::cout << ev.dump() << std::endl;
        freeSession(session);
        return;
    }

    // Tracks any tempfile we have to clean up at end-of-hop.
    std::filesystem::path melTempPath;
    std::string melPath;
    if (hasPcm)
    {
        try
        {
            std::string const b64 = input["pcm_b64"].get<std::string>();
            std::vector<uint8_t> raw = base64Decode(b64);
            if (raw.size() % sizeof(float) != 0)
            {
                Json ev = {{"event", "error"}, {"ok", false},
                    {"error", "pcm_b64_malformed"}, {"id", id},
                    {"hint", "raw float32 LE expected; decoded bytes not divisible by 4"}};
                std::cout << ev.dump() << std::endl;
                freeSession(session);
                return;
            }
            std::vector<float> pcm(raw.size() / sizeof(float));
            std::memcpy(pcm.data(), raw.data(), raw.size());
            // P2 — keep the latest cumulative PCM so VAD can locate a
            // low-energy boundary at rotation time. Drivers send cumulative
            // PCM slices (matching the cumulative-mel convention), so we
            // simply overwrite each chunk's accumulator with the latest.
            session.pcmAccum = pcm;
            int32_t n_frames = 0;
            // Pad to encoder chunk size so [-1, 128, 100] profile is satisfied
            // (first hop ~0.5s → 50 frames → must pad to 100). Mirrors
            // scripts/test_streaming_worker.py:146-147.
            std::vector<float> mel = gMelExtractor->compute(pcm, &n_frames,
                                                            kEncoderMelFramesPerChunk);
            if (n_frames <= 0)
            {
                Json ev = {{"event", "error"}, {"ok", false},
                    {"error", "pcm_too_short"}, {"id", id}, {"n_samples", static_cast<int64_t>(pcm.size())}};
                std::cout << ev.dump() << std::endl;
                freeSession(session);
                return;
            }
            // Write temp safetensors that the existing runHop path consumes.
            melTempPath = std::filesystem::temp_directory_path()
                / ("qwen3_asr_pcm_mel_" + (id.empty() ? std::string("s") : id) + "_"
                   + std::to_string(session.chunkId) + "_"
                   + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
                   + ".safetensors");
            writeMelSafetensors(mel, gMelExtractor->n_mels(), n_frames, melTempPath);
            melPath = melTempPath.string();
        }
        catch (std::exception const& e)
        {
            Json ev = {{"event", "error"}, {"ok", false},
                {"error", "pcm_to_mel_failed"}, {"id", id}, {"detail", e.what()}};
            std::cout << ev.dump() << std::endl;
            freeSession(session);
            return;
        }
    }
    else
    {
        melPath = input["mel_path"].get<std::string>();
    }

    // RAII cleanup for any PCM-derived temp safetensors. Captures by reference
    // so we don't try to remove an empty path when mel_path was provided.
    struct TempCleanup
    {
        std::filesystem::path& p;
        ~TempCleanup()
        {
            if (!p.empty())
            {
                std::error_code ec;
                std::filesystem::remove(p, ec);
            }
        }
    } cleanup{melTempPath};

    bool const isLast = input.value("last", false);
    // audio_sec is OPTIONAL for backward compat with step 2 spike (which omits
    // it and treats the worker as a stateless cumulative-mel decoder). When
    // present, drives the max_input_len cap + auto-segmentation policy.
    // For PCM input, we infer audio_sec from the decoded sample count if the
    // caller didn't provide it explicitly — the cap policy applies equally.
    bool const hasAudioSec = input.contains("audio_sec") && input["audio_sec"].is_number();
    double const audioSec = hasAudioSec ? input["audio_sec"].get<double>() : 0.0;
    session.lastActivity = std::chrono::steady_clock::now();
    session.lastAudioSec = audioSec;

    // §15.6 step 4 — hard refuse: pathological chunk longer than the engine
    // can ever handle in a single hop. Audio_sec=0 path (spike compat) skips
    // this entirely.
    if (hasAudioSec && audioSec > kSingleChunkHardLimitSec)
    {
        Json ev = {{"event", "error"}, {"ok", false}, {"error", "chunk_too_long"},
            {"id", id}, {"chunk_sec", audioSec}, {"limit_sec", kSingleChunkHardLimitSec}};
        std::cout << ev.dump() << std::endl;
        freeSession(session);
        return;
    }

    // §15.6 step 4 — auto-segmentation: if running this hop would push input
    // tokens past the engine cap, run the hop as a segment-final FIRST, save
    // its text to fullText, then signal the driver to rotate and continue.
    // We only auto-segment when audio_sec is provided AND this is NOT the
    // last chunk (last=true always runs the final hop and emits final).
    //
    // P2 — VAD-aligned rotation. When the chunk arrives in PCM mode AND
    // audio_sec is within the VAD trigger band [kVadTriggerSec,
    // kSingleChunkHardLimitSec), invoke AudioVadSplitter on the accumulated
    // PCM to pick a clean boundary; if a true split is returned (>=2
    // chunks), use the first chunk's end as the rotation point. Otherwise
    // fall through to the legacy projected-overflow rotation policy.
    bool const overflowTrigger = hasAudioSec && !isLast && wouldOverflow(audioSec);
    bool vadAligned = false;
    double vadBoundarySec = 0.0;
    double vadCarryoverSec = kCarryoverSec;
    if (!isLast && hasAudioSec && audioSec >= kVadTriggerSec && !session.pcmAccum.empty())
    {
        edgellm_voice_worker::AudioVadSplitter const vad(
            /*sample_rate=*/16000, kVadMaxChunkSec, kVadSearchExpandSec,
            /*min_window_ms=*/100.0, kVadMinChunkSec);
        auto const chunks = vad.split(session.pcmAccum);
        if (chunks.size() >= 2)
        {
            // First boundary = end of chunks[0] in sample index. The tail
            // (everything after this boundary) becomes the next segment's
            // initial PCM window — driver gets `carryover_sec = audio_sec
            // - vad_boundary_sec` so it knows precisely how much audio to
            // keep when starting the next session view.
            int64_t boundarySamples = 0;
            for (auto const& c : chunks)
            {
                boundarySamples += static_cast<int64_t>(c.samples.size());
                break;  // only first chunk's length
            }
            vadBoundarySec = static_cast<double>(boundarySamples) / 16000.0;
            // Carryover is the audio AFTER the VAD boundary up to current
            // accumulator length (typically ~2-3 s of post-boundary speech).
            double const tailSec
                = static_cast<double>(session.pcmAccum.size()) / 16000.0 - vadBoundarySec;
            vadCarryoverSec = std::max(0.0, tailSec);
            vadAligned = true;
        }
    }
    bool const shouldRotate = overflowTrigger || vadAligned;

    int32_t const hopId = session.chunkId;
    // Step 3.1 prefix-prompt: compute prefix from prior rawDecoded via
    // tokenizer rollback, then drive a streaming hop that surfaces the
    // prefix through the assistant message. Requires EdgeLLM Method B fix.
    tokenizer::Tokenizer* tok = runtime.getTokenizerForTesting();
    std::string const prefix = computePrefix(session, tok, /*isFinish=*/isLast);
    HopResult const hop = runStreamingHop(melPath, prefix, session, runtime, stream, stageCounters);
    // Silence unused-warning for the legacy runHop path (kept for one-shot
    // request flows and unused vars from the old call signature).
    (void) maxGenerateLength;
    (void) loraWeightsMap;
    session.chunkId += 1;

    // P2 — explicit prefill failure. Prior behavior silently stored
    // hop.text (which on failure contains an "hop_exception:" string or an
    // empty payload) into rawDecoded and emitted a meaningless partial.
    // Now we surface a structured error event, free the session, and
    // stop processing this stream so the driver fails fast.
    if (!hop.ok)
    {
        Json err = makePrefillFailedEvent(id, hopId, audioSec, runtime);
        err["elapsed_ms"] = hop.totalMs;
        err["hop_text"] = hop.text;  //!< Optional diagnostic payload (e.g. C++ exception string).
        std::cout << err.dump() << std::endl;
        freeSession(session);
        return;
    }
    session.rawDecoded = hop.rawDecoded;

    if (shouldRotate)
    {
        // Append this segment's text and rotate. Strip language prefix so
        // segment texts concatenate cleanly. P1: LCS-based boundary dedup
        // strips overlap re-transcribed from the carryover window.
        std::string segText = stripLanguagePrefix(hop.text);
        std::string const dedupedSegText = dedupAtBoundary(session.fullText, segText);
        session.fullText += dedupedSegText;
        session.segmentCount += 1;
        session.chunkId = 0;
        session.rawDecoded.clear();
        // P2 — VAD-aligned mode trims the in-worker PCM accumulator past
        // the boundary; the new session view starts from the post-boundary
        // tail (driver mirrors this for its own buffer accounting).
        double const carryoverSec = vadAligned ? vadCarryoverSec : kCarryoverSec;
        if (vadAligned)
        {
            int64_t const boundarySamples
                = static_cast<int64_t>(std::llround(vadBoundarySec * 16000.0));
            if (boundarySamples > 0
                && boundarySamples < static_cast<int64_t>(session.pcmAccum.size()))
            {
                session.pcmAccum.erase(session.pcmAccum.begin(),
                    session.pcmAccum.begin() + static_cast<std::ptrdiff_t>(boundarySamples));
            }
            else
            {
                session.pcmAccum.clear();
            }
        }
        else
        {
            session.pcmAccum.clear();
        }
        // Emit the rotation signal — driver MUST trim its audio buffer to
        // the last `carryover_sec` and continue sending cumulative mels
        // starting from there. NOT exposed as a partial/final.
        Json ev = {
            {"event", "segment_rotation"},
            {"id", id},
            {"segment_id", session.segmentCount - 1},
            {"carryover_sec", carryoverSec},
            {"projected_tokens", projectInputTokens(audioSec)},
            {"cap_tokens", kEngineMaxInputLen - kInputSafetyMargin},
            {"segment_text", segText},
            {"elapsed_ms", hop.totalMs},
            {"encoder_ms", hop.stages.encoderMs},
            {"prefill_ms", hop.stages.prefillMs},
            {"decode_ms", hop.stages.decodeMs},
            {"rotation_kind", vadAligned ? "vad_aligned" : "projected_overflow"},
        };
        if (vadAligned)
        {
            ev["vad_boundary_sec"] = vadBoundarySec;
        }
        std::cout << ev.dump() << std::endl;
        return;
    }

    Json ev = {
        {"event", isLast ? "final" : "partial"},
        {"id", id},
        {"hop_id", hopId},
        {"ok", hop.ok},
        {"elapsed_ms", hop.totalMs},
        {"encoder_ms", hop.stages.encoderMs},
        {"prefill_ms", hop.stages.prefillMs},
        {"decode_ms", hop.stages.decodeMs},
    };
    if (isLast)
    {
        // Build final text = concatenated segments + this final hop, with
        // P1 boundary dedup applied to the final segment as well (catches
        // the rotate-then-immediate-finish case).
        std::string segText = stripLanguagePrefix(hop.text);
        std::string const dedupedSegText = dedupAtBoundary(session.fullText, segText);
        std::string finalText = session.fullText + dedupedSegText;
        ev["text"] = finalText;
        ev["segment_count"] = session.segmentCount + 1;
        ev["total_ms"] = hop.totalMs;
        std::cout << ev.dump() << std::endl;
        freeSession(session);
        return;
    }
    // Partial: keep the raw (with language prefix) for parity with step 2 spike.
    ev["text"] = hop.text;
    std::cout << ev.dump() << std::endl;
}

void handleEnd(Json const& input)
{
    std::string const id = input.value("id", "");
    // Step 3 — route to the slot bound at begin; emit end_ack if unknown.
    int32_t const slotId = lookupSlot(id);
    if (slotId < 0)
    {
        Json ev = {{"event", "end_ack"}, {"id", id}};
        std::cout << ev.dump() << std::endl;
        return;
    }
    AsrSessionState& session = gPool->get(slotId)->session;
    // The driver flags the final hop via last=true on a chunk event (which
    // emits the `final` event there). Bare `end` events emit a `final` with
    // whatever fullText we have accumulated and close the session.
    if (session.active && (!session.fullText.empty() || session.segmentCount > 0))
    {
        Json ev = {{"event", "final"}, {"id", id}, {"text", session.fullText},
            {"segment_count", session.segmentCount}, {"ok", true}};
        std::cout << ev.dump() << std::endl;
    }
    else
    {
        Json ev = {{"event", "end_ack"}, {"id", id}};
        std::cout << ev.dump() << std::endl;
    }
    // Step 3 — release the slot back to the pool and drop the routing mapping.
    // releaseSlot() performs freeSession() on the slot (token/mel/PCM/decoded
    // accumulators) so no cross-session KV/state residue survives the reuse.
    releaseSlot(slotId);
    unbindSession(id);
}

// ---------------------------------------------------------------------------
// One-shot legacy handler. Existing M2 behavior preserved verbatim — only
// the surrounding main() loop changes. handleOneShot must produce a JSON
// response byte-equivalent (modulo `total_ms` jitter) to the M2 worker.
// ---------------------------------------------------------------------------
void handleOneShot(Json input, rt::LLMInferenceSpecDecodeRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap)
{
    Json response;
    std::filesystem::path tempPath;
    auto const requestStart = std::chrono::steady_clock::now();
    try
    {
        std::string const id = input.value("id", "");
        input.erase("id");
        int32_t const batchSizeOverride = input.value("batch_size_override", -1);
        int64_t const maxGenerateLengthOverride = input.value("max_generate_length_override", -1);
        input.erase("batch_size_override");
        input.erase("max_generate_length_override");

        tempPath = writeTempInput(input, id);
        std::vector<rt::LLMGenerationRequest> batchedRequests;
        std::tie(loraWeightsMap, batchedRequests)
            = exampleUtils::parseRequestFile(tempPath, batchSizeOverride, maxGenerateLengthOverride);
        if (batchedRequests.empty())
        {
            throw std::runtime_error("No valid ASR requests found");
        }

        Json responses = Json::array();
        bool ok = true;
        for (size_t requestIdx = 0; requestIdx < batchedRequests.size(); ++requestIdx)
        {
            rt::LLMGenerationResponse llmResponse;
            bool const requestOk = runtime.handleRequest(batchedRequests[requestIdx], llmResponse, stream);
            ok = ok && requestOk;
            for (size_t batchIdx = 0; batchIdx < batchedRequests[requestIdx].requests.size(); ++batchIdx)
            {
                bool const hasOutputText = requestOk && batchIdx < llmResponse.outputTexts.size();
                std::string const text = hasOutputText
                    ? llmResponse.outputTexts[batchIdx]
                    : "TensorRT Edge LLM cannot handle this request. Fails.";
                responses.push_back(Json{
                    {"request_idx", requestIdx}, {"batch_idx", batchIdx}, {"output_text", text}});
            }
        }
        double const totalMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - requestStart).count();
        if (!ok)
        {
            if (auto structuredEv = mapAppendStatusToErrorEvent(runtime, id))
            {
                Json ev = std::move(*structuredEv);
                ev["total_ms"] = totalMs;
                response = std::move(ev);
            }
            else
            {
                response = Json{{"id", id}, {"event", "error"}, {"ok", false}, {"responses", responses},
                    {"total_ms", totalMs}};
            }
        }
        else
        {
            response = Json{{"id", id}, {"event", "done"}, {"ok", true}, {"responses", responses},
                {"total_ms", totalMs}};
        }
    }
    catch (std::exception const& e)
    {
        response = Json{{"event", "error"}, {"ok", false}, {"error", e.what()}};
    }
    if (!tempPath.empty())
    {
        std::error_code ec;
        std::filesystem::remove(tempPath, ec);
    }
    std::cout << response.dump() << std::endl;
}

} // namespace

int main(int argc, char** argv)
{
    Args args;
    if (!parseArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    gLogger.setLevel(args.debug ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING);
    auto pluginHandles = loadEdgellmPluginLib();

    // M4 step 5: load mel-preprocessing assets if provided. PCM input via
    // `pcm_b64` chunk events requires both files; mel_path-only callers
    // remain fully supported when these are absent.
    if (!args.melSettingsPath.empty() && !args.melFiltersPath.empty())
    {
        try
        {
            gMelExtractor = std::make_unique<MelExtractor>(args.melSettingsPath, args.melFiltersPath);
            LOG_INFO("MelExtractor loaded (n_fft=%d n_mels=%d hop=%d)",
                     gMelExtractor->n_fft(), gMelExtractor->n_mels(), gMelExtractor->hop_length());
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("MelExtractor init failed: %s — PCM input disabled", e.what());
            gMelExtractor.reset();
        }
    }

    // SPIKE — enable stage timing so the chunk handler can report per-stage ms.
    setProfilingEnabled(true);

    auto const initStart = std::chrono::steady_clock::now();
    std::unordered_map<std::string, std::string> loraWeightsMap;
    bool const enableGraph = std::getenv("EDGE_LLM_ASR_CUDA_GRAPH") == nullptr
        || std::string(std::getenv("EDGE_LLM_ASR_CUDA_GRAPH")) != "0";

    // D1 (Step 2): build the N-slot decoder pool. Slot 0 deserializes the base
    // engine once; slots 1..N-1 SHARE that ICudaEngine via the fork
    // shared-engine constructor (weight memory paid once, contexts independent).
    // Each slot owns its own CUDA stream. CUDA-graph capture is per-slot.
    gMaxSlots = args.maxSlots;
    initSlotPool(args.engineDir, args.multimodalEngineDir, loraWeightsMap, enableGraph);

    double const initMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - initStart).count();
    std::cout << Json{{"event", "ready"}, {"init_ms", initMs}, {"max_slots", gMaxSlots}}.dump() << std::endl;

    // Step 3: begin/chunk/end now route by sessionId across ALL slots (see the
    // handle* functions, which call lookupSlot/acquireSlot/releaseSlot). The
    // legacy one-shot path (lines with no `event` field) keeps using slot 0's
    // runtime + stream — it is a single serial request flow and never touches
    // the slot pool's session state.
    AsrSlot& slot0 = *gPool->get(0);
    rt::LLMInferenceSpecDecodeRuntime* runtime = slot0.thinker.get(); // one-shot legacy path only
    cudaStream_t stream = slot0.stream;                              // one-shot legacy path only
    // SPIKE — cumulative stage entry counters; runHop diffs against these.
    StageTimingCounters stageCounters{};
    // SPIKE — per-hop decode budget. Generous default; driver controls hop cadence.
    int32_t const spikeMaxGenerateLength = 200;

    // Streaming worker stdin loop. Use poll() with 1 s timeout so we can fire
    // an idle-timeout sweep between events (§15.6 step 4).
    //
    // Step 3 — sweep ALL slots (not just slot 0). Any slot whose session has
    // been idle past kIdleTimeoutMs is force-closed: emit a timeout event then
    // release the slot back to the pool and drop its routing mapping. Snapshot
    // the (sessionId, idleMs) of timed-out slots first, then release outside
    // the gather to keep the release path identical to the end/error path.
    auto const checkIdleTimeout = []() {
        auto const now = std::chrono::steady_clock::now();
        for (auto& slotPtr : gPool->slots())
        {
            AsrSessionState& s = slotPtr->session;
            if (!s.active)
            {
                continue;
            }
            auto const idleMs
                = std::chrono::duration_cast<std::chrono::milliseconds>(now - s.lastActivity).count();
            if (idleMs > kIdleTimeoutMs)
            {
                std::string const sid = s.sessionId;
                int32_t const slotId = slotPtr->slotId;
                releaseSlot(slotId);   // freeSession + mark slot free
                unbindSession(sid);    // drop sessionId → slot mapping
                Json ev = {{"event", "timeout"}, {"id", sid}, {"idle_ms", idleMs}};
                std::cout << ev.dump() << std::endl;
            }
        }
    };

    std::string buffer;
    char readBuf[4096];
    bool eof = false;
    while (!eof)
    {
        struct pollfd pfd = {STDIN_FILENO, POLLIN, 0};
        int const pr = ::poll(&pfd, 1, 1000);
        if (pr < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            LOG_ERROR("poll() failed: %s", std::strerror(errno));
            break;
        }
        if (pr == 0)
        {
            checkIdleTimeout();
            continue;
        }
        if (pfd.revents & (POLLIN | POLLHUP))
        {
            auto const n = ::read(STDIN_FILENO, readBuf, sizeof(readBuf));
            if (n <= 0)
            {
                eof = true;
                break;
            }
            buffer.append(readBuf, static_cast<size_t>(n));
        }
        // Process whole lines accumulated in the buffer.
        size_t pos;
        while ((pos = buffer.find('\n')) != std::string::npos)
        {
            std::string const line = buffer.substr(0, pos);
            buffer.erase(0, pos + 1);
            if (line.empty())
            {
                continue;
            }

            Json parsed;
            try
            {
                parsed = Json::parse(line);
            }
            catch (std::exception const& e)
            {
                Json err = {{"event", "error"}, {"ok", false},
                    {"error", std::string("json_parse_failed: ") + e.what()}};
                std::cout << err.dump() << std::endl;
                // Step 3 — malformed input carries no parseable sessionId, so we
                // cannot target a specific slot to drop. The idle-timeout sweep
                // reaps any genuinely stuck slot after kIdleTimeoutMs; emitting
                // the error and continuing is sufficient.
                continue;
            }

            if (!parsed.contains("event"))
            {
                // Backward-compat one-shot: any line that omits `event` flows
                // through the M2 legacy path. handleRequest behavior unchanged.
                handleOneShot(std::move(parsed), *runtime, stream, loraWeightsMap);
                continue;
            }

            std::string const event = parsed.value("event", "");
            if (event == "begin")
            {
                // Step 3 — routes via acquireSlot + binds sessionId → slot.
                handleBegin(parsed);
            }
            else if (event == "chunk")
            {
                // Step 3 — routes via lookupSlot to the slot bound at begin.
                handleChunk(parsed, loraWeightsMap, stageCounters, spikeMaxGenerateLength);
            }
            else if (event == "end")
            {
                // Step 3 — routes via lookupSlot + releases the slot.
                handleEnd(parsed);
            }
            else
            {
                Json err = {{"event", "error"}, {"ok", false}, {"error", "unknown_event"},
                    {"received", event}};
                std::string const evId = parsed.value("id", "");
                if (!evId.empty())
                {
                    err["id"] = evId;
                }
                std::cout << err.dump() << std::endl;
                // Step 3 — unknown event: if it names a live session, release
                // that slot for hygiene per §15.6 step 4 error-path cleanup.
                if (!evId.empty())
                {
                    int32_t const evSlot = lookupSlot(evId);
                    if (evSlot >= 0)
                    {
                        releaseSlot(evSlot);
                        unbindSession(evId);
                    }
                }
            }
        }
        checkIdleTimeout();
    }

    // D1 (Step 2): tear down ALL slots (streams + runtimes), not just slot 0.
    // `stream`/`runtime`/`session` above are references INTO the pool, so we do
    // NOT destroy them individually here (that would double-free slot 0).
    destroySlotPool();
    return EXIT_SUCCESS;
}
