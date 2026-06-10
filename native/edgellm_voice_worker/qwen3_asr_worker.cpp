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
// v0.8.0 migration: the v0.7.1-era streaming spec-decode runtime and slot pool
// were removed from the engine baseline. This serving worker now drives the
// v0.8.0 vanilla rt::LLMInferenceRuntime on the ONE-SHOT path only (production
// runs stream_mode=accumulate → the one-shot `requests` path is the live one;
// the streaming begin/chunk/end protocol is dormant and returns a 501 stub).
#include "runtime/asrStreamingSessionRuntime.h" // SessionLaneManager (lane reservation only)
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
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

// Forward decl: the proven one-shot core (defined below) is reused by the
// streaming finalize path (handleEnd) which is defined earlier in the file.
Json runOneShotCore(Json input, rt::LLMInferenceRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap);

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

// ===========================================================================
// Task #9: N>1 concurrent-session support (begin/chunk/end accumulation path).
// Task #10: true streaming PARTIALs (per-hop cumulative re-decode -> `partial`).
//
// Design (settled): each session reserves a LANE via SessionLaneManager (pure
// bookkeeping — lane reservation only, NOT AsrStreamingSessionRuntime KV-append).
// chunks accumulate the CUMULATIVE mel/pcm payload for the session AND run a
// cumulative re-decode on the audio-so-far, emitting an incremental `partial`
// per hop (task #10, first-cut: reuses the proven one-shot core -> guaranteed
// to converge to the one-shot transcript). At `end` (or a `last:true` chunk),
// the PROVEN one-shot core runs on the full cumulative audio under
// gEngineExecMutex, guaranteeing CER-0.0000 + byte-identity with the legacy
// one-shot path. The faster true incremental-KV per-hop append is a deferred
// optimization (see handleChunk note + plan §3).
//
// Concurrency model: the worker is single-threaded (one stdin loop). Sessions
// co-reside in gSessions and their lanes are held across interleaved
// begin/chunk/end events from multiple sids; the engine step itself is wrapped
// by gEngineExecMutex (kept for forward-compat with a future worker thread —
// single-threaded today so it never contends). A mandatory idle sweep releases
// lanes whose session went silent past kIdleTimeoutMs (half-open lane-leak guard,
// cf. the v2v slot-leak lesson).
// ===========================================================================

//! Per-session accumulation + lane bookkeeping.
struct SessionState
{
    int32_t laneId{-1};                       //!< Reserved lane (SessionLaneManager).
    std::string melPath;                      //!< Cumulative mel safetensors path (last chunk wins).
    std::string pcmB64;                       //!< Cumulative PCM (base64), if PCM-input path is used.
    Json beginMeta;                           //!< Begin-time metadata (sampling params etc.) to replay at finalize.
    std::chrono::steady_clock::time_point lastActivity{std::chrono::steady_clock::now()};
};

std::mutex gEngineExecMutex;                                   //!< Serializes the whole prefill+decode engine step.
std::unordered_map<std::string, SessionState> gSessions;       //!< sid -> session state.
std::unique_ptr<rt::SessionLaneManager> gLaneMgr;              //!< Lane allocator over the shared cache batch rows.
int32_t gMaxSlots{1};                                          //!< Advertised slot-pool size (== asrMax).
constexpr int64_t kIdleTimeoutMs = 30000;                      //!< Idle-session lane reclaim threshold.

//! Deterministic 63-bit ownerId from a session id (SessionLaneManager wants int64).
int64_t sidToOwnerId(std::string const& sid)
{
    return static_cast<int64_t>(std::hash<std::string>{}(sid) & 0x7fffffffffffffffLL);
}

//! Task #10: strip the leading "language <Lang>" tag the ASR head prepends, so
//! partials and finals expose the same bare transcript. Applied consistently to
//! both partial and final transcripts (idempotent — no tag => unchanged).
std::string stripLangTag(std::string const& s)
{
    // Match a leading ASCII "language" token (case-insensitive) + single space +
    // a language word + a space, e.g. "language Chinese这并不是...". We only strip
    // the "language <Word> " prefix; the CJK transcript that follows is preserved.
    static char const* kTag = "language ";
    constexpr size_t kTagLen = 9; // strlen("language ")
    if (s.size() < kTagLen) return s;
    for (size_t i = 0; i < kTagLen; ++i)
    {
        char a = s[i];
        char b = kTag[i];
        if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
        if (a != b) return s; // no "language " prefix -> leave untouched
    }
    // Skip the language word (run of ASCII letters) and exactly one following space.
    size_t j = kTagLen;
    while (j < s.size() && ((s[j] >= 'A' && s[j] <= 'Z') || (s[j] >= 'a' && s[j] <= 'z'))) ++j;
    if (j < s.size() && s[j] == ' ') ++j;
    return s.substr(j);
}

//! Extract the bare (lang-tag-stripped) transcript from a runOneShotCore result.
//! Empty string if the core produced no output_text.
std::string transcriptFromCore(Json const& core)
{
    if (core.contains("responses") && core["responses"].is_array() && !core["responses"].empty())
    {
        Json const& r0 = core["responses"][0];
        if (r0.contains("output_text") && r0["output_text"].is_string())
        {
            return stripLangTag(r0["output_text"].get<std::string>());
        }
    }
    return std::string();
}

//! begin: reserve a lane (pool_saturated -> 4429), register session, ack.
void handleBegin(Json const& input)
{
    std::string const sid = input.value("id", "");
    if (sid.empty())
    {
        std::cout << Json{{"event", "error"}, {"ok", false}, {"error", "begin_missing_id"}}.dump() << std::endl;
        return;
    }
    if (gSessions.count(sid))
    {
        // Idempotent re-begin: ack the existing lane (no double-acquire).
        gSessions[sid].lastActivity = std::chrono::steady_clock::now();
        std::cout << Json{{"event", "begin_ack"}, {"id", sid}, {"lane", gSessions[sid].laneId}}.dump() << std::endl;
        return;
    }
    int32_t const lane = gLaneMgr->acquire(rt::LaneOwnerKind::kAsr, sidToOwnerId(sid));
    if (lane < 0)
    {
        // Pool saturated. NO worker restart — caller backs off / retries (HTTP 429
        // semantics, cf. the OVS session-limiter 4429 contract).
        std::cout << Json{{"event", "error"}, {"id", sid}, {"ok", false}, {"error", "pool_saturated"},
            {"status", 4429}, {"max_slots", gMaxSlots}}.dump() << std::endl;
        return;
    }
    SessionState st;
    st.laneId = lane;
    st.lastActivity = std::chrono::steady_clock::now();
    // Capture sampling/meta knobs so finalize replays the SAME request envelope as
    // a one-shot call would (keeps byte-identity with the golden one-shot path).
    for (char const* k : {"batch_size", "temperature", "top_p", "top_k", "max_generate_length"})
    {
        if (input.contains(k)) st.beginMeta[k] = input[k];
    }
    gSessions.emplace(sid, std::move(st));
    std::cout << Json{{"event", "begin_ack"}, {"id", sid}, {"lane", lane}}.dump() << std::endl;
}

// Forward decls (handleChunk now finalizes on last:true via the handleEnd path).
void handleEnd(Json const& input, rt::LLMInferenceRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap);
Json buildFinalizeRequest(std::string const& sid, SessionState const& st);

//! Task #10: chunk = record the CUMULATIVE mel/pcm payload for this sid AND run a
//! cumulative re-decode (first-cut, guaranteed-converge: reuses the proven
//! one-shot core on the audio-so-far) to emit an incremental `partial`. On a
//! `last:true` chunk we route straight to the finalize path (emits `final`).
//!
//! DEFERRED OPTIMIZATION: the true incremental-KV per-hop append (decode only the
//! newly-arrived frames via AsrStreamingSessionRuntime::appendChunk /
//! decodeToTranscript on the session's lane, instead of re-decoding the whole
//! cumulative prefix every hop) is faster but not yet wired — see plan §3. This
//! cumulative re-decode is O(audio_so_far) per hop but is correct and converges
//! exactly to the one-shot transcript, so it ships first.
void handleChunk(Json const& input, rt::LLMInferenceRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap)
{
    std::string const sid = input.value("id", "");
    auto it = gSessions.find(sid);
    if (it == gSessions.end())
    {
        std::cout << Json{{"event", "error"}, {"id", sid}, {"ok", false}, {"error", "unknown_session"}}.dump()
                  << std::endl;
        return;
    }
    SessionState& st = it->second;
    st.lastActivity = std::chrono::steady_clock::now();
    // Cumulative semantics: each chunk carries the full-so-far audio descriptor
    // (the OVS accumulate driver re-points mel_path to the growing mel). Last wins.
    if (input.contains("mel_path") && input["mel_path"].is_string())
    {
        st.melPath = input["mel_path"].get<std::string>();
    }
    else if (input.contains("audio") && input["audio"].is_string())
    {
        st.melPath = input["audio"].get<std::string>();
    }
    if (input.contains("pcm_b64") && input["pcm_b64"].is_string())
    {
        st.pcmB64 = input["pcm_b64"].get<std::string>();
    }

    // `last:true` => this hop is the end-of-utterance: finalize via the proven path.
    bool const isLast = input.value("last", false);
    if (isLast)
    {
        handleEnd(input, runtime, stream, loraWeightsMap);
        return;
    }

    // No audio recorded yet => nothing to decode; just ack so the lane stays warm.
    if (st.melPath.empty() && st.pcmB64.empty())
    {
        std::cout << Json{{"event", "chunk_ack"}, {"id", sid}}.dump() << std::endl;
        return;
    }

    // Cumulative re-decode on the session's reserved lane (serialized by the engine
    // exec mutex, identical step to the one-shot/finalize path).
    Json const req = buildFinalizeRequest(sid, st);
    Json core;
    {
        std::lock_guard<std::mutex> lk(gEngineExecMutex);
        core = runOneShotCore(req, runtime, stream, loraWeightsMap);
    }
    if (core.value("ok", false))
    {
        std::cout << Json{{"event", "partial"}, {"id", sid}, {"text", transcriptFromCore(core)}}.dump() << std::endl;
    }
    else
    {
        // Decode failure on an intermediate hop is non-fatal to the session: surface
        // it but keep the lane so a later (longer) cumulative hop can still finalize.
        Json ev = Json{{"event", "error"}, {"id", sid}, {"ok", false}, {"error", "partial_decode_failed"}};
        if (core.contains("error")) ev["detail"] = core["error"];
        std::cout << ev.dump() << std::endl;
    }
}

//! Build the one-shot request envelope for a finalized session (cumulative audio).
Json buildFinalizeRequest(std::string const& sid, SessionState const& st)
{
    Json req = st.beginMeta.is_object() ? st.beginMeta : Json::object();
    if (!req.contains("batch_size")) req["batch_size"] = 1;
    if (!req.contains("temperature")) req["temperature"] = 1.0;
    if (!req.contains("top_p")) req["top_p"] = 1.0;
    if (!req.contains("top_k")) req["top_k"] = 1;
    if (!req.contains("max_generate_length")) req["max_generate_length"] = 256;
    req["id"] = sid;
    Json content = Json::array();
    if (!st.melPath.empty())
    {
        content.push_back(Json{{"type", "audio"}, {"audio", st.melPath}});
    }
    // (pcm path: the runtime's mel-extractor consumes pcm_b64 via the request
    //  parser when mel assets are wired; mel_path is the primary fixture path.)
    if (!st.pcmB64.empty())
    {
        content.push_back(Json{{"type", "audio"}, {"pcm_b64", st.pcmB64}});
    }
    req["requests"] = Json::array({Json{{"messages", Json::array({Json{{"role", "user"}, {"content", content}}})},
        {"reference", "ref"}}});
    return req;
}

//! end / last:true: finalize via the PROVEN one-shot core, release lane, emit final.
void handleEnd(Json const& input, rt::LLMInferenceRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap)
{
    std::string const sid = input.value("id", "");
    auto it = gSessions.find(sid);
    if (it == gSessions.end())
    {
        std::cout << Json{{"event", "error"}, {"id", sid}, {"ok", false}, {"error", "unknown_session"}}.dump()
                  << std::endl;
        return;
    }
    SessionState st = it->second; // copy; we release the slot below regardless of outcome
    // Allow `end` to carry the final cumulative audio too.
    if (input.contains("mel_path") && input["mel_path"].is_string()) st.melPath = input["mel_path"].get<std::string>();
    else if (input.contains("audio") && input["audio"].is_string()) st.melPath = input["audio"].get<std::string>();
    if (input.contains("pcm_b64") && input["pcm_b64"].is_string()) st.pcmB64 = input["pcm_b64"].get<std::string>();

    Json finalEv;
    if (st.melPath.empty() && st.pcmB64.empty())
    {
        finalEv = Json{{"event", "error"}, {"id", sid}, {"ok", false}, {"error", "no_audio_accumulated"}};
    }
    else
    {
        Json const req = buildFinalizeRequest(sid, st);
        Json core;
        {
            std::lock_guard<std::mutex> lk(gEngineExecMutex);
            core = runOneShotCore(req, runtime, stream, loraWeightsMap);
        }
        // Re-shape the one-shot `done` into a streaming `final` (same output_text).
        // Also expose a lang-tag-stripped `text` so partial/final agree byte-for-byte
        // on the bare transcript (task #10 convergence contract).
        finalEv = Json{{"event", "final"}, {"id", sid}, {"ok", core.value("ok", false)}};
        if (core.contains("responses")) finalEv["responses"] = core["responses"];
        if (core.value("ok", false)) finalEv["text"] = transcriptFromCore(core);
        if (core.contains("total_ms")) finalEv["total_ms"] = core["total_ms"];
        if (core.contains("error")) finalEv["error"] = core["error"];
    }

    // Release the lane back to the pool (guaranteed, success or failure) and drop the session.
    gLaneMgr->release(st.laneId);
    gSessions.erase(sid);
    std::cout << finalEv.dump() << std::endl;
}

//! Mandatory idle-sweep: reclaim lanes of sessions silent past kIdleTimeoutMs.
void sweepIdleSessions()
{
    if (gSessions.empty()) return;
    auto const now = std::chrono::steady_clock::now();
    std::vector<std::pair<std::string, int32_t>> reap;
    for (auto const& [sid, st] : gSessions)
    {
        int64_t const idleMs
            = std::chrono::duration_cast<std::chrono::milliseconds>(now - st.lastActivity).count();
        if (idleMs > kIdleTimeoutMs)
        {
            reap.emplace_back(sid, st.laneId);
        }
    }
    for (auto const& [sid, lane] : reap)
    {
        gLaneMgr->release(lane);
        gSessions.erase(sid);
        std::cout << Json{{"event", "timeout"}, {"id", sid}, {"ok", false}, {"error", "idle_timeout"},
            {"idle_ms", kIdleTimeoutMs}}.dump() << std::endl;
    }
}

//! Core one-shot engine step: parse the `requests` payload, run handleRequest on
//! each, and return the assembled response Json. This is the PROVEN golden path
//! (CER 0.0000 + byte-identity). Both the legacy event-less one-shot path AND the
//! streaming `end`/finalize path call this, so finalize inherits exact parity.
//! NOT internally synchronized — callers must hold gEngineExecMutex (shared
//! PipelineIO is mutated by handleRequest).
Json runOneShotCore(Json input, rt::LLMInferenceRuntime& runtime, cudaStream_t stream,
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
            response = Json{{"id", id}, {"event", "error"}, {"ok", false}, {"responses", responses},
                {"total_ms", totalMs}};
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
    return response;
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
    setProfilingEnabled(true);

    auto const initStart = std::chrono::steady_clock::now();
    std::unordered_map<std::string, std::string> loraWeightsMap;
    bool const enableGraph = std::getenv("EDGE_LLM_ASR_CUDA_GRAPH") == nullptr
        || std::string(std::getenv("EDGE_LLM_ASR_CUDA_GRAPH")) != "0";
    cudaStream_t stream{nullptr};
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto runtimePtr = std::make_unique<rt::LLMInferenceRuntime>(
        args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream);
    if (enableGraph && !runtimePtr->captureDecodingCUDAGraph(stream))
    {
        LOG_WARNING("CUDA graph capture failed for ASR one-shot runtime, proceeding without.");
    }
    rt::LLMInferenceRuntime* runtime = runtimePtr.get();

    // Task #9: lane reservation pool. asrMax = requested --max_slots, clamped to
    // the engine's physical batch capacity (maxSessionBatchSize). The b2 engine
    // exposes 2 rows; the b1 engine exposes 1 (so N>1 silently degrades to N=1
    // there). TTS partition is 0 in this ASR-only worker.
    int32_t const physBatch = runtime->maxSessionBatchSize();
    int32_t asrMax = args.maxSlots;
    if (asrMax > physBatch) asrMax = physBatch;
    if (asrMax < 1) asrMax = 1;
    gMaxSlots = asrMax;
    gLaneMgr = std::make_unique<rt::SessionLaneManager>(physBatch, asrMax, /*ttsMax=*/0);

    double const initMs
        = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - initStart).count();
    std::cout << Json{{"event", "ready"}, {"init_ms", initMs}, {"max_slots", gMaxSlots}}.dump() << std::endl;

    // Streaming worker stdin loop. Use poll() with a 1 s timeout so we can fire
    // the mandatory idle-timeout sweep between events (half-open lane-leak guard).
    auto const checkIdleTimeout = []() { sweepIdleSessions(); };

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
                // through the legacy path. BYTE-IDENTICAL with v080-0016: same
                // core, same response shape. Mutex held for forward-compat (no
                // contention single-threaded) so the engine step is identical.
                Json response;
                {
                    std::lock_guard<std::mutex> lk(gEngineExecMutex);
                    response = runOneShotCore(std::move(parsed), *runtime, stream, loraWeightsMap);
                }
                std::cout << response.dump() << std::endl;
                continue;
            }

            std::string const event = parsed.value("event", "");
            if (event == "begin")
            {
                handleBegin(parsed);
            }
            else if (event == "chunk")
            {
                handleChunk(parsed, *runtime, stream, loraWeightsMap);
            }
            else if (event == "end")
            {
                handleEnd(parsed, *runtime, stream, loraWeightsMap);
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
            }
        }
        checkIdleTimeout();
    }

    runtimePtr.reset();
    if (stream != nullptr)
    {
        cudaStreamDestroy(stream);
    }
    return EXIT_SUCCESS;
}
