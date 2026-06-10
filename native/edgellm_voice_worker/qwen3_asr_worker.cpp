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
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
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

//! v0.8.0 migration: streaming begin/chunk/end is dormant in production
//! (stream_mode=accumulate uses the one-shot `requests` path). The v0.7.1 slot
//! pool + spec-decode append-prefill machinery that backed the streaming path
//! was removed from the v0.8.0 engine baseline, so streaming returns a 501 stub.
//! Re-enabling it requires porting onto AsrStreamingSessionRuntime (Option A).
void emitStreamingNotSupported(Json const& input)
{
    Json ev = {{"event", "error"}, {"ok", false},
        {"error", "streaming_not_supported_v080_oneshot_worker"}, {"status", 501}};
    std::string const id = input.value("id", "");
    if (!id.empty())
    {
        ev["id"] = id;
    }
    std::cout << ev.dump() << std::endl;
}

void handleBegin(Json const& input) { emitStreamingNotSupported(input); }
void handleChunk(Json const& input) { emitStreamingNotSupported(input); }
void handleEnd(Json const& input) { emitStreamingNotSupported(input); }

void handleOneShot(Json input, rt::LLMInferenceRuntime& runtime, cudaStream_t stream,
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
    double const initMs
        = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - initStart).count();
    std::cout << Json{{"event", "ready"}, {"init_ms", initMs}, {"max_slots", 1}}.dump() << std::endl;

    // D1 (Step 2): build the N-slot decoder pool. Slot 0 deserializes the base
    // engine once; slots 1..N-1 SHARE that ICudaEngine via the fork
    // shared-engine constructor (weight memory paid once, contexts independent).
    // Each slot owns its own CUDA stream. CUDA-graph capture is per-slot.

    // Step 3: begin/chunk/end now route by sessionId across ALL slots (see the
    // legacy one-shot path (lines with no `event` field) keeps using slot 0's
    // runtime + stream — it is a single serial request flow and never touches
    // the slot pool's session state.

    // Streaming worker stdin loop. Use poll() with 1 s timeout so we can fire
    // an idle-timeout sweep between events (§15.6 step 4).
    //
    // Step 3 — sweep ALL slots (not just slot 0). Any slot whose session has
    // been idle past kIdleTimeoutMs is force-closed: emit a timeout event then
    // release the slot back to the pool and drop its routing mapping. Snapshot
    // the (sessionId, idleMs) of timed-out slots first, then release outside
    // the gather to keep the release path identical to the end/error path.
    auto const checkIdleTimeout = []() {};

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
                handleChunk(parsed);
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
