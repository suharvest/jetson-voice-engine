/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "common/trtUtils.h"
#include "requestFileParser.h"
#include "runtime/llmInferenceSpecDecodeRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <optional>
#include <string>
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
    bool debug{false};
};

// ---------------------------------------------------------------------------
// M2 streaming-ASR session scaffold (design doc §12 milestone 2).
//
// M2 introduces session lifecycle (beginAsrSession/endAsrSession) and KV
// capacity caps in the runtime layer. The worker stays one-shot for now —
// the event-driven dispatcher lands in M3. M2 only:
//   1. Declares the single-session state struct + an in-process session table
//      stub so M3 can plug the begin/chunk/end dispatcher in without further
//      header churn.
//   2. Wires the structured KV-capacity error event through, so if a future
//      runtime->appendPrefillEmbeds call (M3+) returns false with status
//      kKvCapacityExceeded, the worker emits the explicit
//      {"event":"error","ok":false,"error":"kv_capacity_exceeded",...} event
//      instead of the generic "error" string.
//
// Idle-session timeout policy (full enforcement lands in M3):
//   - A session is considered idle when (now - last_activity) > 30 seconds.
//   - On the next request that touches that session, the worker shall
//     endAsrSession the stale slot and emit
//     {"event":"error","ok":false,"error":"session_timeout",...}.
//   - M2 leaves last_activity bookkeeping wired into the struct; M3 adds the
//     dispatcher tick that enforces the rule.
// ---------------------------------------------------------------------------
struct AsrSessionState
{
    std::string sessionId;                                         //!< Stable ID emitted by the client at begin.
    std::chrono::steady_clock::time_point lastActivity{};          //!< Updated on every chunk/end touching the slot.
    bool active{false};                                            //!< True between begin and end.
};

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

//! Maps a runtime AppendPrefillStatus to the structured worker-side JSON event.
//! Returns std::nullopt when the status is kOk (caller emits the normal
//! response). For M2 the only non-Ok status the worker surfaces is
//! kKvCapacityExceeded — other failure modes still flow through the generic
//! error path until M3 plumbs the chunked dispatcher in.
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
    default:
        // M3 will expand the structured-error coverage. For now fall through to
        // the generic error path.
        return std::nullopt;
    }
}

enum OptionId : int
{
    HELP = 1000,
    ENGINE_DIR,
    MULTIMODAL_ENGINE_DIR,
    DEBUG,
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " --engineDir=<path> --multimodalEngineDir=<path> [--debug]\n\n"
              << "Reads llm_inference-compatible JSON lines from stdin and writes JSON lines to stdout.\n";
}

bool parseArgs(Args& args, int argc, char** argv)
{
    static struct option options[] = {{"help", no_argument, 0, HELP},
        {"engineDir", required_argument, 0, ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, MULTIMODAL_ENGINE_DIR}, {"debug", no_argument, 0, DEBUG},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case HELP: printUsage(argv[0]); std::exit(EXIT_SUCCESS);
        case ENGINE_DIR: args.engineDir = optarg; break;
        case MULTIMODAL_ENGINE_DIR: args.multimodalEngineDir = optarg; break;
        case DEBUG: args.debug = true; break;
        default: return false;
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

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto const initStart = std::chrono::steady_clock::now();
    std::unordered_map<std::string, std::string> loraWeightsMap;
    auto runtime = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(
        args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream);
    bool const enableGraph = std::getenv("EDGE_LLM_ASR_CUDA_GRAPH") == nullptr
        || std::string(std::getenv("EDGE_LLM_ASR_CUDA_GRAPH")) != "0";
    if (enableGraph && !runtime->captureDecodingCUDAGraph(stream))
    {
        LOG_WARNING("CUDA graph capture failed for ASR worker, proceeding without.");
    }
    double const initMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - initStart).count();
    std::cout << Json{{"event", "ready"}, {"init_ms", initMs}}.dump() << std::endl;

    // M2 session table stub. M2 worker is still one-shot, so this map stays
    // empty in the M2 build — M3 populates it from {"event":"begin"} /
    // {"event":"chunk"} / {"event":"end"} messages on stdin.
    std::unordered_map<std::string, AsrSessionState> sessions;
    (void) sessions; // suppress -Wunused until M3 wires the dispatcher.

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
        {
            continue;
        }

        Json response;
        std::filesystem::path tempPath;
        auto const requestStart = std::chrono::steady_clock::now();
        try
        {
            Json input = Json::parse(line);
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
                bool const requestOk = runtime->handleRequest(batchedRequests[requestIdx], llmResponse, stream);
                ok = ok && requestOk;
                for (size_t batchIdx = 0; batchIdx < batchedRequests[requestIdx].requests.size(); ++batchIdx)
                {
                    bool const hasOutputText = requestOk && batchIdx < llmResponse.outputTexts.size();
                    std::string const text
                        = hasOutputText ? llmResponse.outputTexts[batchIdx] : "TensorRT Edge LLM cannot handle this request. Fails.";
                    responses.push_back(Json{{"request_idx", requestIdx},
                        {"batch_idx", batchIdx},
                        {"output_text", text}});
                }
            }
            double const totalMs
                = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - requestStart).count();
            // M2: if the underlying runtime tagged the failure with a streaming
            // status code (currently only kKvCapacityExceeded propagates),
            // emit the structured event instead of the generic "error" string.
            // For M2 handleRequest never sets the append status, so this
            // branch is dead code at runtime — but the wiring is here so M3
            // can plug appendPrefillEmbeds straight in.
            if (!ok)
            {
                if (auto structuredEv = mapAppendStatusToErrorEvent(*runtime, id))
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

    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
