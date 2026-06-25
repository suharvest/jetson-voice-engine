/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Seeed / jetson-voice-engine. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SparkTTS controllable-TTS resident C++ worker (S1: non-streaming, N=1, per-slot
 * resource structure in place for N>1).
 *
 * Pipeline (all TRT, no Python round-trip — see spec docs/specs/sparktts-streaming-worker.md §4):
 *   stdin JSON-line  {id,text,gender,pitch,speed, temperature/top_k/top_p, max_tokens, ...}
 *     → controllable prompt  <|task_controllable_tts|><|start_content|>TEXT<|end_content|>
 *                            <|start_style_label|><|gender_G|><|pitch_label_P|><|speed_label_S|><|end_style_label|>
 *     → LLMInferenceRuntime greedy decode (batch path) → outputText
 *     → regex parse 32 bicodec_global_*  +  T bicodec_semantic_*
 *     → speaker_decoder engine: global[1,1,32] int64 → d_vector[1,1024] f32
 *     → BiCodec vocoder engine: semantic[1,T] int64 + d_vector[1,1024] → wav[1,1,L] f32 @16kHz
 *     → stdout JSON-line  event=done {output_file/audio_b64, samples, ...}
 *
 * Concurrency (§5): SlotPool<SparkTTSSlot>. N=1 for S1, but every per-request mutable
 * resource (speaker_decoder + BiCodec IExecutionContext, CUDA stream, I/O device buffers,
 * d_vector buffer) is owned PER-SLOT, never as a singleton/class-level mutable — this is the
 * hard requirement that avoids the StatefulCode2WavRunner concurrent-reset class of bug
 * (memory MEMORY: tts_n2_throughput_investigation). The TRT engines themselves are SHARED
 * read-only; only execution contexts are per-slot. The LLM runtime is a single shared object
 * for S1/N=1; S2/S4 will give each slot its own LLM execution context (documented below).
 */

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/slotPool.h"

#include <NvInfer.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using tensorrt_edge_llm::runtime::SlotPool;
using Json = nlohmann::json;

namespace
{

// --------------------------------------------------------------------------- args
struct Args
{
    std::string llmEngineDir;            //!< mixed-precision SparkTTS LLM engine dir
    std::string speakerDecoderEngine;    //!< sparktts_speaker_decoder.fp32.engine (global ids -> d_vector)
    std::string bicodecEngine;           //!< bicodec_decoder_dynT.fp16.engine (semantic + d_vector -> wav)
    int32_t maxSlots{1};                 //!< per-slot concurrency (S1: 1; N>1 reserved)
    int32_t sampleRate{16000};
    bool debug{false};
};

enum OptionId : int
{
    HELP = 1000,
    LLM_ENGINE_DIR,
    SPEAKER_DECODER_ENGINE,
    BICODEC_ENGINE,
    MAX_SLOTS,
    SAMPLE_RATE,
    DEBUG,
};

void printUsage(char const* prog)
{
    std::cerr << "Usage: " << prog
              << " --llmEngineDir=<dir> --speakerDecoderEngine=<file.engine> --bicodecEngine=<file.engine>"
              << " [--maxSlots=N] [--sampleRate=16000] [--debug]\n\n"
              << "Reads JSON lines from stdin:\n"
              << "  {\"id\":\"1\",\"text\":\"你好。\",\"gender\":\"female\",\"pitch\":\"moderate\",\"speed\":\"moderate\","
                 "\"output_file\":\"/tmp/out.wav\"}\n"
              << "Writes JSON lines to stdout (event=ready|done|chunk|error).\n";
}

bool parseArgs(Args& a, int argc, char** argv)
{
    static struct option opts[] = {{"help", no_argument, 0, HELP}, {"llmEngineDir", required_argument, 0, LLM_ENGINE_DIR},
        {"speakerDecoderEngine", required_argument, 0, SPEAKER_DECODER_ENGINE},
        {"bicodecEngine", required_argument, 0, BICODEC_ENGINE}, {"maxSlots", required_argument, 0, MAX_SLOTS},
        {"sampleRate", required_argument, 0, SAMPLE_RATE}, {"debug", no_argument, 0, DEBUG}, {0, 0, 0, 0}};
    int opt;
    while ((opt = getopt_long(argc, argv, "", opts, nullptr)) != -1)
    {
        switch (opt)
        {
        case HELP: printUsage(argv[0]); std::exit(EXIT_SUCCESS);
        case LLM_ENGINE_DIR: a.llmEngineDir = optarg; break;
        case SPEAKER_DECODER_ENGINE: a.speakerDecoderEngine = optarg; break;
        case BICODEC_ENGINE: a.bicodecEngine = optarg; break;
        case MAX_SLOTS: a.maxSlots = std::max(1, std::atoi(optarg)); break;
        case SAMPLE_RATE: a.sampleRate = std::atoi(optarg); break;
        case DEBUG: a.debug = true; break;
        default: return false;
        }
    }
    return !a.llmEngineDir.empty() && !a.speakerDecoderEngine.empty() && !a.bicodecEngine.empty();
}

// --------------------------------------------------------------------------- controllable prompt
// Matches spike make_llm_input.py / single_in.py exactly (raw prompt, NO chat template).
int genderId(std::string const& g)
{
    if (g == "male" || g == "1") return 1;
    return 0; // female default
}
int levelId(std::string const& l)
{
    if (l == "very_low" || l == "0") return 0;
    if (l == "low" || l == "1") return 1;
    if (l == "moderate" || l == "2" || l.empty()) return 2;
    if (l == "high" || l == "3") return 3;
    if (l == "very_high" || l == "4") return 4;
    return 2;
}
std::string buildControllablePrompt(std::string const& text, std::string const& gender, std::string const& pitch,
    std::string const& speed)
{
    std::string attr = "<|gender_" + std::to_string(genderId(gender)) + "|><|pitch_label_"
        + std::to_string(levelId(pitch)) + "|><|speed_label_" + std::to_string(levelId(speed)) + "|>";
    return "<|task_controllable_tts|><|start_content|>" + text + "<|end_content|><|start_style_label|>" + attr
        + "<|end_style_label|>";
}

// --------------------------------------------------------------------------- token parse
// SparkTTS bicodec tokens are `normalized` non-special added tokens, so the C++ tokenizer's
// decode() does NOT render `<|bicodec_global_N|>` back into the output text — text-regex on
// outputText parses 0. We therefore parse directly from outputIds by token-id range, which is
// the robust contract (mirrors spike extract_mixed_tokens.py id->content map):
//   global  : id in [151665, 155760] -> index = id - 151665   (4096 codebook)
//   semantic: id in [155761, 163952] -> index = id - 155761   (8192 codebook)
// Ranges are taken from sparktts_llm_mixed_engine/tokenizer.json added_tokens.
constexpr int32_t kGlobalIdBase = 151665;
constexpr int32_t kGlobalIdEnd = 155760;
constexpr int32_t kSemanticIdBase = 155761;
constexpr int32_t kSemanticIdEnd = 163952;

struct ParsedTokens
{
    std::vector<int32_t> global;   // expect 32
    std::vector<int32_t> semantic; // T
};
ParsedTokens parseTokensFromIds(std::vector<int32_t> const& ids)
{
    ParsedTokens p;
    for (int32_t id : ids)
    {
        if (id >= kGlobalIdBase && id <= kGlobalIdEnd)
            p.global.push_back(id - kGlobalIdBase);
        else if (id >= kSemanticIdBase && id <= kSemanticIdEnd)
            p.semantic.push_back(id - kSemanticIdBase);
    }
    return p;
}

// --------------------------------------------------------------------------- raw TRT engine (shared, read-only)
nvinfer1::ICudaEngine* deserializeEngine(nvinfer1::IRuntime* runtime, std::string const& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open engine: " + path);
    std::streamsize const sz = f.tellg();
    f.seekg(0);
    std::vector<char> buf(static_cast<size_t>(sz));
    if (!f.read(buf.data(), sz)) throw std::runtime_error("Failed to read engine: " + path);
    auto* eng = runtime->deserializeCudaEngine(buf.data(), static_cast<size_t>(sz));
    if (!eng) throw std::runtime_error("deserializeCudaEngine failed: " + path);
    return eng;
}

std::string tensorName(nvinfer1::ICudaEngine* e, nvinfer1::TensorIOMode mode, int which = 0)
{
    int seen = 0;
    for (int i = 0; i < e->getNbIOTensors(); ++i)
    {
        char const* n = e->getIOTensorName(i);
        if (e->getTensorIOMode(n) == mode)
        {
            if (seen == which) return n;
            ++seen;
        }
    }
    throw std::runtime_error("IO tensor not found");
}

// --------------------------------------------------------------------------- PER-SLOT state (§5.3 hard requirement)
// Everything mutable per request lives here. For N>1 the SlotPool holds N of these; each slot
// runs end-to-end on its own CUDA stream + own execution contexts + own d_vector buffer. No
// singleton / class-level mutable state is permitted (avoids the StatefulCode2WavRunner
// concurrent-reset illegal-memory-access bug, MEMORY tts_n2_throughput_investigation).
struct SparkTTSSlot
{
    int32_t slotId{0};
    std::atomic<bool> inUse{false};

    cudaStream_t stream{nullptr};

    // Per-slot execution contexts over the SHARED engines (engines are read-only).
    nvinfer1::IExecutionContext* spkCtx{nullptr};      // speaker_decoder
    nvinfer1::IExecutionContext* bicodecCtx{nullptr};  // BiCodec vocoder

    // Per-slot reusable d_vector host buffer (computed each request from this slot's globals).
    std::vector<float> dVector; // [1024]

    SparkTTSSlot() = default;
    SparkTTSSlot(SparkTTSSlot const&) = delete;
    SparkTTSSlot& operator=(SparkTTSSlot const&) = delete;

    void init(nvinfer1::ICudaEngine* spkEngine, nvinfer1::ICudaEngine* bicodecEngine)
    {
        CUDA_CHECK(cudaStreamCreate(&stream));
        spkCtx = spkEngine->createExecutionContext();
        bicodecCtx = bicodecEngine->createExecutionContext();
        if (!spkCtx || !bicodecCtx) throw std::runtime_error("createExecutionContext failed");
        dVector.assign(1024, 0.0f);
    }
    ~SparkTTSSlot()
    {
        if (spkCtx) delete spkCtx;
        if (bicodecCtx) delete bicodecCtx;
        if (stream) cudaStreamDestroy(stream);
    }
};

// --------------------------------------------------------------------------- GPU device-buffer RAII helper
struct DevBuf
{
    void* ptr{nullptr};
    size_t bytes{0};
    void alloc(size_t n)
    {
        if (n > bytes)
        {
            if (ptr) cudaFree(ptr);
            CUDA_CHECK(cudaMalloc(&ptr, n));
            bytes = n;
        }
    }
    ~DevBuf()
    {
        if (ptr) cudaFree(ptr);
    }
};

// speaker_decoder: global int64[1,1,32] -> d_vector f32[1024] (into slot.dVector)
void runSpeakerDecoder(SparkTTSSlot& slot, nvinfer1::ICudaEngine* eng, std::vector<int32_t> const& global32)
{
    if (global32.size() != 32) throw std::runtime_error("speaker_decoder expects 32 global tokens, got "
        + std::to_string(global32.size()));
    std::string const inName = tensorName(eng, nvinfer1::TensorIOMode::kINPUT);
    std::string const outName = tensorName(eng, nvinfer1::TensorIOMode::kOUTPUT);

    std::vector<int64_t> idx(global32.begin(), global32.end()); // [32] int64
    slot.spkCtx->setInputShape(inName.c_str(), nvinfer1::Dims3{1, 1, 32});

    DevBuf dIn, dOut;
    dIn.alloc(idx.size() * sizeof(int64_t));
    dOut.alloc(1024 * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(dIn.ptr, idx.data(), idx.size() * sizeof(int64_t), cudaMemcpyHostToDevice, slot.stream));
    slot.spkCtx->setTensorAddress(inName.c_str(), dIn.ptr);
    slot.spkCtx->setTensorAddress(outName.c_str(), dOut.ptr);
    if (!slot.spkCtx->enqueueV3(slot.stream)) throw std::runtime_error("speaker_decoder enqueueV3 failed");
    CUDA_CHECK(cudaMemcpyAsync(slot.dVector.data(), dOut.ptr, 1024 * sizeof(float), cudaMemcpyDeviceToHost, slot.stream));
    CUDA_CHECK(cudaStreamSynchronize(slot.stream));
}

// BiCodec: semantic int64[1,T] + d_vector f32[1,1024] -> wav f32[1,1,L]; returns samples
std::vector<float> runBicodec(SparkTTSSlot& slot, nvinfer1::ICudaEngine* eng, std::vector<int32_t> const& semantic)
{
    std::string const semName = tensorName(eng, nvinfer1::TensorIOMode::kINPUT, 0);
    std::string const dvName = tensorName(eng, nvinfer1::TensorIOMode::kINPUT, 1);
    std::string const outName = tensorName(eng, nvinfer1::TensorIOMode::kOUTPUT, 0);
    // Engine IO order: input0=semantic_tokens, input1=d_vector (verified). Be name-robust:
    std::string semTensor = semName, dvTensor = dvName;
    if (semName == "d_vector" || dvName == "semantic_tokens")
    {
        semTensor = "semantic_tokens";
        dvTensor = "d_vector";
    }

    int64_t const T = static_cast<int64_t>(semantic.size());
    std::vector<int64_t> sem64(semantic.begin(), semantic.end());
    slot.bicodecCtx->setInputShape(semTensor.c_str(), nvinfer1::Dims2{1, T});
    slot.bicodecCtx->setInputShape(dvTensor.c_str(), nvinfer1::Dims2{1, 1024});

    DevBuf dSem, dDv, dOut;
    dSem.alloc(sem64.size() * sizeof(int64_t));
    dDv.alloc(1024 * sizeof(float));
    // Output shape resolved after input shapes are set.
    nvinfer1::Dims const outDims = slot.bicodecCtx->getTensorShape(outName.c_str());
    int64_t L = 1;
    for (int d = 0; d < outDims.nbDims; ++d) L *= outDims.d[d];
    dOut.alloc(static_cast<size_t>(L) * sizeof(float));

    CUDA_CHECK(cudaMemcpyAsync(dSem.ptr, sem64.data(), sem64.size() * sizeof(int64_t), cudaMemcpyHostToDevice, slot.stream));
    CUDA_CHECK(cudaMemcpyAsync(dDv.ptr, slot.dVector.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice, slot.stream));
    slot.bicodecCtx->setTensorAddress(semTensor.c_str(), dSem.ptr);
    slot.bicodecCtx->setTensorAddress(dvTensor.c_str(), dDv.ptr);
    slot.bicodecCtx->setTensorAddress(outName.c_str(), dOut.ptr);
    if (!slot.bicodecCtx->enqueueV3(slot.stream)) throw std::runtime_error("BiCodec enqueueV3 failed");
    std::vector<float> wav(static_cast<size_t>(L));
    CUDA_CHECK(cudaMemcpyAsync(wav.data(), dOut.ptr, static_cast<size_t>(L) * sizeof(float), cudaMemcpyDeviceToHost, slot.stream));
    CUDA_CHECK(cudaStreamSynchronize(slot.stream));
    return wav;
}

// --------------------------------------------------------------------------- WAV / base64
std::vector<int16_t> floatToPcm16(std::vector<float> const& s)
{
    std::vector<int16_t> p(s.size());
    for (size_t i = 0; i < s.size(); ++i)
    {
        float c = std::clamp(s[i], -1.0f, 1.0f);
        p[i] = static_cast<int16_t>(std::lrint(c * 32767.0f));
    }
    return p;
}

bool saveWav16(std::string const& path, std::vector<float> const& samples, int32_t sr)
{
    auto pcm = floatToPcm16(samples);
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t const dataBytes = static_cast<uint32_t>(pcm.size() * sizeof(int16_t));
    uint32_t const riff = 36 + dataBytes;
    uint16_t const ch = 1, bits = 16, fmt = 1;
    uint32_t const byteRate = sr * ch * bits / 8;
    uint16_t const blockAlign = ch * bits / 8;
    auto w32 = [&](uint32_t v) { f.write(reinterpret_cast<char const*>(&v), 4); };
    auto w16 = [&](uint16_t v) { f.write(reinterpret_cast<char const*>(&v), 2); };
    f.write("RIFF", 4); w32(riff); f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16); w16(fmt); w16(ch); w32(sr); w32(byteRate); w16(blockAlign); w16(bits);
    f.write("data", 4); w32(dataBytes);
    f.write(reinterpret_cast<char const*>(pcm.data()), dataBytes);
    return static_cast<bool>(f);
}

std::string base64(uint8_t const* d, size_t n)
{
    static constexpr char T[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    o.reserve(((n + 2) / 3) * 4);
    for (size_t i = 0; i < n; i += 3)
    {
        uint32_t b0 = d[i], b1 = i + 1 < n ? d[i + 1] : 0, b2 = i + 2 < n ? d[i + 2] : 0;
        uint32_t t = (b0 << 16) | (b1 << 8) | b2;
        o.push_back(T[(t >> 18) & 0x3F]);
        o.push_back(T[(t >> 12) & 0x3F]);
        o.push_back(i + 1 < n ? T[(t >> 6) & 0x3F] : '=');
        o.push_back(i + 2 < n ? T[t & 0x3F] : '=');
    }
    return o;
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

    cudaStream_t llmStream;
    CUDA_CHECK(cudaStreamCreate(&llmStream));

    auto const t0 = std::chrono::steady_clock::now();

    // --- Shared LLM runtime (S1/N=1: single shared object). ---
    // For N>1 (S2/S4) each slot will hold its own LLM execution context; the engine weights
    // stay shared. S1 keeps one runtime and serializes LLM decode behind the slot it owns.
    std::unordered_map<std::string, std::string> loraWeights;
    auto llmRuntime = std::make_unique<LLMInferenceRuntime>(args.llmEngineDir, "", loraWeights, llmStream);
    if (!llmRuntime->captureDecodingCUDAGraph(llmStream))
        LOG_WARNING("CUDA graph capture failed for SparkTTS LLM, proceeding without.");

    // --- Shared read-only TRT engines for vocoder path. ---
    nvinfer1::IRuntime* trtRuntime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* spkEngine = deserializeEngine(trtRuntime, args.speakerDecoderEngine);
    nvinfer1::ICudaEngine* bicodecEngine = deserializeEngine(trtRuntime, args.bicodecEngine);

    // --- SlotPool (§5): N constructed slots, each with its own contexts/stream/state. ---
    SlotPool<SparkTTSSlot> pool(args.maxSlots);
    for (int32_t s = 0; s < args.maxSlots; ++s)
    {
        auto slot = std::make_unique<SparkTTSSlot>();
        slot->slotId = s;
        slot->init(spkEngine, bicodecEngine);
        pool.slots().push_back(std::move(slot));
    }

    double const initMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << Json{{"event", "ready"}, {"init_ms", initMs}, {"max_slots", args.maxSlots}}.dump() << std::endl;

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty()) continue;
        Json response;
        auto const reqStart = std::chrono::steady_clock::now();
        try
        {
            Json item = Json::parse(line);
            std::string const id = item.value("id", "");
            std::string const text = item.value("text", "");
            std::string const gender = item.value("gender", "female");
            std::string const pitch = item.value("pitch", "moderate");
            std::string const speed = item.value("speed", "moderate");
            std::string const chunkTransport = item.value("chunk_transport", "file");
            std::string outputFile = item.value("output_file", "");
            if (outputFile.empty()) outputFile = "/tmp/spark_tts_worker_" + id + ".wav";
            if (text.empty()) throw std::runtime_error("empty text");

            // S1/N=1: always slot 0 (pool sized 1). N>1 will acquire a free slot here.
            SparkTTSSlot& slot = *pool.slots()[0];

            // ---- 1. LLM controllable decode (greedy, raw prompt) ----
            LLMGenerationRequest req;
            LLMGenerationRequest::Request r;
            Message msg;
            msg.role = "user"; // raw concat (applyChatTemplate=false): role label not emitted
            Message::MessageContent c;
            c.type = "text";
            c.content = buildControllablePrompt(text, gender, pitch, speed);
            msg.contents.push_back(std::move(c));
            r.messages.push_back(std::move(msg));
            req.requests.push_back(std::move(r));
            req.temperature = item.value("temperature", 1.0f);
            req.topP = item.value("top_p", 1.0f);
            req.topK = item.value("top_k", 1); // greedy default (matches spike)
            req.maxGenerateLength = item.value("max_tokens", 3000);
            req.applyChatTemplate = item.value("apply_chat_template", false);
            req.addGenerationPrompt = item.value("add_generation_prompt", false);
            req.enableThinking = false;

            auto const genStart = std::chrono::steady_clock::now();
            LLMGenerationResponse llmResp;
            if (!llmRuntime->handleRequest(req, llmResp, llmStream) || llmResp.outputIds.empty())
                throw std::runtime_error("LLM generation failed");
            auto const genEnd = std::chrono::steady_clock::now();

            // ---- 2. parse 32 global + T semantic (from output token ids) ----
            ParsedTokens tk = parseTokensFromIds(llmResp.outputIds[0]);
            if (tk.global.size() != 32)
                throw std::runtime_error("expected 32 global tokens, parsed " + std::to_string(tk.global.size()));
            if (tk.semantic.empty()) throw std::runtime_error("no semantic tokens parsed");

            // BiCodec engine dynamic-T profile is [8..600]. The mixed-precision LLM can run
            // away on some inputs (e.g. ZH greedy), producing >600 semantic tokens; the spike
            // CLI e2e (vocode_mixed.py) clamps to 600. Mirror that to (a) stay within the
            // vocoder profile and (b) align byte-for-byte with the CLI e2e reference. A clean
            // EOS-terminated decode stays well under 600 and is unaffected.
            int32_t const maxSemantic = item.value("max_semantic", 600);
            bool const semanticTruncated = static_cast<int32_t>(tk.semantic.size()) > maxSemantic;
            if (semanticTruncated) tk.semantic.resize(static_cast<size_t>(maxSemantic));

            // ---- 3. global -> d_vector (speaker_decoder engine, per-slot ctx) ----
            runSpeakerDecoder(slot, spkEngine, tk.global);

            // ---- 4. d_vector + semantic -> wav (BiCodec engine, per-slot ctx) ----
            auto const vocStart = std::chrono::steady_clock::now();
            std::vector<float> wav = runBicodec(slot, bicodecEngine, tk.semantic);
            auto const vocEnd = std::chrono::steady_clock::now();

            // ---- 5. emit ----
            double rms = 0.0;
            for (float v : wav) rms += static_cast<double>(v) * v;
            rms = wav.empty() ? 0.0 : std::sqrt(rms / wav.size());
            double const audioS = static_cast<double>(wav.size()) / args.sampleRate;

            response = Json{{"id", id}, {"event", "done"}, {"ok", true}, {"frames", tk.semantic.size()},
                {"n_global", tk.global.size()}, {"semantic_truncated", semanticTruncated},
                {"samples", wav.size()}, {"sample_rate", args.sampleRate}, {"audio_s", audioS}, {"rms", rms},
                {"generation_ms", std::chrono::duration<double, std::milli>(genEnd - genStart).count()},
                {"vocoder_ms", std::chrono::duration<double, std::milli>(vocEnd - vocStart).count()},
                {"total_ms", std::chrono::duration<double, std::milli>(vocEnd - reqStart).count()}};

            if (chunkTransport == "base64")
            {
                auto pcm = floatToPcm16(wav);
                response["audio_b64"] = base64(reinterpret_cast<uint8_t const*>(pcm.data()), pcm.size() * sizeof(int16_t));
                response["bytes"] = pcm.size() * sizeof(int16_t);
            }
            else
            {
                if (!saveWav16(outputFile, wav, args.sampleRate))
                    throw std::runtime_error("failed to save WAV: " + outputFile);
                response["output_file"] = outputFile;
            }
        }
        catch (std::exception const& e)
        {
            response = Json{{"event", "error"}, {"ok", false}, {"error", e.what()}};
        }
        std::cout << response.dump() << std::endl;
    }

    // Destroy per-slot execution contexts BEFORE the engines that created them (TRT requires
    // context lifetime < engine lifetime, else API Usage Error 3).
    pool.slots().clear();
    // shared engines / runtime cleanup
    if (bicodecEngine) delete bicodecEngine;
    if (spkEngine) delete spkEngine;
    if (trtRuntime) delete trtRuntime;
    CUDA_CHECK(cudaStreamDestroy(llmStream));
    return EXIT_SUCCESS;
}
