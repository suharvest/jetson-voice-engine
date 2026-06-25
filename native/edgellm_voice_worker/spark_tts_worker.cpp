/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Seeed / jetson-voice-engine. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SparkTTS controllable-TTS resident C++ worker.
 *
 * S1: non-streaming N=1, per-slot resource structure.
 * S2: LLM decode driven by edge-llm StreamChannel (incremental per-token consumption),
 *     d_vector triggered the moment 32 bicodec_global tokens are collected, mid-stream
 *     cancel via {"type":"cancel","id":...}, and a hard max-token cap as a runaway safety
 *     valve (the mixed-precision LLM can fail to emit EOS on some ZH inputs). S2 does NOT
 *     yet chunk the vocoder (that is S3): semantic tokens are collected incrementally and
 *     vocoded as one segment once decode terminates.
 *
 * Pipeline (all TRT, no Python round-trip — spec docs/specs/sparktts-streaming-worker.md §4):
 *   stdin JSONL {id,text,gender,pitch,speed, top_k/temperature/top_p, max_tokens, ...}
 *     → controllable prompt (raw, no chat template)
 *     → [decode thread] LLMInferenceRuntime.handleRequest with an attached StreamChannel
 *     → [consumer] drain StreamChannel per iteration → parse delta tokenIds by id range:
 *         global  id∈[151665,155760] → index = id-151665   (collect 32 → trigger d_vector)
 *         semantic id∈[155761,163952] → index = id-155761
 *     → speaker_decoder engine (global[1,1,32]→d_vector[1,1024]) the moment 32 global arrive
 *     → BiCodec engine (semantic[1,T]+d_vector[1,1024]→16kHz wav) after decode finishes
 *     → stdout JSONL: token-progress chunks + done {output_file/audio_b64,...} | cancelled | error
 *
 * Concurrency (§5.3): SlotPool<SparkTTSSlot>. N=1 for S1/S2, but every per-request mutable
 * resource (LLM runtime, speaker_decoder + BiCodec IExecutionContext, CUDA stream, I/O
 * device buffers, d_vector buffer, StreamChannel) is owned PER-SLOT — shared engines are
 * read-only. No singleton/class-level mutable state (avoids the StatefulCode2WavRunner
 * concurrent-reset class of bug, MEMORY tts_n2_throughput_investigation). N>1 (S4) reuses
 * this structure unchanged; the StreamChannel consumer is already per-slot.
 *
 * Cancel: the plain LLMInferenceRuntime observes cancellation through the StreamChannel
 * (channel->cancel() → applyCancellationToFinishStates at the top of each decode iteration,
 * llmInferenceRuntime.cpp:709,729). cancelMap routes {"type":"cancel","id":X} to the slot's
 * in-flight channel. StreamChannelFinalizer guarantees the consumer always unblocks.
 */

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/slotPool.h"
#include "runtime/streaming.h"

#include <NvInfer.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
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
    std::string llmEngineDir;
    std::string speakerDecoderEngine;
    std::string bicodecEngine;
    int32_t maxSlots{1};
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
              << "  {\"id\":\"1\",\"text\":\"你好。\",\"gender\":\"female\",\"pitch\":\"moderate\",\"speed\":\"moderate\"}\n"
              << "  {\"type\":\"cancel\",\"id\":\"1\"}\n"
              << "Writes JSON lines to stdout (event=ready|token_progress|done|cancelled|error).\n";
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
int genderId(std::string const& g)
{
    if (g == "male" || g == "1") return 1;
    return 0;
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
std::string buildControllablePrompt(
    std::string const& text, std::string const& gender, std::string const& pitch, std::string const& speed)
{
    std::string attr = "<|gender_" + std::to_string(genderId(gender)) + "|><|pitch_label_"
        + std::to_string(levelId(pitch)) + "|><|speed_label_" + std::to_string(levelId(speed)) + "|>";
    return "<|task_controllable_tts|><|start_content|>" + text + "<|end_content|><|start_style_label|>" + attr
        + "<|end_style_label|>";
}

// --------------------------------------------------------------------------- bicodec token id ranges
// SparkTTS bicodec tokens are `normalized` non-special added tokens -> tokenizer decode()
// drops them, so we parse from token IDs (sparktts_llm_mixed_engine/tokenizer.json):
//   global  : id [151665, 155760] -> index = id - 151665   (4096 codebook)
//   semantic: id [155761, 163952] -> index = id - 155761   (8192 codebook)
constexpr int32_t kGlobalIdBase = 151665;
constexpr int32_t kGlobalIdEnd = 155760;
constexpr int32_t kSemanticIdBase = 155761;
constexpr int32_t kSemanticIdEnd = 163952;

// --------------------------------------------------------------------------- raw TRT helpers
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

// --------------------------------------------------------------------------- PER-SLOT state (§5.3)
struct SparkTTSSlot
{
    int32_t slotId{0};
    std::atomic<bool> inUse{false};
    std::atomic<bool> shutdown{false};

    // Per-slot LLM runtime (S1/S2 N=1: each slot owns one; for N>1 (S4) slots 1..N-1 share
    // slot 0's engine weights via the shared-engine ctor — structure already per-slot).
    std::unique_ptr<LLMInferenceRuntime> llm;
    cudaStream_t llmStream{nullptr};

    // Per-slot execution contexts over the SHARED read-only vocoder engines.
    cudaStream_t vocStream{nullptr};
    nvinfer1::IExecutionContext* spkCtx{nullptr};
    nvinfer1::IExecutionContext* bicodecCtx{nullptr};
    std::vector<float> dVector; // [1024]

    // Single-slot handoff queue (stdin reader -> this slot's worker thread).
    std::mutex queueMu;
    std::condition_variable queueCv;
    std::deque<Json> queue;
    std::thread worker;

    // The StreamChannel of the in-flight request (for cancel routing). Guarded by chanMu.
    std::mutex chanMu;
    std::shared_ptr<StreamChannel> activeChannel;

    SparkTTSSlot() = default;
    SparkTTSSlot(SparkTTSSlot const&) = delete;
    SparkTTSSlot& operator=(SparkTTSSlot const&) = delete;
    ~SparkTTSSlot()
    {
        if (spkCtx) delete spkCtx;
        if (bicodecCtx) delete bicodecCtx;
        if (vocStream) cudaStreamDestroy(vocStream);
        if (llmStream) cudaStreamDestroy(llmStream);
    }
};

// --------------------------------------------------------------------------- stdout serialization
std::mutex gCoutMutex;
void emitEvent(Json payload)
{
    std::string const line = payload.dump();
    std::lock_guard<std::mutex> lk(gCoutMutex);
    std::cout << line << std::endl;
}

// --------------------------------------------------------------------------- cancel routing (id -> slot's channel)
std::mutex gCancelMu;
std::unordered_map<std::string, SparkTTSSlot*> gCancelMap;
void registerCancel(std::string const& id, SparkTTSSlot* slot)
{
    std::lock_guard<std::mutex> lk(gCancelMu);
    gCancelMap[id] = slot;
}
void unregisterCancel(std::string const& id)
{
    std::lock_guard<std::mutex> lk(gCancelMu);
    gCancelMap.erase(id);
}
bool tripCancel(std::string const& id)
{
    std::shared_ptr<StreamChannel> chan;
    {
        std::lock_guard<std::mutex> lk(gCancelMu);
        auto it = gCancelMap.find(id);
        if (it == gCancelMap.end()) return false;
        SparkTTSSlot* slot = it->second;
        std::lock_guard<std::mutex> clk(slot->chanMu);
        chan = slot->activeChannel;
    }
    if (!chan) return false;
    chan->cancel(); // observed by the decode loop next iteration; wakes the blocked consumer
    return true;
}

// --------------------------------------------------------------------------- vocoder engine runs (per-slot ctx)
void runSpeakerDecoder(SparkTTSSlot& slot, nvinfer1::ICudaEngine* eng, std::vector<int32_t> const& global32)
{
    if (global32.size() != 32)
        throw std::runtime_error("speaker_decoder expects 32 global tokens, got " + std::to_string(global32.size()));
    std::string const inName = tensorName(eng, nvinfer1::TensorIOMode::kINPUT);
    std::string const outName = tensorName(eng, nvinfer1::TensorIOMode::kOUTPUT);
    std::vector<int64_t> idx(global32.begin(), global32.end());
    slot.spkCtx->setInputShape(inName.c_str(), nvinfer1::Dims3{1, 1, 32});
    DevBuf dIn, dOut;
    dIn.alloc(idx.size() * sizeof(int64_t));
    dOut.alloc(1024 * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(dIn.ptr, idx.data(), idx.size() * sizeof(int64_t), cudaMemcpyHostToDevice, slot.vocStream));
    slot.spkCtx->setTensorAddress(inName.c_str(), dIn.ptr);
    slot.spkCtx->setTensorAddress(outName.c_str(), dOut.ptr);
    if (!slot.spkCtx->enqueueV3(slot.vocStream)) throw std::runtime_error("speaker_decoder enqueueV3 failed");
    CUDA_CHECK(cudaMemcpyAsync(slot.dVector.data(), dOut.ptr, 1024 * sizeof(float), cudaMemcpyDeviceToHost, slot.vocStream));
    CUDA_CHECK(cudaStreamSynchronize(slot.vocStream));
}

std::vector<float> runBicodec(SparkTTSSlot& slot, nvinfer1::ICudaEngine* eng, std::vector<int32_t> const& semantic)
{
    std::string const semTensor = "semantic_tokens";
    std::string const dvTensor = "d_vector";
    std::string const outName = tensorName(eng, nvinfer1::TensorIOMode::kOUTPUT, 0);
    int64_t const T = static_cast<int64_t>(semantic.size());
    std::vector<int64_t> sem64(semantic.begin(), semantic.end());
    slot.bicodecCtx->setInputShape(semTensor.c_str(), nvinfer1::Dims2{1, T});
    slot.bicodecCtx->setInputShape(dvTensor.c_str(), nvinfer1::Dims2{1, 1024});
    DevBuf dSem, dDv, dOut;
    dSem.alloc(sem64.size() * sizeof(int64_t));
    dDv.alloc(1024 * sizeof(float));
    nvinfer1::Dims const outDims = slot.bicodecCtx->getTensorShape(outName.c_str());
    int64_t L = 1;
    for (int d = 0; d < outDims.nbDims; ++d) L *= outDims.d[d];
    dOut.alloc(static_cast<size_t>(L) * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(dSem.ptr, sem64.data(), sem64.size() * sizeof(int64_t), cudaMemcpyHostToDevice, slot.vocStream));
    CUDA_CHECK(cudaMemcpyAsync(dDv.ptr, slot.dVector.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice, slot.vocStream));
    slot.bicodecCtx->setTensorAddress(semTensor.c_str(), dSem.ptr);
    slot.bicodecCtx->setTensorAddress(dvTensor.c_str(), dDv.ptr);
    slot.bicodecCtx->setTensorAddress(outName.c_str(), dOut.ptr);
    if (!slot.bicodecCtx->enqueueV3(slot.vocStream)) throw std::runtime_error("BiCodec enqueueV3 failed");
    std::vector<float> wav(static_cast<size_t>(L));
    CUDA_CHECK(cudaMemcpyAsync(wav.data(), dOut.ptr, static_cast<size_t>(L) * sizeof(float), cudaMemcpyDeviceToHost, slot.vocStream));
    CUDA_CHECK(cudaStreamSynchronize(slot.vocStream));
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

// --------------------------------------------------------------------------- S3 overlap-chunk streaming vocoder
// Option A overlap-chunk (spec §3): the BiCodec decoder has no I/O state (causal conv-upsample
// [8,5,4,2], 320 samples/token, raw TRT — no left-context tensor). To vocode incrementally
// without boundary glitches we re-vocode each chunk WITH a left overlap of already-emitted
// tokens, discard the overlap region's audio, and crossfade the first few ms into the tail of
// the previously emitted audio to mask the seam. All per-request streaming state (emittedTokens,
// tail buffer) lives in this struct on the stack of processRequest → strictly per-slot
// (§5.3): no shared mutable vocoder state, avoids the StatefulCode2WavRunner concurrent-reset bug.
struct ChunkStreamState
{
    int32_t emittedTokens{0};          //!< semantic tokens already turned into emitted audio
    int32_t chunkIndex{0};
    int64_t emittedSamples{0};
    std::vector<float> crossfadeTail;  //!< last kFadeSamples of previously emitted audio (for crossfade)
};

constexpr int32_t kUpsamplePerToken = 320; // 16kHz / 50Hz token rate (bicodec_decoder config)
constexpr int32_t kFadeSamples = 160;      // 10ms @16k crossfade at chunk seams

// Vocode semantic[windowStart .. end) with the slot's d_vector, discard the left-overlap output,
// crossfade the seam, and return the samples to emit for this chunk.
std::vector<float> vocodeChunkWindow(SparkTTSSlot& slot, nvinfer1::ICudaEngine* eng,
    std::vector<int32_t> const& semantic, ChunkStreamState& st, int32_t leftOverlap, int32_t endToken)
{
    int32_t const windowStart = std::max(0, st.emittedTokens - leftOverlap);
    int32_t const skipTokens = st.emittedTokens - windowStart; // overlap tokens to discard
    std::vector<int32_t> window(semantic.begin() + windowStart, semantic.begin() + endToken);
    std::vector<float> full = runBicodec(slot, eng, window); // [skipTokens+new]*320 samples
    int64_t const skipSamples = static_cast<int64_t>(skipTokens) * kUpsamplePerToken;
    if (skipSamples >= static_cast<int64_t>(full.size())) return {};
    std::vector<float> out(full.begin() + skipSamples, full.end());

    // Crossfade the seam: blend the first kFadeSamples of `out` with the saved tail of the
    // previous chunk (equal-power-ish linear fade) to suppress residual boundary clicks.
    if (!st.crossfadeTail.empty() && out.size() >= static_cast<size_t>(kFadeSamples))
    {
        int32_t const n = std::min<int32_t>(kFadeSamples, static_cast<int32_t>(st.crossfadeTail.size()));
        for (int32_t i = 0; i < n; ++i)
        {
            float const a = static_cast<float>(i) / n; // 0→1 ramp for new chunk
            out[i] = a * out[i] + (1.0f - a) * st.crossfadeTail[i];
        }
    }
    // Save new tail for the next seam.
    if (out.size() >= static_cast<size_t>(kFadeSamples))
        st.crossfadeTail.assign(out.end() - kFadeSamples, out.end());
    else
        st.crossfadeTail = out;

    st.emittedTokens = endToken;
    return out;
}

void emitAudioChunk(std::string const& id, ChunkStreamState& st, std::vector<float> const& samples,
    bool isFinal, std::string const& chunkTransport, std::chrono::steady_clock::time_point reqStart)
{
    if (samples.empty() && !isFinal) return;
    Json ev = Json{{"id", id}, {"event", "chunk"}, {"ok", true}, {"chunk_index", st.chunkIndex},
        {"samples", samples.size()}, {"sample_rate", 16000}, {"is_final", isFinal},
        {"elapsed_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - reqStart).count()}};
    auto pcm = floatToPcm16(samples);
    if (chunkTransport == "base64")
    {
        ev["audio_b64"] = base64(reinterpret_cast<uint8_t const*>(pcm.data()), pcm.size() * sizeof(int16_t));
        ev["bytes"] = pcm.size() * sizeof(int16_t);
    }
    else
    {
        ev["bytes"] = pcm.size() * sizeof(int16_t); // file/none transport: caller assembles from done WAV
    }
    emitEvent(std::move(ev));
    st.emittedSamples += static_cast<int64_t>(samples.size());
    ++st.chunkIndex;
}

// --------------------------------------------------------------------------- shared engines (read-only)
nvinfer1::ICudaEngine* gSpkEngine{nullptr};
nvinfer1::ICudaEngine* gBicodecEngine{nullptr};
int32_t gSampleRate{16000};

// --------------------------------------------------------------------------- process ONE request (S2 streaming)
void processRequest(SparkTTSSlot& slot, Json const& item)
{
    std::string const id = item.value("id", "");
    auto const reqStart = std::chrono::steady_clock::now();

    auto channel = StreamChannel::create();
    channel->setSkipSpecialTokens(false); // we parse from tokenIds, not text
    {
        std::lock_guard<std::mutex> clk(slot.chanMu);
        slot.activeChannel = channel;
    }
    if (!id.empty()) registerCancel(id, &slot);

    std::thread decodeThread;
    try
    {
        std::string const text = item.value("text", "");
        std::string const gender = item.value("gender", "female");
        std::string const pitch = item.value("pitch", "moderate");
        std::string const speed = item.value("speed", "moderate");
        std::string const chunkTransport = item.value("chunk_transport", "file");
        bool const emitTokenProgress = item.value("emit_token_progress", true);
        int32_t const maxSemantic = item.value("max_semantic", 600); // BiCodec dynamic-T profile cap
        // S3 streaming vocoder (Option A overlap-chunk). Defaults on; set stream_audio=false for
        // single-segment (S2) behavior. firstChunkTokens kept small for low TTFA.
        bool const streamAudio = item.value("stream_audio", true);
        int32_t const firstChunkTokens = std::max(1, item.value("first_chunk_tokens", 12));
        int32_t const chunkTokens = std::max(1, item.value("chunk_tokens", 16));
        int32_t const leftOverlap = std::max(0, item.value("left_overlap_tokens", 12));
        std::string outputFile = item.value("output_file", "");
        if (outputFile.empty()) outputFile = "/tmp/spark_tts_worker_" + id + ".wav";
        if (text.empty()) throw std::runtime_error("empty text");

        LLMGenerationRequest req;
        LLMGenerationRequest::Request r;
        Message msg;
        msg.role = "user";
        Message::MessageContent c;
        c.type = "text";
        c.content = buildControllablePrompt(text, gender, pitch, speed);
        msg.contents.push_back(std::move(c));
        r.messages.push_back(std::move(msg));
        req.requests.push_back(std::move(r));
        req.temperature = item.value("temperature", 1.0f);
        req.topP = item.value("top_p", 1.0f);
        req.topK = item.value("top_k", 1);
        // Hard runaway cap: the mixed-precision LLM can fail to emit EOS on some ZH inputs.
        // maxGenerateLength bounds decode regardless; the runtime also clamps to KV capacity
        // ("Reduce max generation length"). 32 global + ~700 semantic covers a clean utterance.
        req.maxGenerateLength = item.value("max_tokens", 800);
        req.applyChatTemplate = item.value("apply_chat_template", false);
        req.addGenerationPrompt = item.value("add_generation_prompt", false);
        req.enableThinking = false;
        req.streamChannels = {channel};

        std::string decodeErr;
        decodeThread = std::thread([&]() {
            try
            {
                LLMGenerationResponse resp;
                slot.llm->handleRequest(req, resp, slot.llmStream);
            }
            catch (std::exception const& e)
            {
                decodeErr = e.what();
            }
        });

        // ---- Consumer: drain StreamChannel, parse tokens incrementally ----
        std::vector<int32_t> global;
        std::vector<int32_t> semantic;
        bool dVectorReady = false;
        int32_t emittedTokens = 0;
        std::chrono::steady_clock::time_point dVectorAt{};
        std::chrono::steady_clock::time_point firstAudioAt{}; // S3 TTFA marker
        bool semanticTruncated = false;
        FinishReason finalReason = FinishReason::kNotFinished;
        bool finished = false;

        // S3 streaming-vocode state — strictly per-request, on this slot's stack (§5.3).
        ChunkStreamState chunkState;
        std::vector<float> wavAccum; // assembled audio = concatenation of every emitted chunk

        // Vocode + emit ONE overlap-chunk [emittedTokens, endToken); append to wavAccum. isFinal
        // marks the terminal chunk. Returns true if a chunk was emitted.
        auto emitOneChunk = [&](int32_t endToken, bool isFinal) -> bool {
            if (endToken <= chunkState.emittedTokens) return false;
            std::vector<float> cs = vocodeChunkWindow(slot, gBicodecEngine, semantic, chunkState, leftOverlap, endToken);
            if (firstAudioAt.time_since_epoch().count() == 0) firstAudioAt = std::chrono::steady_clock::now();
            wavAccum.insert(wavAccum.end(), cs.begin(), cs.end());
            emitAudioChunk(id, chunkState, cs, isFinal, chunkTransport, reqStart);
            return true;
        };

        while (!finished)
        {
            std::optional<StreamChunk> chunk = channel->waitPop(std::chrono::milliseconds{100});
            if (chunk.has_value())
            {
                for (int32_t tid : chunk->tokenIds)
                {
                    if (tid >= kGlobalIdBase && tid <= kGlobalIdEnd)
                    {
                        if (global.size() < 32) global.push_back(tid - kGlobalIdBase);
                    }
                    else if (tid >= kSemanticIdBase && tid <= kSemanticIdEnd)
                    {
                        if (static_cast<int32_t>(semantic.size()) < maxSemantic)
                            semantic.push_back(tid - kSemanticIdBase);
                        else
                            semanticTruncated = true;
                    }
                    ++emittedTokens;
                }

                // Trigger d_vector the instant 32 global tokens are collected (S2 goal 2).
                if (!dVectorReady && global.size() == 32)
                {
                    runSpeakerDecoder(slot, gSpkEngine, global);
                    dVectorReady = true;
                    dVectorAt = std::chrono::steady_clock::now();
                    if (emitTokenProgress)
                        emitEvent(Json{{"id", id}, {"event", "token_progress"}, {"ok", true},
                            {"stage", "d_vector_ready"}, {"n_global", 32}, {"n_semantic", semantic.size()},
                            {"elapsed_ms", std::chrono::duration<double, std::milli>(dVectorAt - reqStart).count()}});
                }

                // S3: as soon as d_vector is ready and a chunk's worth of semantic has arrived,
                // vocode + emit it (overlap-chunk). This is the TTFA win. Not final here — more
                // tokens may still arrive; the terminal chunk is emitted in the post-decode flush.
                if (streamAudio && dVectorReady)
                {
                    int32_t const want = chunkState.chunkIndex == 0 ? firstChunkTokens : chunkTokens;
                    if (static_cast<int32_t>(semantic.size()) - chunkState.emittedTokens >= want)
                        emitOneChunk(chunkState.emittedTokens + want, /*isFinal=*/false);
                }

                if (chunk->finished)
                {
                    finalReason = chunk->reason;
                    finished = true;
                }
            }
            else if (channel->isFinished() || channel->isCancelled())
            {
                finalReason = channel->getReason();
                finished = true;
            }
        }

        if (decodeThread.joinable()) decodeThread.join();

        // ---- Cancelled: terminal cancelled event, no audio ----
        if (finalReason == FinishReason::kCancelled || channel->isCancelled())
        {
            emitEvent(Json{{"id", id}, {"event", "cancelled"}, {"ok", true}, {"reason", "cancelled"},
                {"n_global", global.size()}, {"n_semantic", semantic.size()}, {"tokens_decoded", emittedTokens},
                {"total_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - reqStart).count()}});
        }
        else
        {
            if (!decodeErr.empty()) throw std::runtime_error("LLM decode failed: " + decodeErr);
            if (global.size() != 32)
                throw std::runtime_error("expected 32 global tokens, parsed " + std::to_string(global.size()));
            if (semantic.empty()) throw std::runtime_error("no semantic tokens parsed");
            if (!dVectorReady) runSpeakerDecoder(slot, gSpkEngine, global);

            auto const vocStart = std::chrono::steady_clock::now();
            std::vector<float> wav; // assembled full audio (for file output + validation)

            if (streamAudio)
            {
                // S3 streaming path: drain remaining semantic as overlap-chunks; the LAST one is
                // marked is_final. Live chunks were already emitted during decode into wavAccum.
                int32_t const total = static_cast<int32_t>(semantic.size());
                bool lastWasFinal = false;
                while (chunkState.emittedTokens < total)
                {
                    int32_t const want = chunkState.chunkIndex == 0 ? firstChunkTokens : chunkTokens;
                    int32_t const endToken = std::min(total, chunkState.emittedTokens + want);
                    bool const isFinal = endToken >= total;
                    emitOneChunk(endToken, isFinal);
                    lastWasFinal = isFinal;
                }
                // Guarantee a terminal is_final chunk even if live emission already consumed all
                // tokens (exact-multiple edge): emit an empty final marker so consumers terminate.
                if (!lastWasFinal)
                    emitAudioChunk(id, chunkState, std::vector<float>{}, /*isFinal=*/true, chunkTransport, reqStart);
                wav = std::move(wavAccum);
            }
            else
            {
                wav = runBicodec(slot, gBicodecEngine, semantic); // S2 single-segment
            }
            auto const vocEnd = std::chrono::steady_clock::now();

            double rms = 0.0;
            for (float v : wav) rms += static_cast<double>(v) * v;
            rms = wav.empty() ? 0.0 : std::sqrt(rms / wav.size());
            double const audioS = static_cast<double>(wav.size()) / gSampleRate;
            double const ttfaMs = firstAudioAt.time_since_epoch().count() != 0
                ? std::chrono::duration<double, std::milli>(firstAudioAt - reqStart).count()
                : 0.0;

            Json done = Json{{"id", id}, {"event", "done"}, {"ok", true}, {"frames", semantic.size()},
                {"n_global", global.size()}, {"tokens_decoded", emittedTokens}, {"streamed", streamAudio},
                {"chunk_count", chunkState.chunkIndex}, {"ttfa_ms", ttfaMs},
                {"semantic_truncated", semanticTruncated}, {"finish_reason", finishReasonName(finalReason)},
                {"samples", wav.size()}, {"sample_rate", gSampleRate}, {"audio_s", audioS}, {"rms", rms},
                {"d_vector_ms", dVectorReady ? std::chrono::duration<double, std::milli>(dVectorAt - reqStart).count() : 0.0},
                {"vocoder_ms", std::chrono::duration<double, std::milli>(vocEnd - vocStart).count()},
                {"total_ms", std::chrono::duration<double, std::milli>(vocEnd - reqStart).count()}};
            // Always write the assembled WAV (used for ASR/parity validation + non-stream callers).
            if (!saveWav16(outputFile, wav, gSampleRate)) throw std::runtime_error("failed to save WAV: " + outputFile);
            done["output_file"] = outputFile;
            if (chunkTransport == "base64" && !streamAudio)
            {
                auto pcm = floatToPcm16(wav);
                done["audio_b64"] = base64(reinterpret_cast<uint8_t const*>(pcm.data()), pcm.size() * sizeof(int16_t));
                done["bytes"] = pcm.size() * sizeof(int16_t);
            }
            emitEvent(std::move(done));
        }
    }
    catch (std::exception const& e)
    {
        channel->cancel(); // unblock decode loop so the thread can be joined
        if (decodeThread.joinable()) decodeThread.join();
        emitEvent(Json{{"id", id}, {"event", "error"}, {"ok", false}, {"error", e.what()}});
    }

    if (!id.empty()) unregisterCancel(id);
    {
        std::lock_guard<std::mutex> clk(slot.chanMu);
        slot.activeChannel.reset();
    }
}

// --------------------------------------------------------------------------- slot worker thread
void slotWorkerLoop(SparkTTSSlot* slot)
{
    while (true)
    {
        Json item;
        {
            std::unique_lock<std::mutex> lk(slot->queueMu);
            slot->queueCv.wait(lk, [&] { return !slot->queue.empty() || slot->shutdown.load(); });
            if (slot->shutdown.load() && slot->queue.empty()) return;
            item = std::move(slot->queue.front());
            slot->queue.pop_front();
        }
        processRequest(*slot, item);
        slot->inUse.store(false, std::memory_order_release); // request complete, slot free
    }
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
    gSampleRate = args.sampleRate;

    auto const t0 = std::chrono::steady_clock::now();

    nvinfer1::IRuntime* trtRuntime = nvinfer1::createInferRuntime(gLogger);
    gSpkEngine = deserializeEngine(trtRuntime, args.speakerDecoderEngine);
    gBicodecEngine = deserializeEngine(trtRuntime, args.bicodecEngine);

    SlotPool<SparkTTSSlot> pool(args.maxSlots);
    std::unordered_map<std::string, std::string> loraWeights;
    for (int32_t s = 0; s < args.maxSlots; ++s)
    {
        auto slot = std::make_unique<SparkTTSSlot>();
        slot->slotId = s;
        CUDA_CHECK(cudaStreamCreate(&slot->llmStream));
        CUDA_CHECK(cudaStreamCreate(&slot->vocStream));
        slot->llm = std::make_unique<LLMInferenceRuntime>(args.llmEngineDir, "", loraWeights, slot->llmStream);
        if (!slot->llm->captureDecodingCUDAGraph(slot->llmStream))
            LOG_WARNING("CUDA graph capture failed for SparkTTS LLM slot %d, proceeding without.", s);
        slot->spkCtx = gSpkEngine->createExecutionContext();
        slot->bicodecCtx = gBicodecEngine->createExecutionContext();
        if (!slot->spkCtx || !slot->bicodecCtx) throw std::runtime_error("createExecutionContext failed");
        slot->dVector.assign(1024, 0.0f);
        pool.slots().push_back(std::move(slot));
    }
    for (auto& slotPtr : pool.slots())
        slotPtr->worker = std::thread(slotWorkerLoop, slotPtr.get());

    double const initMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    emitEvent(Json{{"event", "ready"}, {"init_ms", initMs}, {"max_slots", args.maxSlots}, {"streaming", true}});

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty()) continue;
        Json item;
        try
        {
            item = Json::parse(line);
        }
        catch (std::exception const& e)
        {
            emitEvent(Json{{"event", "error"}, {"ok", false}, {"error", std::string("invalid JSON: ") + e.what()}});
            continue;
        }

        if (item.is_object() && item.value("type", "") == "cancel")
        {
            std::string const cid = item.value("id", "");
            bool const tripped = tripCancel(cid);
            emitEvent(Json{{"event", "cancel_ack"}, {"id", cid}, {"tripped", tripped}});
            continue;
        }

        std::string const id = item.value("id", "");
        int32_t const slotId = pool.acquireFree();
        if (slotId < 0)
        {
            Json ev = {{"event", "error"}, {"ok", false}, {"error", "pool_saturated"}, {"status", 4429},
                {"max_slots", args.maxSlots}};
            if (!id.empty()) ev["id"] = id;
            emitEvent(std::move(ev));
            continue;
        }
        if (!id.empty()) pool.bind(id, slotId);
        SparkTTSSlot& slot = *pool.get(slotId);
        {
            std::lock_guard<std::mutex> lk(slot.queueMu);
            slot.queue.push_back(std::move(item));
        }
        slot.queueCv.notify_one();
    }

    // EOF: shutdown workers, join, free GPU (contexts before engines).
    for (auto& slotPtr : pool.slots())
    {
        slotPtr->shutdown.store(true);
        slotPtr->queueCv.notify_all();
    }
    for (auto& slotPtr : pool.slots())
        if (slotPtr->worker.joinable()) slotPtr->worker.join();
    for (auto& slotPtr : pool.slots())
        slotPtr->llm.reset();
    pool.slots().clear();
    if (gBicodecEngine) delete gBicodecEngine;
    if (gSpkEngine) delete gSpkEngine;
    if (trtRuntime) delete trtRuntime;
    return EXIT_SUCCESS;
}
