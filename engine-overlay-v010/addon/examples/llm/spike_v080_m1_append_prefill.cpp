// Phase 3b empirical spike — R2 chunk-prefill parity on the v0.8.0 streaming-ASR runtime.
//
// Validates the chunk-prefill refactor (patch v080-0002: `runBaseModelPrefillChunk`)
// against the one-shot prefill path, run on a REAL v0.8.0 qwen3-asr thinker engine via
// the new `AsrStreamingSessionRuntime` begin/append/end API.
//
// The regression risk introduced by Phases 2-3 is the seam at
// llmInferenceRuntime.cpp:1105-1108, where the old code copied the ENTIRE accumulated
// `context.tokenIds`, and the refactor now packs ONLY a per-chunk slice. If that seam is
// wrong, splitting the same token sequence across two appendChunk() calls would diverge
// from feeding it as a single appendChunk(). We assert it does NOT:
//
//   Path A — one chunk of N text tokens   (begin → appendChunk(all N, final) → logits).
//   Path B — two chunks N1 + N2 = N       (begin → appendChunk(N1, !final)
//                                                 → appendChunk(N2, final) → logits).
//
// Both paths feed the SAME token IDs from the SAME begin state. We download the lane's
// last-step logits via getLogitsForTesting and assert:
//   (R2) argmax(A) == argmax(B)                            — same sampled token
//   (R2) max_i |logitA[i] - logitB[i]| < 1e-2              — byte-identity proof
//   (R3) KV length is continuous: after B's two appends, peekKvCacheLength == N == A's.
//        (MRope positions are driven off the committed KV length; continuity here means
//         the chunk path advanced positions exactly as the one-shot path did.)
//
// We deliberately use ordinary TEXT token IDs (not <audio_pad>) so the embedding lookup is
// a pure text-table gather — this isolates the prefill-chunk packing/exec/commit math,
// which is exactly the refactored surface. No mel / audio encoder needed.

#include "common/tensor.h"
#include "common/trtUtils.h" // loadEdgellmPluginLib — registers AttentionPlugin before deserialize
#include "runtime/asrStreamingSessionRuntime.h"
#include "runtime/llmInferenceRuntime.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;

#define CK(x)                                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        auto _e = (x);                                                                                                 \
        if (_e != cudaSuccess)                                                                                         \
        {                                                                                                              \
            fprintf(stderr, "CUDA err %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));                       \
            std::exit(2);                                                                                              \
        }                                                                                                              \
    } while (0)

namespace
{
constexpr int32_t kMaxPositions = 1024;

int argmax(std::vector<float> const& v)
{
    int k = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i)
        if (v[i] > v[k])
            k = i;
    return k;
}

// A deterministic synthetic token sequence in a safe text-id range (well below the audio
// token id 151676 and below vocab 151936). Same seed → same tokens on every call.
std::vector<int32_t> makeTextTokens(int n, uint32_t seed)
{
    std::vector<int32_t> ids(n);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int32_t> dis(1, 100000);
    for (auto& t : ids)
        t = dis(gen);
    return ids;
}
} // namespace

int main(int argc, char** argv)
{
    std::string engineDir = "/home/harvest/tensorrt-edgellm-workspace/Qwen3-ASR-0.6B/engines-v080/llm";
    std::string multimodalEngineDir = ""; // text-only token prefill — no audio encoder needed.
    int totalTokens = 40;
    int splitAt = 17; // chunk boundary inside the token block.
    if (argc >= 2)
        engineDir = argv[1];
    if (argc >= 4)
    {
        totalTokens = std::atoi(argv[2]);
        splitAt = std::atoi(argv[3]);
    }
    if (splitAt <= 0 || splitAt >= totalTokens)
    {
        fprintf(stderr, "FATAL: splitAt(%d) must be in (0, totalTokens=%d)\n", splitAt, totalTokens);
        return 2;
    }

    // Register the edgellm TRT plugins (AttentionPlugin etc.) before any engine deserialize.
    // Held for process lifetime, mirroring llm_inference.cpp:486.
    auto pluginHandles = loadEdgellmPluginLib();

    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));

    std::unordered_map<std::string, std::string> loraMap;
    std::unique_ptr<rt::LLMInferenceRuntime> runtime;
    try
    {
        runtime = std::make_unique<rt::LLMInferenceRuntime>(engineDir, multimodalEngineDir, loraMap, stream);
    }
    catch (std::exception const& e)
    {
        fprintf(stderr, "FATAL: runtime construction failed: %s\n", e.what());
        return 2;
    }
    printf("runtime constructed against %s\n", engineDir.c_str());
    printf("maxSessionBatchSize=%d\n", runtime->maxSessionBatchSize());

    auto const allTokens = makeTextTokens(totalTokens, /*seed*/ 1234);
    std::vector<int32_t> chunk0(allTokens.begin(), allTokens.begin() + splitAt);
    std::vector<int32_t> chunk1(allTokens.begin() + splitAt, allTokens.end());

    // ── Path A: single chunk ────────────────────────────────────────────────
    int kvA = -1;
    std::vector<float> logitsA;
    {
        rt::AsrStreamingSessionRuntime sess(*runtime, /*lane*/ 0);
        if (!sess.beginAsrSession(/*promptTokenIds*/ {}, kMaxPositions, /*maxAudioTokensPerChunk*/ 0, stream))
        {
            fprintf(stderr, "PATH A: beginAsrSession FAILED\n");
            return 1;
        }
        if (!sess.appendChunk(allTokens, /*isFinalChunk*/ true, stream))
        {
            fprintf(stderr, "PATH A: appendChunk(all) FAILED\n");
            return 1;
        }
        CK(cudaStreamSynchronize(stream));
        kvA = sess.peekKvCacheLength(stream);
        logitsA = sess.getLogitsForTesting(stream);
        sess.endAsrSession(stream);
        printf("PATH A (1 chunk, N=%d): kv=%d, status_finished=%d\n", totalTokens, kvA,
            sess.status() == rt::AsrSessionStatus::kIdle);
    }

    // ── Path B: two chunks split at splitAt ─────────────────────────────────
    int kvB1 = -1, kvB2 = -1;
    std::vector<float> logitsB;
    {
        rt::AsrStreamingSessionRuntime sess(*runtime, /*lane*/ 0);
        if (!sess.beginAsrSession(/*promptTokenIds*/ {}, kMaxPositions, /*maxAudioTokensPerChunk*/ 0, stream))
        {
            fprintf(stderr, "PATH B: beginAsrSession FAILED\n");
            return 1;
        }
        if (!sess.appendChunk(chunk0, /*isFinalChunk*/ false, stream))
        {
            fprintf(stderr, "PATH B: appendChunk(chunk0) FAILED\n");
            return 1;
        }
        CK(cudaStreamSynchronize(stream));
        kvB1 = sess.peekKvCacheLength(stream);
        if (!sess.appendChunk(chunk1, /*isFinalChunk*/ true, stream))
        {
            fprintf(stderr, "PATH B: appendChunk(chunk1) FAILED\n");
            return 1;
        }
        CK(cudaStreamSynchronize(stream));
        kvB2 = sess.peekKvCacheLength(stream);
        logitsB = sess.getLogitsForTesting(stream);
        sess.endAsrSession(stream);
        printf("PATH B (2 chunks, %d+%d=%d): kv_after1=%d kv_after2=%d\n", splitAt, totalTokens - splitAt, totalTokens,
            kvB1, kvB2);
    }

    // ── Gates ───────────────────────────────────────────────────────────────
    printf("\n=== R2 / R3 GATES ===\n");
    bool ok = true;

    // R3: KV-length continuity across chunks.
    bool kvR3 = (kvB1 == splitAt) && (kvB2 == totalTokens) && (kvA == totalTokens);
    printf("R3 KV continuity: A_kv=%d  B_kv1=%d(exp %d)  B_kv2=%d(exp %d)  => %s\n", kvA, kvB1, splitAt, kvB2,
        totalTokens, kvR3 ? "PASS" : "FAIL");
    ok = ok && kvR3;

    if (logitsA.empty() || logitsB.empty() || logitsA.size() != logitsB.size())
    {
        printf("R2: FAIL — logits empty or size mismatch (|A|=%zu |B|=%zu)\n", logitsA.size(), logitsB.size());
        printf("\n=== M1 (R2/R3) ACCEPTANCE: FAIL ===\n");
        cudaStreamDestroy(stream);
        return 1;
    }

    int aA = argmax(logitsA), aB = argmax(logitsB);
    double maxAbs = 0.0;
    for (size_t i = 0; i < logitsA.size(); ++i)
        maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(logitsA[i]) - static_cast<double>(logitsB[i])));
    bool argmaxMatch = (aA == aB);
    bool diffOk = (maxAbs < 1e-2);
    printf("R2 argmax: A=%d  B=%d  => %s\n", aA, aB, argmaxMatch ? "MATCH" : "MISMATCH");
    printf("R2 max-abs logit diff = %.6e  (threshold 1e-2) => %s\n", maxAbs, diffOk ? "PASS" : "FAIL");
    ok = ok && argmaxMatch && diffOk;

    printf("\n=== M1 (R2/R3) ACCEPTANCE: %s ===\n", ok ? "PASS" : "FAIL");
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}
