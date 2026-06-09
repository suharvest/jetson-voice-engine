// Phase 3b empirical spike — session lifecycle + capacity refusal on the v0.8.0 runtime.
//
// Validates the begin/append/end lifecycle of `AsrStreamingSessionRuntime` (patch
// v080-0002) on a REAL v0.8.0 qwen3-asr thinker engine. Three scenarios:
//
//   Scenario 1 — Session pair (clean teardown):
//       begin → appendChunk×2 → end  →  begin → appendChunk×2 → end.
//     Both sessions consume identical token slices from a fresh begin. endAsrSession resets
//     the KV cache (resetForSessionEnd), so the second session must start from the same
//     initial state and produce BIT-IDENTICAL last-step logits (max-abs diff < 1e-2).
//     Also asserts peekKvCacheLength returns to 0 after each end.
//
//   Scenario 2 — Capacity refusal (KV-overflow → refuse, NOT silent advance):
//       begin with maxPositions=M; push chunks of length L. The append whose cumulative
//       positions would exceed M MUST be refused (appendChunk returns false) and MUST NOT
//       advance the committed KV length. A successful append never pushes kv past M.
//       After end + begin, a fresh chunk must succeed (state recovered).
//
//   Scenario 3 — Append-after-end refusal (lifecycle guard):
//       appendChunk after endAsrSession (status kIdle) must return false (no active session)
//       and leave no state. Mirrors the old m2 per-chunk-overflow refusal intent: the runtime
//       refuses out-of-lifecycle work rather than silently advancing.
//
// Uses ordinary TEXT token IDs (no mel) to isolate the prefill/commit lifecycle.

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
std::vector<int32_t> makeTextTokens(int n, uint32_t seed)
{
    std::vector<int32_t> ids(n);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int32_t> dis(1, 100000);
    for (auto& t : ids)
        t = dis(gen);
    return ids;
}

double maxAbsDiff(std::vector<float> const& a, std::vector<float> const& b)
{
    if (a.size() != b.size() || a.empty())
        return 1e30;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return m;
}

bool scenario1_sessionPair(rt::LLMInferenceRuntime& rt, cudaStream_t stream)
{
    printf("\n--- Scenario 1: session pair (begin/append×2/end ×2) ---\n");
    constexpr int32_t kMaxPos = 512;
    auto c0 = makeTextTokens(20, 7);
    auto c1 = makeTextTokens(18, 9);

    auto runOne = [&](char const* label) -> std::vector<float> {
        rt::AsrStreamingSessionRuntime sess(rt, 0);
        if (!sess.beginAsrSession({}, kMaxPos, 0, stream))
        {
            fprintf(stderr, "  [%s] begin FAILED\n", label);
            return {};
        }
        if (!sess.appendChunk(c0, false, stream) || !sess.appendChunk(c1, true, stream))
        {
            fprintf(stderr, "  [%s] append FAILED\n", label);
            return {};
        }
        CK(cudaStreamSynchronize(stream));
        int kv = sess.peekKvCacheLength(stream);
        auto logits = sess.getLogitsForTesting(stream);
        sess.endAsrSession(stream);
        CK(cudaStreamSynchronize(stream));
        int kvEnd = sess.peekKvCacheLength(stream);
        printf("  [%s] kv_after_appends=%d (exp %d)  kv_after_end=%d (exp 0)\n", label, kv,
            (int) (c0.size() + c1.size()), kvEnd);
        if (kvEnd != 0)
        {
            fprintf(stderr, "  [%s] end did not zero KV\n", label);
            return {};
        }
        return logits;
    };

    auto la = runOne("session_a");
    auto lb = runOne("session_b");
    if (la.empty() || lb.empty())
    {
        printf("  Scenario 1: FAIL (empty logits)\n");
        return false;
    }
    double d = maxAbsDiff(la, lb);
    bool pass = d < 1e-2;
    printf("  Scenario 1: max-abs diff between sessions = %.6e => %s\n", d, pass ? "PASS" : "FAIL");
    return pass;
}

bool scenario2_kvOverflow(rt::LLMInferenceRuntime& rt, cudaStream_t stream)
{
    printf("\n--- Scenario 2: capacity refusal (cumulative > maxPositions) ---\n");
    constexpr int32_t kMaxPos = 250; // small cap to force overflow
    constexpr int chunkLen = 100;    // 100,100 ok (=200); 3rd (=300) must refuse
    auto chunk = makeTextTokens(chunkLen, 3);

    rt::AsrStreamingSessionRuntime sess(rt, 0);
    if (!sess.beginAsrSession({}, kMaxPos, 0, stream))
    {
        printf("  FAIL: begin\n");
        return false;
    }

    int successCount = 0;
    bool refused = false;
    for (int i = 0; i < 5; ++i)
    {
        int kvBefore = sess.peekKvCacheLength(stream);
        bool ok = sess.appendChunk(chunk, false, stream);
        CK(cudaStreamSynchronize(stream));
        int kvAfter = sess.peekKvCacheLength(stream);
        printf("  call %d: kv_before=%d ok=%d kv_after=%d status=%d\n", i, kvBefore, ok ? 1 : 0, kvAfter,
            (int) sess.status());
        if (ok)
        {
            ++successCount;
            if (kvAfter > kMaxPos)
            {
                printf("  FAIL: successful append pushed kv=%d past cap=%d\n", kvAfter, kMaxPos);
                return false;
            }
        }
        else
        {
            refused = true;
            if (kvAfter != kvBefore)
            {
                printf("  FAIL: refused append advanced state (before=%d after=%d)\n", kvBefore, kvAfter);
                return false;
            }
            break;
        }
    }
    if (!refused)
    {
        printf("  FAIL: never refused — capacity guard inactive\n");
        return false;
    }
    printf("  successes=%d before refusal (state preserved on refusal)\n", successCount);

    // Recover: end + begin + fresh append must succeed.
    sess.endAsrSession(stream);
    CK(cudaStreamSynchronize(stream));
    if (sess.peekKvCacheLength(stream) != 0)
    {
        printf("  FAIL: end did not zero KV\n");
        return false;
    }
    rt::AsrStreamingSessionRuntime sess2(rt, 0);
    if (!sess2.beginAsrSession({}, kMaxPos, 0, stream) || !sess2.appendChunk(chunk, true, stream))
    {
        printf("  FAIL: fresh session after overflow could not append\n");
        return false;
    }
    CK(cudaStreamSynchronize(stream));
    printf("  fresh-session append kv=%d (exp %d)\n", sess2.peekKvCacheLength(stream), chunkLen);
    sess2.endAsrSession(stream);
    printf("  Scenario 2: PASS\n");
    return true;
}

bool scenario3_appendAfterEnd(rt::LLMInferenceRuntime& rt, cudaStream_t stream)
{
    printf("\n--- Scenario 3: append-after-end refusal (lifecycle guard) ---\n");
    auto chunk = makeTextTokens(10, 5);
    rt::AsrStreamingSessionRuntime sess(rt, 0);
    if (!sess.beginAsrSession({}, 256, 0, stream) || !sess.appendChunk(chunk, true, stream))
    {
        printf("  FAIL: setup\n");
        return false;
    }
    CK(cudaStreamSynchronize(stream));
    sess.endAsrSession(stream);
    CK(cudaStreamSynchronize(stream));
    // Now status is kIdle; appendChunk must refuse.
    bool ok = sess.appendChunk(chunk, true, stream);
    printf("  append after end: ok=%d status=%d (expect ok=0)\n", ok ? 1 : 0, (int) sess.status());
    if (ok)
    {
        printf("  FAIL: append accepted after end\n");
        return false;
    }
    printf("  Scenario 3: PASS\n");
    return true;
}
} // namespace

int main(int argc, char** argv)
{
    std::string engineDir = "/home/harvest/tensorrt-edgellm-workspace/Qwen3-ASR-0.6B/engines-v080/llm";
    std::string multimodalEngineDir = "";
    if (argc >= 2)
        engineDir = argv[1];

    // Register the edgellm TRT plugins (AttentionPlugin etc.) before any engine deserialize.
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
    printf("runtime constructed against %s (maxSessionBatchSize=%d)\n", engineDir.c_str(),
        runtime->maxSessionBatchSize());

    bool s1 = scenario1_sessionPair(*runtime, stream);
    bool s2 = scenario2_kvOverflow(*runtime, stream);
    bool s3 = scenario3_appendAfterEnd(*runtime, stream);

    printf("\n=== M2 SESSION LIFECYCLE SUMMARY ===\n");
    printf("  Scenario 1 (session pair / clean teardown):  %s\n", s1 ? "PASS" : "FAIL");
    printf("  Scenario 2 (KV-overflow refusal):            %s\n", s2 ? "PASS" : "FAIL");
    printf("  Scenario 3 (append-after-end refusal):       %s\n", s3 ? "PASS" : "FAIL");
    bool all = s1 && s2 && s3;
    printf("=== %s ===\n", all ? "M2 ACCEPTANCE: PASS" : "M2 ACCEPTANCE: FAIL");
    cudaStreamDestroy(stream);
    return all ? 0 : 1;
}
