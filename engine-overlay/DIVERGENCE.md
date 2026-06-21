# voxedge-engine DIVERGENCE.md

```text
upstream:       github.com/NVIDIA/TensorRT-Edge-LLM (Apache 2.0)
UPSTREAM_PIN:   364769036fc83351d9d0aac4cc064a8e56a83178   (= tag v0.7.1)
fork_branch:    v071/customvoice-product  (extracted HEAD 893ba2a, 90 commits ahead)
extraction:     2026-05-31 (initial overlay extraction; build-verify DEFERRED → Jetson host)
```

## (a)/(b) 判定准则

1. 上游自己代码里的正确性 bug → **(a)** 可上游化,提 PR,合并后退役 patch + bump pin。
2. 我们产品/模型/路线图专属特性或性能路径 → **(b)** 自留。
3. 拿不准 → 默认 **(b) + pr_candidate: true**,下次 review 重评。

> 整体 diff 全集: **A=40, M=27, D=0** (高度 additive)。
> addon/ 收 40 个 A 文件 (纯新增,ISO);patches/ 8 个主题 patch 覆盖全部 27 个 M 文件。
> 混合主题 (含 (a) 与 (b),需 hunk 拆) 已逐条标 `mixed:`。

---

## orin-tegra-build-compat
category:     a
pr_candidate: true
upstream_pr:  (none yet — file after splitting private kernel/plugin-registration hunks out)
retirement:   merged → drop 0001 patch, bump UPSTREAM_PIN past merge commit
rationale:    Jetson 原生 build: auto-detect Tegra (`/etc/nv_tegra_release`) → 默认 EMBEDDED_TARGET=jetson-orin; aarch64 保留用户 CMAKE_CUDA_ARCHITECTURES; CuTe DSL GEMM sm_87 重启用 + CUDA driver lib link + `--wrap=_cudaLaunchKernelEx`
scope:        patches=0001-orin-tegra-build-compat.patch (CMakeLists.txt, cmake/CuteDsl.cmake, cpp/CMakeLists.txt)
mixed:        YES — 含 (a) build-compat 与 (b) 私有 kernel/plugin 注册 (cpp/CMakeLists.txt 里 W8A16 / MOSS / CP kernel+plugin 注册行)。上游化前需 hunk 拆: build-compat → PR; 私有注册留 overlay。
validation:   hardware (orin-nx / orin-nano native build)
last_replay:  — (initial extraction)

## weight-streaming-budget
category:     a
pr_candidate: true
upstream_pr:  (none yet)
retirement:   merged → drop 0002 patch, bump UPSTREAM_PIN
rationale:    Orin 15GB 统一内存: engine build set kWEIGHT_STREAMING flag (builderUtils) + `EDGELLM_WEIGHT_STREAMING_BUDGET` env runtime budget (parseWeightStreamingBudgetEnv / applyWeightStreamingBudget),避免一次性 pin 全权重
scope:        patches=0002-weight-streaming-budget.patch (builderUtils.cpp, llmRuntimeUtils.{cpp,h}, llmEngineRunner.{cpp,h}, eagleDraftEngineRunner.cpp)
mixed:        YES — llmEngineRunner.* 里的 shared-engine 构造器 (ASR slot-pool D1) 是 (b) 并发逻辑,与 weight-streaming (a) 混在同文件; llmRuntimeUtils.cpp 还含 (b) generateMultimodalIndices audioIndexBase hunk (ASR session)。上游化前 hunk 拆: budget → (a) PR; shared-engine ctor + audioIndexBase → (b) overlay。
validation:   hardware (Orin NX engine build + runtime)
last_replay:  —

## asr-streaming-session
category:     b
pr_candidate: false
upstream_pr:  —
rationale:    边缘流式 ASR worker 核心: chunked-prefill / MRope streaming-ASR session init (audioRunner), beginAsrSession/endAsrSession 生命周期 + slot 状态 (llmInferenceSpecDecodeRuntime)。DIVERGENCE 标这是 fork 里最易冲突的 CORE 编辑,rebase 重点守护
scope:        patches=0003-asr-streaming-session.patch (audioRunner.{cpp,h}, multimodalRunner.h, llmInferenceSpecDecodeRuntime.{cpp,h}); addon=examples/llm/spike_m1/m2/m35/m36.cpp (+ examples/llm/CMakeLists.txt 注册见 0008)
note:         llmRuntimeUtils.cpp 的 audioIndexBase hunk 物理上落在 0002 patch (同文件与 weight-streaming 混),逻辑归属本主题
validation:   hardware ASR roundtrip + perf gate; spike gtest
last_replay:  —

## tts-slotpool-concurrency
category:     b
pr_candidate: false
upstream_pr:  —
rationale:    TTS N=2 并发护城河: 通用 SlotPool<TSlot> + per-slot 状态 ownership + statefulCode2WavRunner; per-request CUDA stream; 绑产品 worker 模型,非上游 request 路径
scope:        addon=cpp/runtime/slotPool.h, cpp/multimodal/statefulCode2WavRunner.{cpp,h}, examples/omni/qwen3_tts_streaming_worker.cpp; patches=0004-tts-slotpool-concurrency.patch (qwen3OmniTTSRuntime.{cpp,h}, code2WavRunner.cpp QWEN3_TTS_CODE2WAV_PROFILE hook)
note:         slotPool.h 单独评估为 pr_candidate (通用模板),但当前随并发主题自留
validation:   hardware N=2 burst (30/30 0 CUDA err, audio MD5 byte-identical)
last_replay:  —

## customvoice-language-conditioning
category:     b
pr_candidate: false
upstream_pr:  —
rationale:    CustomVoice checkpoint 专属: language_id codec 条件 + 9-row prefix layout (vs 上游 8-row); CuTe DSL disabled 时 ELLM_CHECK; per-request `language` 字段; 导出 codec_language_id map。上游无此 checkpoint
scope:        patches=0005-customvoice-language-conditioning.patch (talkerMLPKernels.{cu,h}, examples/omni/qwen3_tts_inference.cpp per-request language, experimental/llm_loader/export_all_cli.py codec_language_id)
validation:   hardware (radxa SenseVoice ASR byte-exact transcribe)
last_replay:  —

## server-sse-disconnect-and-openai-api  (mixed)
category:     a (SSE-disconnect + generic server subset) / b (tool-calling subset)
pr_candidate: true
upstream_pr:  SSE-disconnect 部分 PENDING — **DO NOT auto-submit (用户指派他人提交)**
retirement:   SSE-disconnect merged → 从本 patch 删该 hunk (commit 0898b5f) + bump UPSTREAM_PIN 越过合并点
rationale:    本 patch 物理合并了两个逻辑主题,因为它们在 api_server.py/engine.py 的 commit 历史上交错 (13e2c71, ca36892 落在 0898b5f 之前),在 read-only fork 上无法 rebase 拆成独立可 base-apply 的 patch:
              - SSE/client-disconnect 取消 (commit 0898b5f, (a) PR-pending): /v1/chat/completions client 断开 (barge-in) → worker 继续生成 → 下请求 race 共享 engine ctx → Myelin "already loaded binary graph" 崩溃。修: api_server 加 _watch_disconnect 轮询 is_disconnected → channel.cancel(); engine.py generate_stream finally 协作取消
              - OpenAI 兼容 server 增强: cache metrics (ca36892, (a)), MTP startup assertion + EAGLE engine-dir fallback (87a958c, (a)), warmup KV reuse (805b22a/d5697fe/abce9f1, (a)); tool_call 早发 delta (c952043, (b) 偏产品)
mixed:        YES — 含 (a) 与 (b)。上游化前需按 commit/hunk 拆: SSE-fix 单独 PR (禁止自动提交); 通用 server robustness 子集另 PR; tool-calling 早发留 overlay
scope:        patches=0006-server-sse-disconnect-and-openai-api.patch (experimental/server/api_server.py, experimental/server/engine.py); 配套 addon=experimental/server/tests/*, experimental/server/launch.sh; 文档见 0007
validation:   server e2e + voice-agent barge-in repro (memory: trt_edge_llm_sse_disconnect_pr); experimental/server/tests/* unit
last_replay:  — (SSE-fix commit 0898b5f, 2026-05-22)

## server-openai-api-docs
category:     a
pr_candidate: true
upstream_pr:  (随 server robustness 子集一起)
retirement:   随 0006 的 (a) 子集上游
rationale:    experimental-server.md server 端点文档 (cleanly 可分离,单独文件)
scope:        patches=0007-server-openai-api-docs.patch (docs/source/user_guide/examples/experimental-server.md)
validation:   doc only
last_replay:  —

## build-misc-example-registration
category:     b (worker/spike 注册) / 琐碎 (.gitignore)
pr_candidate: false
upstream_pr:  —
rationale:    example/worker/spike target CMake 注册 (随对应 addon worker/spike) + .gitignore macOS 噪音行 (.DS_Store / ._*)
scope:        patches=0008-build-misc-example-registration.patch (examples/omni/CMakeLists.txt qwen3_tts_streaming_worker target, examples/llm/CMakeLists.txt spike_m1/m2/m35/m36 targets, .gitignore)
note:         .gitignore macOS 行单评 (a)-cand 或可丢弃; worker/spike 注册随 addon 自留
validation:   hardware (worker 二进制存在 + 命名)
last_replay:  —

---

## (b) 自留 addon (无 patch 触点 / 仅 CMake 注册)

> 以下 addon 引擎侧组件为产品模型/量化/并发足迹,其 CORE 触点仅落在 0001 (cpp/CMakeLists.txt 注册)
> 或 0008 (example CMake)。逐主题单列 DIVERGENCE 便于 rebase 与 HF artifact 追踪。

## w8a16-quantization
category:     b
pr_candidate: false
upstream_pr:  —
rationale:    Qwen3-TTS talker AWQ W8A16 on Orin: 量化 kernel + TRT plugin + ONNX 量化脚本 (engine 435MB, −49.5% vs FP16)
scope:        addon=cpp/kernels/w8A16LinearKernels/w8A16Linear.{cu,h}, cpp/plugins/w8A16LinearPlugin/w8A16LinearPlugin.{cpp,h}, scripts/quantize_onnx_matmul_w8a16{,_awq}.py, scripts/create_qwen3_vocoder50_wrapper.py; CORE 触点=0001 patch 的 cpp/CMakeLists.txt W8A16 注册 hunk
validation:   hardware (radxa ASR semantic match), addon/docs/known-issues/w8a16-talker-handoff.md
last_replay:  —

## qwen3-tts-cp-kernel
category:     b
pr_candidate: false
upstream_pr:  —
rationale:    Qwen3-TTS code-predictor (CP) CUDA kernel; 绑产品 TTS 路径
scope:        addon=cpp/kernels/qwen3TtsCpKernels/qwen3TtsCpKernels.{cu,h}; CORE 触点=0001 patch cpp/CMakeLists.txt CP 注册 hunk
validation:   hardware (TTS CP 输出)
last_replay:  —

## moss-tts-nano-port   ← 首个 overlay 迁移范例
category:     b
pr_candidate: false
upstream_pr:  —
rationale:    MOSS-TTS-Nano 产品模型移植 (linear-attention KV kernel + stateful runtime + JSON-line worker)。上游无此模型。TTFA 157ms, 19× faster than ORT。最干净一类: 几乎全 ISO 新文件, 仅 1 处 CMake 注册算 CORE
scope:        addon=cpp/kernels/kvCacheUtilKernels/mossLinearKvKernels.{cu,h}, cpp/runtime/mossTtsNanoRuntime.{cpp,h}, cpp/workers/moss_tts_nano_worker.cpp, cpp/workers/build_moss_worker.sh, unittests/mossLinearKvKernelsTests.cu, unittests/mossTtsNanoSmokeMain.cpp; CORE 触点=0001 patch cpp/CMakeLists.txt MOSS 注册 hunk only
validation:   unittests/mossLinearKvKernelsTests (5/5 PASS), mossTtsNanoSmokeMain; hardware 3 zh prompts CER=0, N=2 byte-identical
last_replay:  2026-06-10 v0.8.0 (patch v080-0011): ported verbatim (0 source adaptation, GLOB auto-register + 1 CMake smoke-main-exclude hunk); edgellmCore+worker built; mossLinearKvKernelsTests 5/5 PASS; 3 zh prompts roundtrip-correct on orin-nx (paraformer_trt). See docs/plans/v080-0011-moss-tts-nano-roundtrip-acceptance.md

## worker-cancel-protocol
category:     b
pr_candidate: false
upstream_pr:  —
rationale:    worker 侧 {"type":"cancel","id":...} 协作取消 (cancelMap + atomic flag)。绑产品 worker IPC, 与上游可上游的 server 侧 SSE-fix 区分
scope:        包含在 tts-slotpool worker addon=examples/omni/qwen3_tts_streaming_worker.cpp cancel hunk (无独立 patch, 整文件为 addon)
validation:   stress 200+ early-break, audio MD5 byte-identical
last_replay:  — (commit 776cd03)

## qwen3-awq-deploy-container  (docs/recipes)
category:     b
pr_candidate: false
upstream_pr:  —
rationale:    Qwen3 AWQ Orin 部署/导出/打包配方脚本 (export/build/package/serve)。非 runtime 镜像输入,overlay 文档/配方区
scope:        addon=docs/deploy-container/* (9), docs/known-issues/{qwen35-orin-nx-oom,w8a16-talker-handoff}.md (2)
note:         迁移时建议归 addon/scripts/qwen3/ 与 overlay 文档区 (现按原 fork 相对路径 docs/deploy-container/ 保存)
validation:   recipe scripts (run on Orin/WSL build host)
last_replay:  —

## server-test-scaffolding
category:     a (candidate)
pr_candidate: true
upstream_pr:  (随 server 通用化一起)
rationale:    experimental/server test 包 + system-prompt cache 分支测试 + launch.sh。随 SSE/server 通用子集上游候选; tool_call parser 测试绑 (b) tool-calling 特性
scope:        addon=experimental/server/tests/{__init__.py,test_cache_messages_branch.py,test_tool_call_stream_parser.py}, experimental/server/launch.sh
validation:   unit
last_replay:  —
```

## known-issue: stale example demo breaks `make all`
category:    b (carried)
discovered:  2026-05-31 (orin-nx real compile verify)
issue:       The overlay changes `AudioChunkCallback` to a 3-arg signature
             (modified cpp/runtime/qwen3OmniTTSRuntime.h). The upstream example
             `examples/llm/llm_inference.cpp:700` still uses a 1-arg lambda →
             `make all` fails on that demo target. **Shippable targets
             (NvInfer_edgellm_plugin + qwen3_tts_streaming_worker) build clean
             (exit 0)** — only the unrelated demo breaks.
fix:         add a patch updating examples/llm/llm_inference.cpp's callback
             lambda to the 3-arg form, OR exclude that example from the build
             (CMake option). Needed only for a clean full-tree `make all` / CI.
status:      open (non-blocking — product compiles)
