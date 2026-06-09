# v080-0009 — qwen3-tts CustomVoice ASR-roundtrip: CORRECTED verdict (en + zh PASS)

Date: 2026-06-10
Host: orin-nx (Orin NX 16GB, JetPack 6, CUDA 12.6.68, TRT 10.3.0.30) — verified `Linux aarch64` + `orinnx`.
Branch: feat/edgellm-v080-migration

## TL;DR

**v0.8.0 qwen3-tts CustomVoice IS intelligible for BOTH English and Chinese**, ASR-roundtrip
verified, byte-exact, deterministic. This **reverses** the prior verdict in
`v080-tts-customvoice-roundtrip-acceptance.md` ("NOT correct for either en or zh").

No source fix was required. The original migration hypothesis — that a v0.7.1 user→assistant
role coercion was missed — is **refuted**: the coercion is already present in the v0.8.0
`qwen3_tts_inference.cpp` (box commit `5ad0cbb`, lines 392-398) and is also a no-op in
isolation. The prior "garbled" results were **test-harness / config artifacts**, not a real
TTS defect.

## What the prior pass got wrong (root cause of the false negative)

The prior pass concluded en+zh garbled. Re-running on the **same canonical binary**
(`build/examples/omni/qwen3_tts_inference`, md5 `d6f76540e14d549580b4603a3fd3b679`) and the
**same engines** (`engines-v080-tts/{talker,code_predictor,code2wav}`, code2wav md5
`566c389eea9337195e6b58a846a4ed39`), the cause was isolated to the **input config**, not the model:

| input config | frames | EN ASR-roundtrip (paraformer_trt) | verdict |
|---|---|---|---|
| `apply_chat_template=true`, `add_generation_prompt=true` (canonical) | 56–57 | `theweatherisreallynicetodayletusgoforawalktogether` | **CORRECT** |
| `add_generation_prompt=false` | 51 | `...goforaquote` (tail drift) | degraded |
| `apply_chat_template=false` | 164 (runaway) | `我不` | **GARBLED** |

The prior garbled 34-frame / RMS≈0.0095 EN golden corresponds to a non-canonical prefill path
(no/incorrect chat-template application), NOT the production code path. With the canonical config
the audio is correct.

A secondary contributor: the prior pass leaned on a marginal **Qwen3-ASR-0.6B** harness whose own
known-good sanity transcript was already wrong (`今天亲戚真不错`, 亲戚≠天气). This pass uses the
production **paraformer_trt** ASR (`seeed-voice` :8621 `/asr`), validated on known-good audio first
(matcha TTS `今天天气真不错` → paraformer `嗯天天气气真不错`, ≈ correct; EN matcha
`The weather is really nice today` → `theweatherisreallynicetoday`, exact).

## THE ROUNDTRIP GATE (raw, this run)

Greedy decode (`talker_temperature=0.0, talker_top_k=1`), `apply_chat_template=true`,
`add_generation_prompt=true`, `enable_thinking=false`, speaker `vivian`.

### English — input: "The weather is really nice today, let us go for a walk together."
- projectToTalkerInput: seqLen=23, N=15, outputSeqLen=26, langId=2050, prefixRows=9, speakerId=3065.
- **57 audio frames**, dur **4.56s**, sr 24000, **RMS 0.09173** (non-silent).
- Deterministic: two independent runs → byte-identical audio md5 `d4381b851482de6f14bfdb0f5e0335a5`.
- **paraformer_trt roundtrip: `theweatherisreallynicetodayletusgoforawalktogether`**
  → "The weather is really nice today let us go for a walk together" → **byte-exact. CORRECT.**
- (Stochastic defaults temp=0.9/top_k=50 also correct: 56 frames, same transcript.)

### Chinese — input: "今天天气真不错"
- langId=2055, 9-row prefix (prefixRows=9), seqLen=12, N=4, outputSeqLen=15.
- **24 audio frames** (NOT the prior 2-frame early-EOS), dur **1.92s**, **RMS 0.08449**.
- audio md5 `b6faabe77aa7ae19eadfe905121f25eb`.
- **paraformer_trt roundtrip: `今天天气真不错`** → **byte-exact. CORRECT.**

Golden refs (this run): `docs/audio-evidence/`
- `v080-customvoice-en-57frames-2026-06-10.wav`  md5 `d4381b851482de6f14bfdb0f5e0335a5`
- `v080-customvoice-zh-24frames-2026-06-10.wav`   md5 `b6faabe77aa7ae19eadfe905121f25eb`

## Isolation evidence (why no source change)

1. **Role coercion already present.** `examples/omni/qwen3_tts_inference.cpp:392-398` already
   coerces a single user-role message → assistant before the talker prefill. Hypothesis refuted.

2. **Split-projection is a no-op.** Rebuilt the box binary with the split-projection reverted to
   a single multi-row `invokeTalkerMLP(thinkerEmbed, ...)` call (binary md5 `05c356366ba28d996681df8750f4fbc7`).
   EN output was **byte-identical** (`d4381b85...`) to the split-projection binary and transcribed
   byte-exact. Split vs no-split → identical audio. The prior pass's "split-projection has no effect"
   was correct; this pass confirms it AND confirms the binary is intelligible regardless. (Box source
   left in canonical committed state with split-projection in place; binary rebuilt back to
   `d6f76540...`.)

## Verdict

**TTS accuracy gate GREEN.** v0.8.0 qwen3-tts CustomVoice produces INTELLIGIBLE, roundtrip-verified,
byte-exact speech for English AND Chinese with the canonical inference config
(`apply_chat_template=true`, `add_generation_prompt=true`). The prior "garbled both languages"
finding was a config/harness artifact and is superseded by this document. The `--wrap` toolchain
fix (v080-0008) remains necessary; no further source change is required for accuracy.

**Caller-facing config note:** drivers/integrations MUST send `apply_chat_template=true` and
`add_generation_prompt=true` to qwen3_tts_inference. Disabling chat-template produces runaway,
garbled audio — that path was the source of the earlier false negative.
