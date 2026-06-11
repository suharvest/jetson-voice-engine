# 流式 ASR「定长 decode + 保准确率」机制原理

> 适用：Qwen3-ASR 在 TensorRT-Edge-LLM 上的流式逐字幕（partial）。
> 一句话：**用「assistant-prefix 续写 + chunk-and-confirm 回滚」让每个 partial 只解码「最近不稳定的尾巴 + 新词」，把 per-hop decode 从 O(整段转写) 压成常数，同时靠「回滚未确认尾巴」保住准确率。**
> 源头：v0.7.x worker（`asr-worker-build-verify/qwen3_asr_worker.cpp`）；v0.8.0 `OVS_ASR_STREAM_PREFIX` 路径是它的忠实移植。

---

## 1. 问题：流式 partial 的两难

流式 ASR 每收到一小段音频（一个 hop，≈0.5–1s）就要吐一次「目前听到的全文」（partial）。两种朴素做法都不行：

| 朴素做法 | 每 hop 成本 | 准确率 |
|---|---|---|
| **A. 每 hop 重解全文**（部署 #13 的 first-cut） | decode ∝ 当前转写长度 → 整段 **O(N²)** | ✅ 正确、可修订（每次重新推导，能纠错） |
| **B. commit 后只续写**（无回滚） | decode 只解新词 → **定长** | ❌ 错。部分音频上模型会吐**早句号 `。`+EOS 甚至幻觉**，一旦 commit 进前缀就**永久锁死、无法修订** |

实测部署路径（做法 A）：cumulative 1→15.4s，decode **69→727ms（涨 10.5×）**，per-hop total 98→784ms。短指令（≤3s）无感，中长语音明显卡。

做法 B 的失败样本（无回滚移植版，真实跑出来）：
```
zh_long_02  @2.8s 模型自信幻觉 → '《中国共产党历史》。' + EOS
            这个带 EOS 的错前缀毒死所有后续 hop（续写看到"已完成"的句子 → 吐空）
            final 永远卡死在 '《中国共产党历史》'，CER 0.975
```

**核心矛盾**：要 decode 定长就得复用/commit 前次解码；但在部分音频上 commit 会锁死改不了的错。看似不可兼得。

---

## 2. 核心洞察：修订是「局部」的

随着音频变长，转写里**只有最近的尾巴会变**，靠前的词早已稳定。

```
1.0s:  适当使用博客，可以使学生变得。          ← 句尾"变得。"是猜的（早句号）
3.0s:  适当使用博客可以使学生变得更善于分析…   ← "变得"后的句号被改掉、续上"更善于…"
                ^^^^^^^^^^ 这段稳定了        ^^^^^^^^^ 只有尾巴在变
```

所以**不需要重解全文来修订——只需重解「不稳定的尾巴」**。这就是打破矛盾的关键：把 decode 限制在尾巴上，老词当作已确认、直接复用。

---

## 3. 机制详解

维护一个累积转写 `rawDecoded`。每个 hop：

```
computePrefix(rawDecoded):                      # 算"已确认前缀"
    if chunkId < unfixedChunkNum: return ""     # warmup：前 N hop 不信任何前缀，全重解
    tokens = tokenizer.encode(rawDecoded)
    end = len(tokens) - unfixedTokenNum          # 砍掉最后 unfixedTokenNum 个"未确认"token
    prefix = tokenizer.decode(tokens[:end])
    while prefix 末尾是半个多字节字符(U+FFFD):    # UTF-8 守卫：别在汉字中间切
        end -= 1; prefix = tokenizer.decode(tokens[:end])
    return prefix

每 hop:
    prefix = computePrefix(rawDecoded)
    # 把已确认前缀当 assistant 消息喂回，模型"续写"而非"重启"
    request = {
        system:    context,
        user:      <全量累积音频 mel>,
        assistant: kAssistantGenPrompt + "language X<asr_text>" + prefix,
    }, apply_chat_template=true, add_generation_prompt=false,
       max_generate_length = maxDecodeTokensPerHop   # 封顶
    generated = runtime.decode(request)               # 只生成"续写部分"
    rawDecoded = prefix + generated                   # 新全文 = 确认前缀 + 重解的尾巴+新词
    emit partial = rawDecoded
```

关键参数（v0.7.x 默认）：

| 参数 | 默认 | 作用 |
|---|---|---|
| `unfixedTokenNum` | 5 | **回滚窗口**：每 hop 把最后 5 个 token 当"未确认"丢弃重解 |
| `unfixedChunkNum` | 2 | **warmup**：前 2 个 hop 不用前缀（早期音频太不稳，确认什么都为时过早） |
| `maxDecodeTokensPerHop` | 64 | 每 hop 续写 token 上限（封顶，防失控） |

### 为什么 assistant-prefix 能让模型"续写"
chat template 渲染成 `system / user(audio) / assistant(已确认转写)`，且 `add_generation_prompt=false`——模型看到一个**未结束的 assistant 回合**，于是从前缀末尾**接着往下生成**，而不是从头重转写。运行时把前缀 **prefill**（并行、便宜），只对续写部分做 **autoregressive decode**（串行、贵）。v0.8.0 vanilla runtime 原生支持这个（不需额外 "Method B" 改动，已 probe 验证）。

---

## 4. 为什么 decode 定长

每 hop 实际 decode 的 token 数 ≈ `unfixedTokenNum(回滚的尾巴) + 本 hop 新音频对应的新词` ——**与整段转写长度无关，是个常数**。已确认前缀走 prefill，不进 decode 的串行循环。

实测（zh_long_03，rollback 路径 vs 部署 #13）：
```
            部署累积重解(#14)          rollback 续写
cum_sec | decode  total |  decode  total   gen字/hop
  1.0   |   69     98   |   68.7   97.3      2
  3.0   |  151    183   |   96.3  127.5      8
  6.0   |  290    326   |  110.3  146.5      7
 10.0   |  487    531   |  125.5  172.5      7
 15.4   |  727    784   |   71.1  129.3      0(EOS)
```
- **decode 定长 68–125ms**（部署涨到 727ms，10.5×）；每 hop 只生成 **2–8 字**。
- per-hop total：10s **快 3.1×**、15.4s **快 6.1×**，越长越省。
- prefill 仍随累积音频缓涨 28→58ms（这部分 rollback 不优化，是另一码事 = #12 的 <5% 地板）。

---

## 5. 为什么准确率不掉（回滚窗口的作用）

模型在**部分音频**上吐的早句号 `。`+EOS / 幻觉，**总是落在转写的最末尾**——也就是落在 `unfixedTokenNum` 的回滚窗口内。下一 hop：

1. `computePrefix` 把这段未确认尾巴**丢弃**；
2. 用**更完整的音频**重新解码这段尾巴；
3. 更多上下文让模型纠正：早句号被去掉、幻觉被真实内容替换。

只有**足够老**（滑出回滚窗口）的 token 才被永久确认——而那时它已经被多个 hop 的更长音频反复印证、稳定了。

**走查（真实跑出来的 zh_long_02）：**
```
partial[0]:  《中国共产党历史》。          ← hop0 幻觉（全在未确认窗口）
partial[1]:  桥下垂直净空十五米，干巷末余。  ← hop1：幻觉被回滚重解，纠回真实内容！
partial[2]:  桥下垂直净空十五米，该项目于二零一一年八月完工。
partial[3]:  …但直到二零一七年三月才开始通车。   ← 正确
```

实测 G-correct（rollback 版）：
```
zh_long_01  CER 0.0000   zh_long_02  CER 0.0250   zh_long_03  CER 0.0115
```
残差仅标点（`，`vs`。`、少个空格），是「确认后不再修订」窗口外的微小代价——v0.7.x 接受这个权衡。

**回滚窗口大小的取舍**：`unfixedTokenNum` 太小 → 不稳定 token 过早被确认 → 错；太大 → 每 hop 重解更多尾巴 → decode 变贵、且 partial 抖动更久才稳定。5 是 v0.7.x 在 0.5s hop 节奏下的经验值。hop 越粗（一次进的音频越多）通常要相应调大。

---

## 6. byte-exact final 精修（生产采用）

「确认前缀不再修订」带来的 CER 0.01–0.025 只影响 **partial 的最终累积值**。生产里：

- **partial**：照用便宜的回滚续写（flat decode，体感快）；
- **final**：在 `last:true` / `handleEnd` 时，丢掉累积的 `rawDecoded`，**做一次干净的 `runOneShotCore` 全量重解**（无前缀，部署已验证的黄金路径）→ final **逐字节 == 一次性结果**。

代价：结尾**多一次** O(audio) 解码（仅一次，正是部署路径本来每 hop 都做的事）。结果：**partial 便宜 + final byte-exact**，**严格优于现部署**（同样的 final 精度 + 中长语音 partial 快 3–6×）。

---

## 7. 与其他方案对比

| 方案 | per-hop decode | partial 准确率/可修订 | final 精度 |
|---|---|---|---|
| 部署 #13 累积重解 | O(audio) 增长 | ✅ 正确、可修订 | byte-exact |
| 无回滚续写（坏） | 定长 | ❌ 锁死早 EOS/幻觉 | CER 0.08–0.975 |
| **回滚续写（本机制）** | **定长** | ✅ 局部可修订 | CER 0.01–0.025 |
| **回滚续写 + byte-exact final** | **定长**（partial）+ 1 次全解（final） | ✅ 局部可修订 | **byte-exact** |
| #12 增量-KV peek | O(audio)（peek 解全文） | ✅ 正确 | byte-exact，但没压平 decode（优化的是 prefill 地板 <5%） |

**注**：#12（增量-KV）优化的是 **prefill** 侧（每 hop 重 prefill 全量音频→增量追加），实测只占 29–57ms 小地板，<5%；本机制优化的是 **decode** 侧（真正主导的 69–727ms）。两者正交，可叠加。

---

## 8. 边界与限制

- **prefill 不被优化**：每 hop 仍重新编码/prefill 全量累积音频（28→58ms 缓涨）。要连这块也压平需叠加 #12 的增量-KV 追加。
- **回滚窗口是经验参数**：`unfixedTokenNum`/`unfixedChunkNum` 需按 hop 节奏与语言/模型调；窗口外的早确认错误不可逆（靠 byte-exact final 兜底）。
- **依赖运行时支持 assistant-prefix 续写**（prefill 前缀 + 从中续生成）。v0.8.0 vanilla runtime 已验证支持。
- **短指令（≤3s）收益≈0**：hop 少，累积重解本来就快（都 <200ms）。收益随语音变长才显现。

---

## 9. 参考实现与产物

- v0.7.x 源：`asr-worker-build-verify/qwen3_asr_worker.cpp` —— `computePrefix`(L781-830)、`runStreamingHop`(L834-912)、per-hop caller(L1300-1340)、session 参数(L145-150)。
- v0.8.0 移植：worker `OVS_ASR_STREAM_PREFIX=1` 路径，patch `engine-overlay/patches/v080-0026-prefix-rollback-WORKS.patch`（393 行，worker-side、默认 OFF）。
- 验收：`docs/plans/v080-0026-prefix-rollback-streaming-WORKS.md`（G-correct + G-perf 实测）。
- 背景与踩坑（含"无回滚→根本性误判"的纠错）：`docs/plans/v080-0025-prefix-cap-decode-FINDING.md`（已标 SUPERSEDED）。
