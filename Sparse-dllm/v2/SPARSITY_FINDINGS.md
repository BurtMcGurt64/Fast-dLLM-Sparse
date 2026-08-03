# Sparsity in Fast-dLLM v2 — Token-level & KV-level Decoding (batched, b>1)

**Meeting brief.** Study of two orthogonal sparsity axes in the Fast-dLLM v2 diffusion small step,
extended to **batch size > 1**. Model: Fast-dLLM v2 7B (Qwen2.5-7B backbone, 32-token blocks).
Hardware: RTX PRO 6000 (94 GB). Baseline: the *optimized* dense Fast-dLLM v2 (a hard baseline).
Speedups = decode throughput (tok/s) vs dense; accuracy = LongBench token-F1 / GSM8K exact-match.

---

## TL;DR (the one slide)

Two levers speed up the diffusion small step, on **different hardware bottlenecks**:

| axis | what it cuts | speedup | accuracy | verdict |
|---|---|---|---|---|
| **Token policy** (compute) | query rows recomputed per step | **~1.3×** (grows to ~1.5× w/ batch) | **free** — ≈ dense on 4 tasks | **the win** |
| **KV streaming** (memory) | prefix KV bytes read | +0.1–0.4× on top | **−0.01 to −0.06 F1**, un-tunable | speed-for-accuracy knob |
| **Composed** | both | up to **~1.8×** @ high batch | inherits the KV cost | near-multiplicative, not free |

**Headline:** the **token policy is a free ~1.3× at b>1** (the deliverable). Composing streaming-KV
stacks to ~1.8× at high batch but costs content coverage that **selection cannot recover** (content ≈
position) — so it's a characterized tradeoff, not a free multiplier.

---

## 1. What was built

The **small step** (diffusion denoising within a block) is ~94% of decode forwards and the only forward
these methods touch (prefill + commit stay dense). Two independent sparsity axes:

- **Token policy (compute axis)** — Xinze's method, **newly extended to b>1**. A token freezes once its
  hidden state stops changing across denoising steps (staleness `l2` < thr) and is not recomputed for
  `freeze` steps; only *active* positions run through `forward_block_positions`. Frozen tokens keep
  cached KV so others still attend. **Batching** via a *union of active positions* (skip a position only
  if frozen in all sequences).
- **KV streaming (memory axis)** — small steps read only the first `n_sink` + last `n_recency` prefix
  **blocks** (a `kv_window` truncation; absolute rope preserved → retrieval-preserving StreamingLLM).
- **Metric:** `l2` (relative ‖Δh‖/‖h‖) ≫ `cosine` — cosine over-locks (freezes tokens whose direction
  is stable but magnitude still shifts). All results use `l2`.
- **Arms:** `dense` · `token` (compute only) · `kv` (memory only) · `compose` (both).

---

## 2. Finding 1 — Token policy is a **free ~1.3×** (the result)

**Accuracy is untouched across four LongBench tasks** (bs=64, gen=256, n=200):

| task | dense F1 | token F1 | Δ |
|---|---|---|---|
| hotpotqa | 0.244 | 0.241 | −0.003 |
| gov_report | 0.215 | 0.214 | −0.001 |
| 2wikimqa | 0.145 | 0.144 | −0.001 |
| multifieldqa_en | 0.408 | 0.410 | +0.002 |

All within ±0.01 (F1 noise floor). **The compute axis costs no accuracy, on every task tested.**

**Speed: ~1.2–1.4×, growing with batch** (small step becomes compute-bound as batch amortizes weight
loads). Clean example (gov_report, monotonic): 1.17× (bs8) → 1.24× → 1.33× → **1.38× (bs64)**.
Threshold Pareto (GSM8K): `l2 thr=0.01` is the near-lossless knee at ~1.2×; `thr=0.02` → ~1.4× for a
few points. skip% holds ~50% (batch-union is mild — a shared prefix makes sequences freeze correlated
positions).

> *Measurement note:* early single-run timings were noisy/non-monotonic due to CUDA cold-start on the
> first config; the harness now warms up and medians `--repeat` runs. Accuracy numbers were always clean.

---

## 3. Finding 2 — KV streaming buys speed but **costs content coverage**

Streaming adds speed (more at high batch) but **loses accuracy on any task that needs the middle of the
context**, and the loss does **not** tune away with window size.

**Standalone KV cost** (bs=64, gen=256):
| task | dense F1 | kv F1 | Δ |
|---|---|---|---|
| hotpotqa (multi-hop) | 0.228 | 0.215 | −0.013 |
| gov_report (summarization) | 0.186 | 0.124 | **−0.062** |

**Window Pareto — no near-lossless setting** (hotpotqa, bs=64, n=200):
| n_recency | blocks kept (of 128) | F1 |
|---|---|---|
| dense | 128 | 0.244 |
| 4 | 6 | 0.208 |
| 8 | 10 | 0.210 |
| 16 | 18 | 0.201 |
| 32 | 34 | 0.220 |

F1 stays ~0.20–0.22 regardless of window — it never climbs back to dense, even keeping 27% of context.

**Why (the mechanism):** the cost is from **having a KV budget at all** (dropping most blocks), not from
*how* you pick within it. Multi-hop evidence is diffuse; summaries need full coverage — no position
window captures either. This reconfirms the prior project's core result: **content selection ≈ position**
(commit-score top-K does not beat sink+recency). *You cannot select your way out of needing the context.*
(Note: the prior "recency pinning helps gov_report" was about **coherence** — repetition/looping — not
**coverage**. Streaming keeps output fluent, not complete.)

---

## 4. Finding 3 — Composition: the roofline crossover

The axes are genuinely orthogonal (compute vs memory), so at high batch — FFN compute-bound **and** KV
read memory-bound — they compound **nearly multiplicatively** (hotpotqa, gen=256):

| bs | token | kv | compose | token×kv (if mult.) |
|---|---|---|---|---|
| 16 | 1.16× | 1.18× | 1.23× | 1.37× |
| 64 | 1.26× | 1.22× | 1.40× | 1.54× |
| 128 | 1.31× | 1.45× | **1.81×** | 1.90× |

- Composition gain grows with batch: +0.05× → +0.14× → **+0.36×**; at bs=128, composed (1.81×) ≈ the
  product (1.90×).
- **Bonus:** dense throughput *drops* at bs=128 (KV cache hits the memory wall) while sparse arms keep
  scaling → **KV sparsity extends the servable batch range** past where dense thrashes.
- **Clean cost split:** token F1 ≈ dense at every batch (compute free); *all* accuracy cost is the KV
  axis; compose inherits only the KV cost.

---

## 5. Honest bottom line & recommendation

**Stable conclusion (invariant across every run): token axis = free; KV axis = coverage-costly and
un-selectable.**

- **Present the token policy as the contribution:** Xinze's method extended to b>1, **free ~1.3×**
  (→ ~1.5× at high batch) on a hard baseline, validated accuracy-neutral on 4 tasks. Clean and defensible.
- **Present KV composition as a characterized extension**, not a headline: it stacks on throughput (up to
  ~1.8×) and extends batch-scaling, but the memory axis costs content coverage that selection can't
  recover — which independently reconfirms the prior content ≈ position finding.
- **Open decision for the meeting:** is the higher-batch composition regime (≥1.8×, needs chunked prefill
  to push past bs~100 at 4k) worth pursuing, or is the free token-policy 1.3× the paper?

---

## 6. Limitations & caveats (state these explicitly)

- **Scope:** decode throughput vs *optimized* dense v2, gen=256, 4k ctx. The ~1.3–1.8× depends on gen
  length (short gen → prefill dominates → lower; gen=128 gave ~1.1–1.2×) and batch. Report the operating
  point.
- **Memory ceiling:** bs>~100 at 4k OOMs on the **dense prefill FFN activation** (~18.5 GB), not the KV
  cache. Higher batch needs **chunked prefill** (unimplemented).
- **Metric:** gov_report is officially ROUGE-L; driver uses token-F1 as a proxy — trust the *relative*
  drop, not the absolute.
- **Noise:** F1 ~±0.02–0.03 at n=200; single-run speed is timing-noisy (use `--repeat`, warmup is
  automatic). One bs=32 row once showed compose < both single axes — physically impossible = noise.
- **skip% is an unreliable cross-arm proxy** (block_size denominator vs ≤8-position subblock steps) —
  trust speed + F1.
- **MAGE-style content arming not re-tested here**; standalone content-vs-position is settled in the
  prior project (`baseline_compare.py`).

---

## 7. Reproduce

```bash
# Headline: token-only free ~1.3x across tasks x batches (accuracy-neutral)
python run_policy_eval.py --task longbench \
  --lb_tasks hotpotqa,gov_report,2wikimqa,multifieldqa_en \
  --ctx 4096 --n 200 --gen 256 --uniform --metric l2 --thr 0.01 \
  --kv_mode none --batch_sizes 8,16,32,64 --repeat 3

# 4-way decomposition (dense / kv / token / compose) at one batch
python run_policy_eval.py --task longbench --lb_task hotpotqa --ctx 4096 --n 200 --gen 256 \
  --uniform --metric l2 --thr 0.01 --kv_mode fourway --n_sink 2 --n_recency 8 --batch_sizes 64 --repeat 3

# KV window Pareto
for rec in 4 8 16 32; do
  python run_policy_eval.py --task longbench --lb_task hotpotqa --ctx 4096 --n 200 --gen 256 \
    --uniform --metric l2 --thr 0.01 --kv_mode compose --n_sink 2 --n_recency $rec --batch_sizes 64 --repeat 3
done
```

Code: `run_policy_eval.py` (driver) · `generation_functions.py` (`batch_sample_token_policy_flashattention`,
b>1 union, `kv_window`) · `model/modeling.py` (`forward_block_positions`, `kv_window` read-truncation).
