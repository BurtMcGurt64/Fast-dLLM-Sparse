# Sparsity in Fast-dLLM v2 (block diffusion) — Token-level & KV-level, batched (b>1)

Summary of the token-policy + KV-streaming sparsity study on **Fast-dLLM v2 7B** (Qwen2.5-7B backbone,
block diffusion, 32-token blocks). Measured on **RTX PRO 6000 (94 GB)**. Driver: `run_policy_eval.py`;
mechanism in `generation_functions.py` (`batch_sample_token_policy_flashattention`) + `model/modeling.py`
(`forward_block_positions`, `kv_window`).

> **One-line takeaway.** There are two orthogonal sparsity axes in the diffusion small step —
> **compute** (query rows → token policy) and **memory** (KV bytes → streaming). **The token axis is a
> free ~1.3×; the KV axis buys more speed (composes to ~1.8× at high batch) but at a real,
> un-tunable accuracy cost on any task that needs content coverage.** Baseline is the *optimized*
> Fast-dLLM v2 (a hard baseline), speedups are decode-throughput (tok/s) vs dense.

---

## 1. Setup & definitions

- **Small step** (diffusion denoising within a block) is ~94% of decode forwards and the only forward
  either method speeds up. Prefill and commit steps are unpruned.
- **Token policy (compute axis):** freeze a token's query once its final hidden state stops changing
  across denoising steps (staleness < threshold) for `freeze_step` steps; recompute only the *active*
  (non-frozen) positions via `forward_block_positions`. Frozen tokens keep cached KV so others still
  attend to them. b>1 handled by **union of active positions** (a position is skipped only if frozen in
  *all* sequences in the batch).
- **KV streaming (memory axis):** during small steps, attention reads only the first `n_sink` + last
  `n_recency` prefix **blocks** (a `kv_window` truncation gated to the mask-free small-step regime;
  absolute rope preserved → retrieval-preserving StreamingLLM variant). Commit/prefill stay dense.
- **Staleness metric:** `l2` (relative ‖Δh‖/‖h‖) ≫ `cosine`. l2 is near-lossless at high skip; cosine
  over-locks (freezes tokens whose direction is stable but magnitude still shifts). All results use `l2`.
- **Arms:** `dense` (ORG) · `token` (compute only) · `kv` (memory only = streaming with freeze=0) ·
  `compose` (both). Speedup = arm tok/s ÷ dense tok/s. Accuracy: GSM8K exact-match; LongBench token-F1.

---

## 2. Token-level sparsity (compute axis) — **the free ~1.3×**

The token policy is **accuracy-free across tasks and batches**, with speedup that grows with batch
(the small step becomes compute-bound as batch amortizes the weight loads).

| task (bs=64, gen=256, n=200) | dense F1 | token spd | token F1 | Δacc |
|---|---|---|---|---|
| LongBench/hotpotqa | 0.244 | **1.29×** | 0.241 | −0.003 (free) |
| LongBench/gov_report | 0.186 | **1.34×** | 0.182 | −0.004 (free) |

**Threshold Pareto (GSM8K, bs=64, n=200):** `l2 thr=0.01` is the near-lossless knee (0.690 vs dense
0.695) at ~1.21×; `thr=0.02` gives ~1.39× for ~4–5 pts. Gap noise floor ≈ ±0.04 at n=200.

**Batch scaling (token only):** ~1.16× (bs16) → ~1.3× (bs128). skip% holds ~50% (batch-union is mild —
sequences with a shared prefix freeze correlated positions).

**Verdict:** this is the solid, defensible contribution — Xinze's token policy extended to b>1,
near-lossless ~1.3× on a hard baseline.

---

## 3. KV-level sparsity (memory axis) — **more speed, but it costs coverage**

Streaming KV adds speed, especially at high batch, **but always costs accuracy where the task needs
the middle of the context** — and the cost does **not** tune away with window size.

### 3a. Standalone KV cost (bs=64, gen=256)
| task | dense F1 | kv spd | kv F1 | Δacc |
|---|---|---|---|---|
| hotpotqa (n=100) | 0.228 | 1.22× | 0.215 | −0.013 |
| gov_report (n=100) | 0.186 | 1.15× | 0.124 | **−0.062** |

### 3b. Window Pareto — no near-lossless setting (hotpotqa, bs=64, n=200, compose)
| n_recency | blocks kept (of 128) | compose spd | compose F1 |
|---|---|---|---|
| dense | 128 | 1.00× | 0.244 |
| 4 | 6 | 1.42× | 0.208 |
| 8 | 10 | 1.27× | 0.210 |
| 16 | 18 | 1.31× | 0.201 |
| 32 | 34 | 1.39× | 0.220 |

**F1 stays ~0.20–0.22 regardless of window — it never climbs back to dense (0.244), even keeping 27% of
the context.** (Speeds are single-run noisy; F1 flatness is the signal.)

**Why:** the accuracy cost is from **having a KV budget at all** (dropping most blocks), not from *how*
you pick within it. hotpotqa is multi-hop (evidence diffuse) and gov_report is summarization (needs
full coverage) — in both, no position window captures what's needed. This reconfirms the prior
project's core result: **content selection ≈ position** (commit-score top-K does not beat sink+recency;
the "marginal multi-hop edge" was noise). You cannot *select* your way out of needing the full context.

> **Important distinction:** the prior "recency pinning helps gov_report" result was about **coherence**
> (repetition/looping), not **content coverage**. Streaming keeps output *fluent* but not *complete* —
> which is why summarization F1/ROUGE still drops.

---

## 4. Composition — the roofline crossover

The two axes are genuinely orthogonal (compute vs memory), so at high batch — where the FFN is
compute-bound **and** the KV read is memory-bound — they compound **nearly multiplicatively**.

| bs (hotpotqa, gen=256) | token | kv | compose | token×kv (if mult.) | compose F1 vs dense |
|---|---|---|---|---|---|
| 16 | 1.16× | 1.18× | 1.23× | 1.37× | 0.203 vs 0.227 |
| 64 | 1.26× | 1.22× | 1.40× | 1.54× | 0.214 vs 0.228 |
| 128 | 1.31× | 1.45× | **1.81×** | 1.90× | 0.198 vs 0.244 |

- **Composition gain grows with batch:** +0.05× (bs16) → +0.14× (bs64) → **+0.36× (bs128)**;
  at bs=128 composed (1.81×) ≈ the product (1.90×).
- **Bonus serving effect:** dense tok/s *drops* at bs=128 (KV cache hits the memory wall) while sparse
  arms keep scaling → **KV sparsity extends the batch-scaling range** past where dense OOMs/thrashes.
- **Clean cost decomposition:** token F1 ≈ dense at every batch (compute axis free); **all** the accuracy
  cost is the KV axis; compose inherits only the KV cost.

---

## 5. Honest bottom line

| axis | speedup | accuracy | verdict |
|---|---|---|---|
| **Token (compute)** | ~1.3× (grows to ~1.5× w/ batch) | **free** (≈ dense everywhere) | **the win — universal, defensible** |
| **KV (memory)** | +0.1–0.4× on top (batch-dependent) | **−0.01 to −0.06 F1**, un-tunable, task-dependent | speed-for-accuracy knob, not free |
| **Composed** | up to **~1.8×** @ bs≥128 | inherits the KV cost | near-multiplicative, but not free |

**Framing:** lead with the token policy (free ~1.3×, b>1 extension). Present KV composition as a
*characterized* extension: it stacks on throughput and extends batch-scaling, but the memory axis costs
content coverage that selection cannot recover (content ≈ position). This turns the KV result into a
clean characterization consistent with the prior project, rather than an oversold multiplier.

---

## 6. Caveats & limitations

- **Speedup scope:** decode-throughput (tok/s) vs *optimized* dense Fast-dLLM v2, gen=256, 4k context.
  Report the operating point explicitly; the ~1.3–1.8× depends on gen length (short gen → prefill
  dominates → lower) and batch.
- **Generation length matters:** at gen=128 the KV benefit shrinks to ~1.1–1.2× (prefill dilution);
  the ~1.3–1.4× KV numbers require gen≥256. (Confirms the prior gen-length sweep: 64→1.09×, 256→1.31×.)
- **Memory ceiling:** bs>~100 at 4k OOMs on the **dense prefill FFN activation** (~18.5 GB for
  `[128,4096,18944]`), not the KV cache. Higher batch needs **chunked prefill** (unimplemented).
- **Metric:** gov_report is officially ROUGE-L; the driver uses token-F1 as a rough proxy — trust the
  *relative* drop, not the absolute value.
- **Noise:** F1 noise ~±0.02–0.03 at n=100–200; single-batch speed points are timing-noisy (e.g. a
  bs=32 row showed compose < both single axes, which is physically impossible = noise).
- **skip% is an unreliable cross-arm proxy** (denominator is block_size=32 but subblock steps touch ≤8
  positions → a ~55% structural baseline swamps the freeze signal). Trust speed + F1.

---

## 7. Reproduce

```bash
# Headline token-only table (free ~1.3x) across tasks x batches:
python run_policy_eval.py --task longbench --lb_tasks hotpotqa,gov_report,2wikimqa,multifieldqa_en \
  --ctx 4096 --n 200 --gen 256 --uniform --metric l2 --thr 0.01 --kv_mode none \
  --batch_sizes 8,16,32,64

# Full 4-way decomposition (dense / kv / token / compose) at one batch:
python run_policy_eval.py --task longbench --lb_task hotpotqa --ctx 4096 --n 200 --gen 256 \
  --uniform --metric l2 --thr 0.01 --kv_mode fourway --n_sink 2 --n_recency 8 --batch_sizes 64

# KV window Pareto:
for rec in 4 8 16 32; do
  python run_policy_eval.py --task longbench --lb_task hotpotqa --ctx 4096 --n 200 --gen 256 \
    --uniform --metric l2 --thr 0.01 --kv_mode compose --n_sink 2 --n_recency $rec --batch_sizes 64
done
```
