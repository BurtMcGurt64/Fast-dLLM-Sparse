# Fast-dLLM v2 Sparsity — Project State & Decision Memo
---

## TL;DR

The original direction — **Top-K block KV sparsity to speed up Fast-dLLM v2** — is **exhausted as a
speedup contribution.** It produces **no headline on any metric**:

| Metric | Result | Why |
|---|---|---|
| Latency (e2e) | **~1.3×** (1.36× best) | weight floor + prefill dilution |
| **Throughput** (tok/s) | **~1.06×** (measured 2026-07-19) | thin KV at real contexts + pins force frac=41% |
| Memory / batch capacity | **0** | current top-K reads selectively but does **not evict** — full KV resident |

Separately, **every adjacent research direction is already published** (see the prior-art map). Two
things survive as genuinely ours: a **coherence mechanism** (needs one more control experiment) and
the **honest characterization** itself. One systems angle is **unmeasured**: recovering the small
step's wasted memory bandwidth (a kernel/operator problem, independent of sparsity).

**This is a decision point for the PI, not an open optimization.**

---

## 1. What we set out to do

Use commit-step attention scores to load only the top-K most-attended prefix blocks during the
denoising small steps, reducing KV bytes read → faster decoding. The motivating "3–5×" was always a
KV-read kernel measured in isolation, never end-to-end.

## 2. What we now know (measured)

### 2a. The speedup is capped by physics, and we can derive it
Every decode forward reads **weights (~13 GB, unprunable) + KV (∝ batch×context, prunable)**. Since
it's memory-bound, `time ∝ bytes`, so:

```
speedup = (W + KV) / (W + frac·KV)
```

This one formula **reproduces every measured row.** At B=32, short context (~768–2560 tok), KV is
only **10–25% of the bytes** — so even pruning KV to zero gives **1.1–1.35×**. The weights dominate;
top-K can't touch them.

### 2b. The context/prefill vise (why no regime escapes ~1.3×)
- **Short context** → KV is a thin slice of each forward → pruning barely helps → ~1.1–1.3×.
- **Long context** → KV finally matters per-forward, but **prefill** now dominates e2e → ~1.3–1.36×.

Squeezed to ~1.3× from both directions, measured three independent ways (8k-context sweep,
short-prompt long-gen sweep, and the byte arithmetic above).

### 2c. Latency vs throughput vs decode-only — reconciling the numbers
| Scope / regime | Speedup | Note |
|---|---|---|
| Per small step, B=64 (microbench, frac 0.2) | 2.4× | isolated; does NOT survive to e2e |
| Decode-only, 8k | 1.59× | excludes prefill |
| **End-to-end latency, best** | **1.36×** | includes prefill |
| **End-to-end throughput (2026-07-19)** | **1.06×** | flat across B=8→64; even 0.90× at B=8 |

The 2.4×/1.6× numbers are components measured in favorable isolation. The honest serving numbers are
the last two rows.

### 2d. Throughput probe detail (2026-07-19, ctx 512 / gen 512)
```
batch |  D_tok/s | S_tok/s | frac | tput x
  8   |    724   |   655   | 41%  | 0.90x
  16  |    805   |   871   | 41%  | 1.08x
  32  |    948   |  1010   | 41%  | 1.06x
  48  |    975   |  1044   | 41%  | 1.07x
  64  |    995   |  1052   | 41%  | 1.06x
```
- Throughput ratio is **flat ~1.06×**, does **not** rise with batch (the hoped-for mechanism failed).
- **Peak memory identical** dense vs sparse → no eviction → **no capacity/batch gain.**
- Dense throughput itself **saturates** (724→995 tok/s = only 1.37× for 8× batch) — that saturation
  is a real inefficiency (see §6).

### 2e. Sparsity does NOT reduce the number of steps (NFE)
Hypothesis: pruning distracting context → sharper confidence → fewer denoising steps. **Refuted.**
At matched accuracy, TPF (tokens/forward) ratio = **1.01–1.02×** — sparse NFE ≈ dense NFE. The lever
that moves NFE is the confidence threshold, and it moves dense and sparse identically.

## 3. Escape levers tried and closed

| Lever | Outcome | Evidence |
|---|---|---|
| Longer generation | asymptotes to decode ceiling, still ~1.3× | gen-length sweep |
| Higher batch | throughput saturates ~B=8–16; ratio flat 1.06× | §2d |
| Self-speculation (diffusion draft + AR verify) | **published** | DFlash (ICML), DEER, Nemotron, PSD |
| Sparsity → NFE reduction | **null** (TPF 1.01×) | §2e |
| Trained NFE reduction (self-distillation/consistency) | **published** | SDTT, T3D (block diffusion), CDLM, CD4LM |
| Throughput framing | **1.06×**, no capacity gain | §2d |

## 4. Prior-art map (the field is saturated at every axis)

**Algorithm axis**
- **Focus-dLLM** (2602.02159) — confidence-guided context focusing; a *more advanced* version of our
  top-K block selection (adds query-side sparsity). Per-step compute only, does **not** touch NFE.
- **DFlash** (2602.06036, ICML), **DEER** (2512.15176), **PSD** (2605.15609),
  **Self-Spec Diffusion** (2510.04147) — diffusion drafter + AR verifier, lossless ~6×.
- **SDTT** (2410.21035), **T3D** (block + full diffusion), **CDLM** (2511.19269),
  **CD4LM** (2601.02236), **Trajectory Self-Distillation** (2602.12262) — trained few-step / NFE.
- **SparseD** (2509.24014), **Sparse-dLLM** (2508.02558) — KV sparsity / dynamic eviction.

**Serving/systems axis**
- **HERALD** (2606.21633) — high-throughput block-diffusion serving via CPU–GPU cooperative KV retrieval.
- **DiLaServe** (2606.29094) — SLO-attainment serving; its **Confidence-Threshold Controller** already
  does the "NFE knob as SLO dial" idea.
- **eLLM** — adaptive fine-grained KV serving.
- Production: Gemini Diffusion, Mercury 2, LLaDA 2.x (800–1000+ tok/s).

## 5. What survives as genuinely ours

1. **The recency/sink-pinning coherence fix.** Plain top-K degenerates long-form generation (**4/4
   prompts, rep 0.81, looping text**); pinning restores it to dense quality (**rep 0.12 ≈ dense 0.13**).
   Focus-dLLM/HERALD/DiLaServe all list coherence as a non-target.
   - **Caveat / open control:** in the run above, pinned kept 66% vs plain's 20% — **frac not matched.**
     Must show recency-positioned blocks preserve coherence at the *same* budget as content-only
     selection (needs a budget-capped-pinning loader change) to attribute the effect to *position*,
     not *quantity*.
2. **The honest batched, end-to-end, weight-floor-aware characterization.** The whole sparsity subfield
   reports B=1 / long-context / decode-only numbers; nobody reports realistic batched serving against
   the weight floor. We have the tooling and the byte model.

## 6. The one unmeasured systems angle

The **dense** small step runs at **~20% of peak memory bandwidth** at high batch (bandwidth falls
64%→20% from B=1→64), and dense throughput saturates (1.37× for 8× batch). That wasted bandwidth is a
real, attackable inefficiency — **independent of sparsity** — about how the model is *served*, not what
it computes. Recovering it (kernel fusion / CUDA graphs / better batched layout) is a
**Mercury-flavored operator problem** and the only speedup angle not yet ruled out.
- **Go/no-go:** one `ncu` profiling session — is the loss launch/dispatch overhead (fixable) or a
  fundamental memory pattern (dead)?
- **Risk:** it's a *different project* (kernel/systems, not sparsity); **HERALD may already cover it**
  (unread in detail).

## 7. Where this leaves us — options

1. **Kernel/operator throughput (Direction A).** Recover the 20% bandwidth on the dense small step.
   Only live speedup angle; Mercury-flavored; needs `ncu` go/no-go + HERALD overlap check + PI scope.
2. **Consolidate what's real (non-speedup).** Characterization + coherence fix → a workshop paper or
   solid technical report. Honest, finishable, *not* a headline speedup.
3. **Pivot the problem.** The topic (diffusion-LLM efficiency) is saturated at every axis. A genuinely
   open speedup problem likely lives outside it — needs the PI's field/agenda knowledge.

## 8. Recommendation

The technical question ("does sparsity give a speedup on this checkpoint?") is **answered: no, on any
metric.** Continuing to probe it is negative-EV. This is a **PI decision** — pivot the problem, change
the checkpoint, greenlight the kernel angle with scope buy-in, or accept the characterization+coherence
outcome. Bring them this memo and the one-line ask:

> *"Sparsity on Fast-dLLM v2 gives ~1.3× latency, ~1.06× throughput, 0 capacity gain (all measured),
> and every adjacent direction — algorithm and serving — is published. The one unmeasured angle is
> recovering the small step's 20%-of-peak bandwidth (a kernel/operator problem, Mercury-flavored,
> independent of sparsity). I have the characterization, a coherence mechanism, and honest benchmarks
> in hand. Where do you want me to point next?"*

---

## Appendix — scripts (all runnable on this box)
| Script | Purpose |
|---|---|
| `throughput_probe.py` | tok/s + peak-mem + OOM ceiling, dense vs sparse across batch (§2d) |
| `sparsity_nfe_probe.py` | TPF/NFE at matched accuracy — the sparsity→NFE null (§2e) |
| `coherence_eval.py` | dense vs plain vs pinned top-K, repetition/distinct-4 (§5) |
| `batched_long_gen.py` | short-prompt long-gen speedup + accuracy modes |
| `runtime_breakdown.py` | prefill/commit/small time split → per-eval ceiling |
| `memory_vs_compute.py` | empirical roofline, per-step speedup vs batch, the 20%-bandwidth finding |
| `measure_kv_bytes.py` | resident weight vs KV byte split |

## Appendix — key constants
- Weights ~13.05 GB (46.5% of bytes at B=32/8k); KV 15.03 GB (53.5%).
- KV ≈ 56 KB / token / sequence (28 layers × 4 KV heads × 128 dim × 2 (k,v) × 2 B).
- Decode ≈ 17 forwards / 32-token block (16 small + 1 commit); small steps ≈ 94% of decode forwards.
- Per-step sparsity speedup (microbench): 1.9× (B16) → 2.15× (B32) → 2.4× (B64) — does not survive to e2e.
