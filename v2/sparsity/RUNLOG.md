# RUNLOG — Trained Dynamic Depth (A1) experiments

Reproducibility log: exact commands, hyperparameters, and results, so nothing lives only in
terminal scrollback. Machine: remote `yuke-team` (RTX PRO 6000 Blackwell). Model: Fast-dLLM v2 7B.
All eval numbers gsm8k unless noted, `--thresholds 0.9`, n=100 (CI ≈ ±9%). Date: 2026-08-08.

> NOTE ON PROVENANCE: file edits are made in the local Mac repo and synced to the server to run.
> Checkpoints (`.pt`) live on the SERVER (`/tmp/...`) and are NOT in git — preserve the important
> ones manually (see "Checkpoint registry" below).

---

## Scripts
| script | role |
|---|---|
| `depth_router_derisk.py` | teacher-forced capacity probe + controls (static, random); defines `DepthRouter`, `RoutedLayer`, `wrap_layers`, `set_router_state` (gate modes `soft`/`ste`/`hard`) |
| `depth_router_train_a1.py` | **A1 generation-aware training** (on-policy self-distillation to frozen original; dense prefill+commit; STE gate; budget curriculum; optional `--sched_slope`, `--dataset`) |
| `depth_router_gen_eval.py` | installs router into real decode loop; reports orig / lora_dense / routed accuracy + `keep_ss` + early/late allocation; `--ablate_gate_commit`, `--force_commit_keep`, `--dataset` |
| `nfe_probe_stage1.py` (edited) | `_math_pred` fixed: truncate at hallucinated "Problem:", take FIRST `\boxed` (was grabbing last, hallucinated box) |

---

## Headline model
`a1_0.65` — the reference model. Free (average) budget, ~21% small-step layer skip, gsm8k accuracy
preserved, exhibits emergent convergence allocation.
```
# TRAIN (produces /tmp/a1_0.65.pt on the server)
uv run v2/train_scripts/depth_router_train_a1.py --steps 800 --budget_warmup_frac 0.4 \
    --target_keep 0.65 --sched_slope 0.0 --save /tmp/a1_0.65.pt
# EVAL
uv run v2/train_scripts/depth_router_gen_eval.py --load /tmp/a1_0.65.pt --n_examples 100 --thresholds 0.9
# -> orig 68% | lora_dense 65% | routed 70% @ keep_ss 0.79 | [early 0.69 / late 0.85]
```

---

## Results

### 1. Naive fine-tuning collapses (motivates A1)
Teacher-forced training (snapshot objective) → generation collapse (numeric mode-collapse), dose-dependent:
gsm8k 62.5% (0 steps) → 7.5% (100 steps) → 0% (400 steps). Cause: teacher-forcing/exposure-bias gap.

### 2. A1 fixes it + Pareto frontier (gsm8k, n=100, orig=68%)
Recipe: `--steps 800 --budget_warmup_frac 0.4 --sched_slope 0.0 --target_keep <TK>`.
| target_keep | keep_ss (infer) | routed_acc |
|---|---|---|
| 0.90 | 0.96 | 70% |
| 0.85 | 0.93 | 72% |
| 0.80 | 0.89 | 71% |
| 0.75 | 0.86 | 69% |
| 0.70 | 0.83 | 61% (noise outlier) |
| 0.65 | 0.79 | 70% |
Read: accuracy PRESERVED within noise down to keep_ss ~0.79 (~20% small-step layer-compute). The +2/+3
over orig is noise, NOT improvement. The 0.70→61% is a non-monotonic outlier (noise-dominated).

### 3. Recoverable-KV validated (dense commit is load-bearing)
```
uv run v2/train_scripts/depth_router_gen_eval.py --load /tmp/a1_0.65.pt --n_examples 100 \
    --thresholds 0.9 --ablate_gate_commit --force_commit_keep 0.80
```
- router-gated commit (~8% skip): 73% (no hurt — UNDERPOWERED, only 8% skipped).
- forced ~20% commit skip (COMMIT-keep 0.80): **70% → 15%** (−55%), while same ~20% on small steps is free.
- => commit is load-bearing; small-step skips are recoverable. Diffusion-specific, AR-impossible.
- (Optional cliff figure: `--force_commit_keep 0.90 0.85` maps where 8%-fine → 20%-craters.)

### 4. Emergent convergence allocation (three-way control, gsm8k n=100, orig=68%)
`gen_eval` routed line reports `[early / late]` keep (mostly-masked vs mostly-decoded states).
| model | budget | acc | keep_ss | early / late |
|---|---|---|---|---|
| a1_0.65 | avg (free) | 70% | 0.79 | 0.69 / 0.85 |
| a1_flat | per-state uniform (slope 0) | 53% | 0.77 | 0.72 / 0.79 |
| a1_sched | per-state + schedule (slope 0.4) | 65% | 0.80 | 0.68 / 0.87 |
Commands: `a1_flat` = `--target_keep 0.65 --sched_slope 0.0`; `a1_sched` = `--target_keep 0.65 --sched_slope 0.4`
(both `--steps 800 --budget_warmup_frac 0.4`). Read: (a) convergence allocation helps (uniform 53% ≪
allocated 65-70%); (b) the free avg-budget router self-discovers depth-near-commit — explicit schedule NOT
needed (emergent ≥ engineered). Verdict on step-conditioned schedule (#2): drop it.

### 5. MATH (transfer: gsm8k-trained model, MATH-eval) — after `_math_pred` fix
```
uv run v2/train_scripts/depth_router_gen_eval.py --load /tmp/a1_0.65.pt --dataset math \
    --max_new_tokens 512 --n_examples 100 --thresholds 0.9
```
-> orig 42% | lora_dense 36% | routed 32% @ keep_ss 0.85 | [early 0.74 / late 0.91]
Read: TRANSFER (trained gsm8k, eval MATH). ~10pt drop = LoRA domain-shift (42→36) + routing (36→32),
borderline noise. Emergent allocation TRANSFERS. NOTE: before the parser fix this read a bogus 10%
(model over-generates hallucinated "Problem:"s; parser was grabbing the last, hallucinated \boxed).

---

## Checkpoint registry (all on server `/tmp/`, NOT in git — preserve manually)
| file | config | note |
|---|---|---|
| a1_0.65.pt | avg budget, target 0.65, steps 800, warmup 0.4 | **HEADLINE / reference model** — preserve! |
| a1_0.70/0.75/0.80/0.85/0.90.pt | frontier sweep, same recipe, varying target | keep for frontier repro or delete (configs above) |
| a1_flat.pt | per-state budget, slope 0, target 0.65 | control (three-way) |
| a1_sched.pt | per-state budget, slope 0.4, target 0.65 | control (three-way) |
| depthr*.pt | early/naive/derisk runs | junk — deletable |

---

## Known caveats / open items
- n=100 (CI ±9%): "preserved" claims are within-noise; rigor needs n=300-500 + ≥2 seeds.
- gsm8k-trained (prototype convenience). CORRECT protocol = train ONCE on BROAD prompts, eval all
  tasks with the same checkpoint (do NOT fine-tune per eval). Not yet done.
- Speedup NOT measured: gating SIMULATES skipping (no compute saved). Wall-clock needs a real
  compute-skip kernel; predicted ~1.1x standalone / ~2x stacked IF skips are whole-layer (unknown).
- Prototype training: bs=1, 1 block/example, on-policy rollout + per-state grad forward (slow).
- MATH still over-generates (NFE ~190, wasteful, harmless to scoring post-fix).
