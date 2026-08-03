# Future Directions — slide content (2026-07-20)

---

## Slide A — Why inference-only sparsity isn't the path to a headline

**The method is capped** (measured): ~1.3× shippable, ~1.6× decode-only, ~2× stacked-w/-INT8 ceiling.
The >2× lever is reducing forward passes (NFE) — a **training** contribution.

**The space is crowded** — ~20 papers in ~12–18 mo across the exact sub-angles:
- KV sparsity/selection: MAGE, LoSA, Sparse-dLLM, UNIQUE, dLLM-Cache, FlashBlock
- Serving: HERALD, dLLM-Serve, Sangam
- Speculative/verify: SimSD (7.5×), S2D2, DFlash
- Step-reduction/distillation: **dParallel (ICLR'26, 8.5×)**, T3D, SDTT (ICLR'25), CD4LM, d3LLM
- A survey already exists (Parallel Text Generation, 2508.08712)

**Strongest signal:** I independently re-derived two of these (my "unmask more per step" = dParallel's
certainty-forcing; my trajectory-distillation fix = T3D) *without knowing they existed*. When multiple
groups land the same method in months, the low-hanging fruit is gone.

---

## Slide B — Proposed pivot: efficiency for (discrete-diffusion) VLA

**Why:** my toolkit transfers directly, into a *younger, less dense* field with *real deployment demand*
(robots need ~30 Hz; diffusion action heads run many denoising steps → same NFE bottleneck I've studied).

**What transfers (1:1 with my summer work):**
- Block-diffusion decoding, KV/token sparsity, NFE reduction, rigorous efficiency measurement.
- Discrete-diffusion VLA (Fast-dVLA, 2603.25661) is my closest fit and **under-explored vs continuous
  flow policies** — where 1-step / 100× results already exist (One-Step Flow Policy, FRMD, FlowPolicy).

**Honest calibration (not greenfield):** the obvious transfers are *seeded* — BlockVLA & TBD-VLA (block
diffusion), VLA-Cache & SpecPrune-VLA (sparsity), a systematic efficiency survey (2510.17111). But
density is 1–3 papers/sub-angle (vs 5–10 in text-dLLM), the field is fragmented → open *combinations*,
and my block-diffusion + sparsity + NFE + measurement skill-stack in one person is unusual.

**The lesson I'm applying:** don't chase the obvious intersection (that's what gets scooped) — target a
specific unclaimed *combination* or a real deployment insight, and validate novelty in a day, not a month.

---

## Slide C — What I'm asking

1. Is a rigorous, honestly-measured ~2× publishable for our target venue — or does the bar require >2×?
2. If >2×: that's the NFE/training axis, largely published (dParallel/T3D). Adopt-and-differentiate,
   switch checkpoint, or redirect?
3. Is applying my efficiency + measurement toolkit to **discrete-diffusion VLA** worth scoping? I'm
   excited about it and it's where my skills are scarcer.

*(Note to self: confirm which "fast-vla" — Fast-dVLA is Song et al. (academic), NOT the NVIDIA/Fast-dLLM
team; NVIDIA's VLA line is GR00T N1.7. Don't misattribute in the meeting.)*
