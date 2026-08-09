# Concepts Primer — ML fundamentals, mapped to THIS project's code

Read this alongside the generic blogs/papers in your reading list. Each concept is defined plainly,
then pointed at exactly where it lives in **your** code, so the abstract idea becomes concrete.
Main files: `depth_router_train_a1.py` (A1 training), `depth_router_gen_eval.py` (generation test),
`depth_router_derisk.py` (defines the router + teacher-forced probe).

---

## Phase 0 — the 5 core ideas

### 1. Training / loss / gradient descent
**Plain:** you define a *loss* (a number that's low when the model behaves how you want), and the
optimizer nudges the model's weights to make it lower (gradient = "which direction lowers the loss").
**In your code:** `loss.backward()` (compute the nudge direction) + `opt.step()` (apply it) in
`depth_router_train_a1.py`'s training loop. Your loss = `kl + λ·(keep−target)²`.
**Why:** the whole project is "invent the *right* loss so minimizing it gives a fast-but-accurate model."

### 2. Knowledge distillation (teacher / student)
**Plain:** train a "student" model to copy the outputs of a "teacher" model, instead of copying
ground-truth labels. The teacher's output distribution ("soft targets") is a richer signal than a hard label.
**In your code:** `set_lora_enabled(model, False)` = the **teacher** (original full model). `set_lora_enabled(model, True)` + routers on = the **student** (depth-reduced). `compute_loss` runs both on the same input and makes the student match the teacher.
**Why:** A1 *is* distillation. The student (skipping layers) learns to reproduce the full model's predictions.

### 3. KL divergence
**Plain:** a number measuring how different two probability distributions are (0 = identical).
**In your code:** `F.kl_div(log_softmax(student_logits), softmax(teacher_logits))` in `compute_loss`.
It's the "student ≈ teacher" part of the loss.
**Why:** when this is ~0, the depth-reduced student produces the same next-token distribution as the full model.

### 4. Teacher forcing & exposure bias  ← *this is why the naive version collapsed*
**Plain:** "teacher forcing" = during training you always feed the *correct* previous tokens.
"Exposure bias" = at generation time the model feeds *its own* (imperfect) tokens, a distribution it
never trained on, so errors compound. A model can look great in training and fall apart generating.
**In your code / history:** the failed naive fine-tune trained on masked *snapshots* with the clean
answer in the input (the `[noisy|clean]` doubled batch) → the model learned to copy → gsm8k 62.5% → 0%.
**Why:** this is the exact failure A1 fixes. Understanding this = understanding the whole project pivot.

### 5. On-policy vs off-policy  ← *the fix*
**Plain:** "off-policy" = train on some fixed external data. "On-policy" = train on data the model
*generates itself*, so training matches what happens at inference.
**In your code:** `rollout_block()` in `depth_router_train_a1.py` — the student *decodes a block itself*
(under `no_grad`) and you record the states it visits; `compute_loss` then trains on *those* states.
No external answer is fed → no copying → exposure bias fixed.
**Why:** this is the single idea that separates A1 (works) from the naive fine-tune (collapses). It's also
what dOPSD / self-future do — the technique family is shared; your *application* (depth) differs.

---

## Phase 1 — the components your method is built from

### 6. LoRA (Low-Rank Adaptation)
**Plain:** instead of updating all 7B weights (expensive, easy to break), freeze them and add tiny
trainable "adapter" matrices (A, B) to each layer. Cheap, and reversible.
**In your code:** `LoRALinear` (in `nfe_probe_stage2b.py`), injected by `inject_lora()`. `B` starts at 0
so at init the adapter does nothing (the model is unchanged). `set_lora_enabled` toggles it via `scale`.
**Why:** your student is "original model + LoRA + routers." Turning LoRA off recovers the teacher.

### 7. Mixture-of-Depths / layer skipping (a.k.a. "Skip-GPT in spirit")
**Plain:** not every token needs every layer. A small "router" decides, per token, whether to run a
layer or skip it (pass the input straight through). Fewer layers run = less compute.
**In your code:** `RoutedLayer.forward`: `out = input + g·(layer_out − input)`. `g=1` → full layer;
`g=0` → skip (output = input). `g` comes from `DepthRouter` (a tiny MLP). This is your core method.
**Why:** this is exactly what your postdoc meant by Skip-GPT. The novelty is doing it for block-diffusion.

### 8. Straight-through estimator / Gumbel-softmax
**Plain:** "skip or keep" is a hard yes/no (0 or 1), but you can't take a gradient through a hard
decision. Trick: use the hard 0/1 in the forward pass, but pretend it was a smooth sigmoid for the
gradient. That's the "straight-through estimator" (STE).
**In your code:** `DepthRouter.forward`, `mode="ste"`: `g = hard + (soft − soft.detach())`. The value
is `hard` (0/1); the gradient flows through `soft` (sigmoid). `mode="hard"` = inference (no gradient
needed); `mode="soft"` = the older Gumbel (random) version that caused the train/inference mismatch.
**Why:** it's how the discrete router becomes trainable. Skim-level understanding is enough.

---

## Phase 2 — the diffusion context

### 9. Masked/block diffusion + the 3 forward types
**Plain:** unlike GPT (one token at a time), a diffusion LLM fills in a block of masked tokens over
several "denoising steps," each step unmasking the tokens it's now confident about.
**In your code / `CLAUDE.md`:** three forward types — **prefill** (read the prompt, dense),
**small step** (denoise the current block; ~94% of forwards; *this is what you make cheaper*),
**commit** (finalize the block, write its KV to cache, dense).
**Why — the diffusion-specific bit:** because the block is revisited across small steps and finalized by
a *dense* commit, a layer skipped during a small step gets corrected before it's committed. That's the
**recoverable-KV / dense-commit** structure — the thing AR can't do (in GPT a skip is permanent). In
`depth_router_gen_eval.py` you enforce it: routers **bypassed on prefill + commit, active only on small steps**.

---

## The KV-drift falsification (what those earlier experiments were)
**Plain:** a "probe" is a small classifier that tries to predict something from a feature. Here: "can we
predict whether a layer is skippable from how much a token's Key/Value (or hidden state) *changed*
between denoising steps?" You compare its accuracy (AUC = a 0.5–1.0 score for a yes/no predictor) to a
baseline that just uses the hidden state.
**Result:** the hidden state already contains everything the drift feature does — drift added ~nothing.
This is the **"sufficient statistic"** idea: since Key/Value are computed *from* the hidden state, a
predictor given the hidden state can't be helped by also getting a function of it.
**Why it matters now:** it's why you should **not** feed KV-staleness/Δh to the router — it's redundant.
Source: `v2/sparsity/interventional_probe.py` + memory `kv-keydrift-router.md`.

---

## Putting it together: what A1 actually does (one paragraph)
For a prompt, the **student** (original model + LoRA + routers that skip some layers on small steps)
**decodes a block itself** and you record the partial states it passes through (**on-policy**, `rollout_block`, no gradient). Then for each recorded state you run the **teacher** (same model, LoRA off, all
layers = full depth, `no_grad`) and the **student** (routers on, gradient), and train the student to
**match the teacher's next-token distribution** (**KL / distillation**) on the still-masked positions,
plus a **budget** term pushing it to skip a target fraction of layers. Prefill and commit stay **dense**
(recoverable-KV). Because training uses the student's *own* generated states and targets the *full-depth*
model (not a copied answer), it avoids the **exposure-bias collapse** the naive fine-tune hit. Result: a
model that skips ~15–20% of small-step layer compute on gsm8k with no measurable accuracy loss.

**How this differs from dOPSD:** dOPSD also does on-policy self-distillation for dLLMs, but its teacher's
advantage is *seeing future denoising steps* (temporal) and its goal is *reasoning accuracy*, with **no
layer skipping**. Yours: teacher's advantage is *full depth*, goal is *compute reduction* via dynamic depth.
Same technique family, different axis — so the *method* is yours, the *loss* is cited machinery.
