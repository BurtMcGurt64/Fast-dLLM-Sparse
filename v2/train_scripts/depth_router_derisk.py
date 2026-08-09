"""
DEPTH-ROUTER DE-RISK — is co-trained DYNAMIC layer sparsity live on Fast-dLLM v2?

This is the go/no-go probe for the ONE surviving layer-sparsity direction (see memory
[[trained-dynamic-depth]]): NOT a router on a frozen model (dead — v2 has no slack, =
the key-drift null), but CO-TRAINING backbone(LoRA)+router so v2 *becomes* depth-adaptive,
MoD/FlexiDepth-style. Slack is trained in from a dense init, so "v2 has no static
redundancy" does not kill it. The block-diffusion hook: route depth along the REFINEMENT
axis — the router is conditioned on the per-block mask ratio t (a convergence proxy:
high t = early/unconverged step -> want MORE depth; low t = nearly converged -> less).

WHAT THIS MEASURES (and what it does NOT):
  - It trains the ACTUAL model (LoRA + per-layer router) and reads the trained policy off
    directly — this is the correct instrument (an inference probe on the frozen model
    cannot predict post-training behavior; that was the earlier confusion).
  - It SIMULATES skipping by gating a layer's net contribution to zero, MoD-style:
        out = input + g * (layer_out - input),   g in [0,1] per (token, layer)
    g=1 -> full layer; g=0 -> identity passthrough (= skip the block for that token).
    This faithfully measures QUALITY-at-budget without yet building the compute-saving
    inference kernel. Real FLOP savings + generation-loop eval + recoverable-KV/dense-commit
    are the NEXT step, only worth building if this probe is positive.

TWO GO/NO-GO READOUTS (printed at the end):
  (1) QUALITY @ BUDGET: student (gated) vs dense teacher KL + masked token-accuracy at the
      target keep-rate. If quality craters at a modest budget (e.g. keep 70%), v2 resists
      being made depth-adaptive -> fold into the honest characterization paper.
  (2) DYNAMISM: keep-rate bucketed by t. If the router learns to spend MORE depth at high t
      (early/unconverged) and LESS at low t (converged), the refinement-axis policy is real
      -> GO. If keep-rate is flat across t and the per-layer mask is ~static, the "dynamic"
      value proposition is dead even if quality holds (it's then just static layer pruning,
      which the comparative-analysis paper already shows fails on distilled models).

LOSS = consistency-distillation to the frozen DENSE teacher (KL on masked positions)
     + native masked-diffusion CE anchor
     + budget term to hit a target average keep-rate.
(The anti-step-inflation term belongs to the inference path — noted, not in this probe.)

Run:
  CUDA_VISIBLE_DEVICES=0 uv run v2/train_scripts/depth_router_derisk.py \
      --train_examples 3000 --steps 400 --target_keep 0.7 --eval_n 300
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)                                   # sibling: nfe_probe_stage2b
sys.path.insert(0, os.path.join(HERE, ".."))               # v2/
sys.path.insert(0, os.path.join(HERE, "../experiments"))   # v2/experiments -> helper.load_model

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from helper import load_model
# reuse the working LoRA + diffusion-batch machinery from the NFE probe
from nfe_probe_stage2b import LoRALinear, inject_lora, build_train_examples

MASK_ID = 151665


# --------------------------------------------------- diffusion batch (returns t)
def make_diffusion_batch_t(input_ids, labels, mask_id, bd, mask_low):
    """As nfe_probe_stage2b.make_diffusion_batch, but ALSO returns per-token mask ratio t
    over the full doubled sequence [B, 2L] (noisy half = per-block t; clean half = 0), so
    the router can condition on the convergence proxy. Loss mask unchanged."""
    B, L = input_ids.shape
    noisy = input_ids.clone()
    blk = input_ids.reshape(B * L // bd, bd)
    bb, ll = blk.shape
    t = torch.rand(bb, device=input_ids.device) * (1.0 - mask_low) + mask_low  # Uniform(mask_low,1)
    eps = 1e-3
    p_mask = ((1 - eps) * t + eps)[:, None].repeat(1, ll)
    mask_indices = torch.rand(bb, ll, device=input_ids.device) < p_mask
    x_t = torch.where(mask_indices, mask_id, blk).reshape(B, L)
    noisy[labels != -100] = x_t[labels != -100]
    lab = labels.clone()
    lab[noisy != mask_id] = -100                          # loss only on masked target tokens

    t_tok = t.reshape(bb, 1).repeat(1, ll).reshape(B, L)  # per-token t for the noisy half
    t_ctx = torch.cat([t_tok, torch.zeros_like(t_tok)], dim=1)  # clean context half = converged (t=0)
    doubled = torch.cat([noisy, input_ids], dim=1)        # [noisy | clean]
    return doubled, lab, t_ctx


# ----------------------------------------------------------------- depth router
class DepthRouter(nn.Module):
    """Per-token keep-gate for one decoder layer, conditioned on (hidden, t).
    Gumbel-sigmoid in train; hard 0/1 (STE) at eval. Bias init positive -> starts ~all-keep
    (identity-safe warm start, like LoRA B=0)."""
    def __init__(self, d, hid=128, init_keep_bias=3.0):
        super().__init__()
        self.t_proj = nn.Linear(1, 16)
        self.net = nn.Sequential(
            nn.Linear(d + 16, hid), nn.GELU(), nn.Linear(hid, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_keep_bias)  # start near keep=1

    def forward(self, hidden, t_ctx, tau=1.0, mode="soft"):
        # hidden [B,S,d], t_ctx [B,S]; mode in {"soft","ste","hard"}
        te = self.t_proj(t_ctx.to(hidden.dtype).unsqueeze(-1))          # [B,S,16]
        logit = self.net(torch.cat([hidden, te], dim=-1)).squeeze(-1)   # [B,S]
        if mode == "hard":                                             # deterministic, no grad (eval/rollout)
            g = (logit > 0).to(hidden.dtype)
            return g, g
        if mode == "ste":                                             # deterministic FORWARD + straight-through grad
            hard = (logit > 0).to(hidden.dtype)                        # matches inference exactly (no noise)
            soft = torch.sigmoid(logit / tau)
            g = hard + (soft - soft.detach())
            return g, hard
        u = torch.rand_like(logit).clamp_(1e-6, 1 - 1e-6)             # "soft": Gumbel-sigmoid (exploration)
        gumbel = torch.log(u) - torch.log(1 - u)
        soft = torch.sigmoid((logit + gumbel) / tau)
        g_hard = (soft > 0.5).to(hidden.dtype)
        g = g_hard + (soft - soft.detach())
        return g, g_hard


class RoutedLayer(nn.Module):
    """Wraps a decoder layer with a DepthRouter. out = input + g*(layer_out - input).
    Shared per-forward state (t_ctx, tau, hard, bypass) is set on the module before each
    model forward (mirrors topk.py's per-layer arming pattern)."""
    def __init__(self, layer, d, init_keep_bias=3.0):
        super().__init__()
        self.layer = layer
        self.router = DepthRouter(d, init_keep_bias=init_keep_bias)
        self._t_ctx = None
        self._tau = 1.0
        self._gate_mode = "soft"   # "soft" (gumbel) | "ste" (det+STE, A1 train) | "hard" (eval)
        self._bypass = False       # teacher pass: force g=1 (dense), no router
        self._force = None         # eval control: force this layer's gate to 0 or 1 (static mask)
        self._rand_p = None        # eval control: random per-token keep at this prob (isolates learned routing)
        self.last_gate = None      # [B,S] hard gate from the most recent forward (for stats)
        self.last_gate_diff = None # [B,S] differentiable STE gate (for the budget loss grad)

    @property
    def self_attn(self):
        # batch_sample's finished-sample cleanup does `layer.self_attn` — delegate to the wrapped layer
        return self.layer.self_attn

    def forward(self, hidden_states, *args, **kwargs):
        out = self.layer(hidden_states, *args, **kwargs)
        if self._bypass:
            self.last_gate = None
            return out
        if self._force is not None:                     # static-mask control
            fv = float(self._force)
            self.last_gate = torch.full(hidden_states.shape[:2], fv,
                                        device=hidden_states.device, dtype=hidden_states.dtype)
            return out if fv >= 0.5 else hidden_states
        if self._rand_p is not None:                    # random per-token control (matched rate)
            g = (torch.rand(hidden_states.shape[:2], device=hidden_states.device)
                 < self._rand_p).to(hidden_states.dtype)
            self.last_gate = g
            return hidden_states + g.unsqueeze(-1) * (out - hidden_states)
        t_ctx = self._t_ctx
        if t_ctx is None or t_ctx.shape[1] != hidden_states.shape[1]:
            # fallback: no step context available -> neutral (t=0)
            t_ctx = torch.zeros(hidden_states.shape[:2], device=hidden_states.device,
                                dtype=hidden_states.dtype)
        g, g_hard = self.router(hidden_states, t_ctx, tau=self._tau, mode=self._gate_mode)
        self.last_gate = g_hard.detach()
        self.last_gate_diff = g            # keeps grad -> budget loss can move the router
        return hidden_states + g.unsqueeze(-1) * (out - hidden_states)


def wrap_layers(model, init_keep_bias=3.0):
    d = model.config.hidden_size
    wrappers = []
    layers = model.model.layers
    for i in range(len(layers)):
        p0 = next(layers[i].parameters())          # match model device AND dtype (bf16)
        w = RoutedLayer(layers[i], d, init_keep_bias=init_keep_bias).to(device=p0.device, dtype=p0.dtype)
        layers[i] = w
        wrappers.append(w)
    n_router = sum(p.numel() for w in wrappers for p in w.router.parameters())
    print(f"  wrapped {len(wrappers)} layers with depth routers ({n_router/1e6:.2f}M router params)", flush=True)
    return wrappers


def set_router_state(wrappers, t_ctx=None, tau=None, hard=None, bypass=None, gate_mode=None):
    for w in wrappers:
        if t_ctx is not None: w._t_ctx = t_ctx
        if tau is not None:   w._tau = tau
        if hard is not None:  w._gate_mode = "hard" if hard else "soft"   # back-compat
        if gate_mode is not None: w._gate_mode = gate_mode
        if bypass is not None: w._bypass = bypass


# ----------------------------------------------------------------------- stats
def gate_stats(wrappers, t_ctx, seq_len):
    """Per-layer keep-rate and keep-rate bucketed by t over the NOISY half (first seq_len).
    Returns (per_layer_keep [L], buckets dict t_range->keep)."""
    t_noisy = t_ctx[:, :seq_len]                          # [B, L]
    buckets = {"t[.5,.7)": (0.5, 0.7), "t[.7,.85)": (0.7, 0.85), "t[.85,1]": (0.85, 1.01)}
    per_layer = []
    bucket_keep = {k: [] for k in buckets}
    for w in wrappers:
        if w.last_gate is None:
            continue
        g = w.last_gate[:, :seq_len]                      # [B, L]
        per_layer.append(g.mean().item())
        for name, (lo, hi) in buckets.items():
            m = (t_noisy >= lo) & (t_noisy < hi)
            if m.any():
                bucket_keep[name].append(g[m].mean().item())
    per_layer = torch.tensor(per_layer) if per_layer else torch.zeros(len(wrappers))
    bucket_avg = {k: (sum(v) / len(v) if v else float("nan")) for k, v in bucket_keep.items()}
    return per_layer, bucket_avg


@torch.no_grad()
def evaluate(model, wrappers, ex, args, bd, V, n_eval):
    """Student(gated, hard) vs dense teacher on held-out batches: KL + masked token-acc at
    the learned budget, plus dynamism buckets. Stays in train mode (no_grad) so the doubled
    [noisy|clean] batch hits the gen_mask path, not eval_mask; dropout is zeroed in main()."""
    model.train()
    g = torch.Generator().manual_seed(args.seed + 1)
    kls, accs, keeps = [], [], []
    per_layer_acc = torch.zeros(len(wrappers))
    buckets_acc = None
    # per-t-bucket AGREEMENT (the headroom diagnostic): where do errors concentrate across t?
    agree_bins = [("t[.3,.5)", 0.3, 0.5), ("t[.5,.7)", 0.5, 0.7),
                  ("t[.7,.85)", 0.7, 0.85), ("t[.85,1]", 0.85, 1.01)]
    agree_bk = {name: [] for name, _, _ in agree_bins}
    n_batches = max(1, n_eval // args.batch_size)
    for _ in range(n_batches):
        idx = torch.randint(0, len(ex), (args.batch_size,), generator=g)
        input_ids = torch.tensor([ex[i][0] for i in idx], device=args.device)
        labels = torch.tensor([ex[i][1] for i in idx], device=args.device)
        doubled, lab, t_ctx = make_diffusion_batch_t(input_ids, labels, MASK_ID, bd, args.mask_low)
        set_router_state(wrappers, t_ctx=t_ctx)
        mpos = (lab.reshape(-1) != -100)

        # teacher: dense
        set_router_state(wrappers, bypass=True)
        t_out = model.model(input_ids=doubled, labels=lab, block_size=bd, use_cache=False)
        t_logits = model.lm_head(t_out.last_hidden_state[:, :args.seq_len, :]).reshape(-1, V).float()

        # student: hard-gated
        set_router_state(wrappers, bypass=False, hard=True)
        s_out = model.model(input_ids=doubled, labels=lab, block_size=bd, use_cache=False)
        s_logits = model.lm_head(s_out.last_hidden_state[:, :args.seq_len, :]).reshape(-1, V).float()

        kl = F.kl_div(F.log_softmax(s_logits[mpos], -1), F.softmax(t_logits[mpos], -1),
                      reduction="batchmean")
        acc = (s_logits[mpos].argmax(-1) == t_logits[mpos].argmax(-1)).float().mean()
        kls.append(kl.item()); accs.append(acc.item())
        # agreement bucketed by t (masked positions only)
        tt_all = t_ctx[:, :args.seq_len].reshape(-1)
        match_all = (s_logits.argmax(-1) == t_logits.argmax(-1))
        for name, lo, hi in agree_bins:
            m = mpos & (tt_all >= lo) & (tt_all < hi)
            if m.any():
                agree_bk[name].append(match_all[m].float().mean().item())
        pl, ba = gate_stats(wrappers, t_ctx, args.seq_len)
        per_layer_acc += pl / n_batches
        keeps.append(pl.mean().item())
        if buckets_acc is None:
            buckets_acc = {k: v / n_batches for k, v in ba.items()}
        else:
            for k in ba:
                buckets_acc[k] += ba[k] / n_batches
    set_router_state(wrappers, hard=False)
    agree_avg = {k: (sum(v) / len(v) if v else float("nan")) for k, v in agree_bk.items()}
    return (sum(kls) / len(kls), sum(accs) / len(accs), sum(keeps) / len(keeps),
            per_layer_acc, buckets_acc, agree_avg)


@torch.no_grad()
def forced_agreement(model, wrappers, ex, args, bd, V, n_eval, forces):
    """THE decisive dynamism control: fix a STATIC per-layer 0/1 mask (forces) and measure
    argmax-agreement vs the dense teacher. Compare to the dynamic router at matched budget.
    dynamic >> static  => per-token/refinement routing adds value (real contribution).
    dynamic ~= static   => it's just trained layer-dropping (weak, near comparative-paper turf)."""
    model.train()
    gen = torch.Generator().manual_seed(args.seed + 2)
    accs = []
    nb = max(1, n_eval // args.batch_size)
    for _ in range(nb):
        idx = torch.randint(0, len(ex), (args.batch_size,), generator=gen)
        input_ids = torch.tensor([ex[i][0] for i in idx], device=args.device)
        labels = torch.tensor([ex[i][1] for i in idx], device=args.device)
        doubled, lab, t_ctx = make_diffusion_batch_t(input_ids, labels, MASK_ID, bd, args.mask_low)
        set_router_state(wrappers, t_ctx=t_ctx)
        mpos = (lab.reshape(-1) != -100)
        for w in wrappers:
            w._force = None
        set_router_state(wrappers, bypass=True)
        t_logits = model.lm_head(model.model(input_ids=doubled, labels=lab, block_size=bd,
                   use_cache=False).last_hidden_state[:, :args.seq_len, :]).reshape(-1, V).float()
        set_router_state(wrappers, bypass=False)
        for w, f in zip(wrappers, forces):
            w._force = f
        s_logits = model.lm_head(model.model(input_ids=doubled, labels=lab, block_size=bd,
                   use_cache=False).last_hidden_state[:, :args.seq_len, :]).reshape(-1, V).float()
        accs.append((s_logits[mpos].argmax(-1) == t_logits[mpos].argmax(-1)).float().mean().item())
    for w in wrappers:
        w._force = None
    return sum(accs) / len(accs)


@torch.no_grad()
def random_agreement(model, wrappers, ex, args, bd, V, n_eval, per_layer_keep):
    """THE control that isolates the LEARNED router: random per-token keep at each layer's
    OWN learned rate (per-layer budget preserved, only WHICH tokens is randomized).
    dynamic >> random => the router picks the right tokens (real learned routing).
    dynamic ~= random => it's just trained stochastic depth; the learned router adds nothing."""
    model.train()
    gen = torch.Generator().manual_seed(args.seed + 3)
    accs = []
    nb = max(1, n_eval // args.batch_size)
    for _ in range(nb):
        idx = torch.randint(0, len(ex), (args.batch_size,), generator=gen)
        input_ids = torch.tensor([ex[i][0] for i in idx], device=args.device)
        labels = torch.tensor([ex[i][1] for i in idx], device=args.device)
        doubled, lab, t_ctx = make_diffusion_batch_t(input_ids, labels, MASK_ID, bd, args.mask_low)
        set_router_state(wrappers, t_ctx=t_ctx)
        mpos = (lab.reshape(-1) != -100)
        for w in wrappers:
            w._rand_p = None
        set_router_state(wrappers, bypass=True)
        t_logits = model.lm_head(model.model(input_ids=doubled, labels=lab, block_size=bd,
                   use_cache=False).last_hidden_state[:, :args.seq_len, :]).reshape(-1, V).float()
        set_router_state(wrappers, bypass=False)
        for w, p in zip(wrappers, per_layer_keep.tolist()):
            w._rand_p = p
        s_logits = model.lm_head(model.model(input_ids=doubled, labels=lab, block_size=bd,
                   use_cache=False).last_hidden_state[:, :args.seq_len, :]).reshape(-1, V).float()
        accs.append((s_logits[mpos].argmax(-1) == t_logits[mpos].argmax(-1)).float().mean().item())
    for w in wrappers:
        w._rand_p = None
    return sum(accs) / len(accs)


def scheduled_budget(wrappers, t_ctx, seq_len, target_keep, slope, t_mean, mask_low, n_bins):
    """A3: constrain the AVERAGE keep within each t-bucket to follow a schedule
        target(t) = target_keep + slope * (t - t_mean)     [E_t = target_keep -> matched avg budget]
    slope>0 keeps MORE depth at high t (early/unconverged). Buckets (not per-token) so content
    routing WITHIN a bucket is preserved; only the per-t average is pushed. Returns (budget, mean_keep)."""
    gk = torch.stack([w.last_gate_diff[:, :seq_len] for w in wrappers], 0).mean(0)  # [B, seq] per-token keep
    t_tok = t_ctx[:, :seq_len]
    edges = torch.linspace(mask_low, 1.0, n_bins + 1, device=gk.device)
    terms = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (t_tok >= lo) & ((t_tok < hi) if b < n_bins - 1 else (t_tok <= hi))
        if m.any():
            tgt = (target_keep + slope * (t_tok[m].mean() - t_mean)).clamp(0.05, 1.0)
            terms.append((gk[m].mean() - tgt) ** 2)
    budget = torch.stack(terms).mean() if terms else gk.mean() * 0.0
    return budget, gk.mean().detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--train_examples", type=int, default=3000)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=256, help="multiple of bd_size (32)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mask_low", type=float, default=0.3,
                   help="block mask ratio ~ Uniform(mask_low,1); spread gives the router t-signal")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--target_keep", type=float, default=0.7, help="target avg fraction of layers kept")
    p.add_argument("--lambda_budget", type=float, default=5.0)
    p.add_argument("--sched_slope", type=float, default=0.0,
                   help="A3: t-dependent budget at matched avg. 0=flat; >0 keeps MORE depth at high t "
                        "(early/unconverged); <0 = anti-convergence sanity check.")
    p.add_argument("--sched_bins", type=int, default=5)
    p.add_argument("--lambda_ce", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=1.0, help="Gumbel-sigmoid temperature")
    p.add_argument("--eval_n", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None)
    args = p.parse_args()

    print("Loading model, injecting LoRA, wrapping layers with routers...", flush=True)
    model = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    bd = model.model.bd_size
    assert args.seq_len % bd == 0, f"seq_len must be a multiple of bd_size={bd}"
    inject_lora(model, args.lora_r, args.lora_alpha)   # base frozen, LoRA trainable
    for m in model.modules():                           # deterministic eval w/o mode switch
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    wrappers = wrap_layers(model)                       # routers trainable (fresh params)
    V = model.config.vocab_size

    trainable = [q for q in model.parameters() if q.requires_grad]
    print(f"  {sum(x.numel() for x in trainable)/1e6:.1f}M trainable (LoRA + routers)", flush=True)

    print("\nBuilding train data...", flush=True)
    ex = build_train_examples(tokenizer, args.train_examples, args.seq_len, args.seed)
    print(f"  {len(ex)} training examples", flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    t_mean = (args.mask_low + 1.0) / 2.0                 # E[t] -> schedule keeps avg budget matched

    print(f"\n=== TRAIN  steps={args.steps}  target_keep={args.target_keep}  "
          f"sched_slope={args.sched_slope} ===", flush=True)
    for step in range(args.steps):
        model.train()
        idx = torch.randint(0, len(ex), (args.batch_size,), generator=gen)
        input_ids = torch.tensor([ex[i][0] for i in idx], device=args.device)
        labels = torch.tensor([ex[i][1] for i in idx], device=args.device)
        doubled, lab, t_ctx = make_diffusion_batch_t(input_ids, labels, MASK_ID, bd, args.mask_low)
        set_router_state(wrappers, t_ctx=t_ctx, tau=args.tau, hard=False)
        mpos = (lab.reshape(-1) != -100)

        # teacher (dense, frozen path) — no grad
        with torch.no_grad():
            set_router_state(wrappers, bypass=True)
            t_out = model.model(input_ids=doubled, labels=lab, block_size=bd, use_cache=False)
            t_logits = model.lm_head(t_out.last_hidden_state[:, :args.seq_len, :]).reshape(-1, V).float()

        # student (routed) — with grad
        set_router_state(wrappers, bypass=False)
        s_out = model.model(input_ids=doubled, labels=lab, block_size=bd, use_cache=False)
        s_logits = model.lm_head(s_out.last_hidden_state[:, :args.seq_len, :]).reshape(-1, V).float()

        kl = F.kl_div(F.log_softmax(s_logits[mpos], -1), F.softmax(t_logits[mpos], -1),
                      reduction="batchmean")
        ce = F.cross_entropy(s_logits, lab.reshape(-1), ignore_index=-100)
        budget, keep = scheduled_budget(wrappers, t_ctx, args.seq_len, args.target_keep,
                                        args.sched_slope, t_mean, args.mask_low, args.sched_bins)
        loss = kl + args.lambda_ce * ce + args.lambda_budget * budget
        loss.backward()
        opt.step(); opt.zero_grad()

        if step % 20 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.3f}  KL {kl.item():.3f}  "
                  f"CE {ce.item():.3f}  keep {keep.item():.3f}  "
                  f"peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
        if args.eval_every and (step + 1) % args.eval_every == 0:
            kl_e, acc_e, keep_e, _, ba, _ = evaluate(model, wrappers, ex, args, bd, V, min(120, args.eval_n))
            dyn = "  ".join(f"{k}={v:.2f}" for k, v in ba.items())
            print(f"    [eval @{step+1}] KL={kl_e:.3f} agree={acc_e:.1%} keep={keep_e:.2f} | {dyn}", flush=True)

    # ---------------------------------------------------------------- final readout
    kl_e, acc_e, keep_e, per_layer, ba, agree_bk = evaluate(model, wrappers, ex, args, bd, V, args.eval_n)
    print("\n" + "=" * 60)
    print(f"DEPTH-ROUTER DE-RISK  (target_keep={args.target_keep})")
    print("=" * 60)
    print(f"(1) QUALITY @ BUDGET: keep={keep_e:.2f}  student-vs-dense KL={kl_e:.3f}  "
          f"argmax-agreement={acc_e:.1%}")
    print("    -> high agreement (say >90%) at a real budget cut = v2 CAN be made depth-adaptive.")
    print(f"\n(2) DYNAMISM (keep-rate by mask ratio t; convergence proxy):")
    for k, v in ba.items():
        print(f"      {k:>10}  keep={v:.3f}")
    spread = (max(v for v in ba.values() if v == v) - min(v for v in ba.values() if v == v))
    print(f"    t-spread = {spread:.3f}  (want MORE keep at high t / early step; flat = static policy)")
    print(f"\n    per-layer keep-rate: " + " ".join(f"{x:.2f}" for x in per_layer.tolist()))
    plspread = (per_layer.max() - per_layer.min()).item()
    print(f"    per-layer spread = {plspread:.3f}  (some structure expected; fully uniform is odd)")

    # (2d) HEADROOM diagnostic: agreement-vs-dense bucketed by t (needs NO forced schedule)
    print(f"\n(2d) AGREEMENT by mask ratio t (where do errors concentrate?):")
    for k in ("t[.3,.5)", "t[.5,.7)", "t[.7,.85)", "t[.85,1]"):
        print(f"      {k:>10}  agree={agree_bk[k]:.3f}")
    vals = [v for v in agree_bk.values() if v == v]
    ag_spread = (max(vals) - min(vals)) if vals else float("nan")
    print(f"    agreement spread = {ag_spread:.3f}")
    print("    HIGH-t worse than LOW-t => compute starved on unconverged tokens => convergence")
    print("    reallocation has headroom (build A1). FLAT => no headroom => Hook A dead, use Hook B.")

    # (2b) THE decisive control: dynamic routing vs a matched-budget STATIC per-layer mask
    n_skip = round((1.0 - keep_e) * len(wrappers))
    order = torch.argsort(per_layer)                    # lowest learned keep first
    forces = [1.0] * len(wrappers)
    for i in order[:n_skip].tolist():
        forces[i] = 0.0
    static_agree = forced_agreement(model, wrappers, ex, args, bd, V, min(200, args.eval_n), forces)
    print(f"\n(2b) DYNAMISM CONTROL (matched budget):")
    print(f"      dynamic per-token routing agreement = {acc_e:.1%}  (keep={keep_e:.2f})")
    print(f"      static: skip {n_skip} lowest-keep layers = {static_agree:.1%}  (keep={1 - n_skip/len(wrappers):.2f})")
    print(f"      gap = {acc_e - static_agree:+.1%}  (dynamic minus static)")

    # (2c) THE control that isolates the LEARNED router from generic stochastic-depth robustness
    rand_agree = random_agreement(model, wrappers, ex, args, bd, V, min(200, args.eval_n), per_layer)
    print(f"\n(2c) LEARNED-ROUTER CONTROL (matched per-layer budget, tokens randomized):")
    print(f"      dynamic (learned router) agreement = {acc_e:.1%}")
    print(f"      random per-token @ same rates      = {rand_agree:.1%}")
    print(f"      gap = {acc_e - rand_agree:+.1%}  (learned minus random)")
    print("\nGO if (1) holds AND (2b) dynamic >> static AND (2c) learned >> random.")
    print("If (2c) learned ~= random -> it's trained stochastic depth; the LEARNED router adds")
    print("nothing (you could skip randomly). If (2) t-spread ~0 the win is per-TOKEN, not")
    print("refinement-axis -> reframe as MoD-for-diffusion, not the convergence story.")

    if args.save:
        sd = {k: v for k, v in model.state_dict().items()
              if ".A." in k or ".B." in k or ".router." in k}
        torch.save(sd, args.save)
        print(f"\nsaved LoRA+router -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
