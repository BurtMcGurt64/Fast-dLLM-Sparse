"""
DRY-RUN lock instrumentation for token-skipping in block diffusion.

This does NOT change the model's computation — it only *measures* how many tokens WOULD be locked
(and when, and how much FFN FLOP that would save) under the cosine/l2 staleness metrics across a
threshold sweep. Zero risk to quality; it's the derisk before building the real gather/scatter.

Question it answers: do current-block tokens settle EARLY (lots of skippable compute) or LATE
(nothing to skip)? One run → the full lock-rate × step × threshold × metric picture.

Mechanism:
  - Forward hooks on each layer's k_proj / v_proj capture per-step K/V (pre-RoPE — valid for
    staleness because same-position RoPE is an orthogonal rotation, identical at both steps, so it
    preserves cosine AND relative-L2).
  - We wrap model.forward to detect small steps (mirroring topk.py's detection) and reset per block.
  - Each small step: staleness(cur K/V vs prev, same block) → lock rate at each candidate threshold.

Usage:
    from skip import LockInstrument
    inst = LockInstrument(model).install()
    ... run generation ...
    inst.report(); inst.uninstall()

NOTE (v1, untested on-device): token alignment across steps assumes the small step processes the
same current-block token positions with a stable T. If T changes (new sub-block/block) we reset.
The first run prints the shapes it actually sees — use that to confirm/fix alignment.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from staleness import cosine_staleness, l2_staleness

# candidate thresholds swept offline from one run (lock when staleness <= thr)
COSINE_THRS = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
L2_THRS     = [1e-2, 5e-2, 1e-1, 2e-1, 5e-1]


class LockInstrument:
    def __init__(self, model, verbose_shapes=3):
        self.model = model
        self.cfg = model.config
        self.n_kv = self.cfg.num_key_value_heads
        self.hd = getattr(self.cfg, "head_dim", self.cfg.hidden_size // self.cfg.num_attention_heads)
        self._orig_forward = None
        self._handles = []
        self._cap = {}          # layer_idx -> {'k':tensor, 'v':tensor} captured THIS forward
        self._prev = {}         # layer_idx -> {'k':tensor, 'v':tensor} from PREVIOUS small step
        self._step_in_block = 0
        self._verbose_left = verbose_shapes
        # stats: dict[(metric, thr)] -> dict[step_idx] -> [sum_lock_rate, count]
        self.stats = {}
        self.ffn_frac = self._ffn_flop_fraction()

    def _ffn_flop_fraction(self):
        """Rough per-token FLOP share of the FFN vs attention projections, so a lock rate maps to a
        theoretical FFN-freeze FLOP saving. FFN: gate+up+down ≈ 3*h*i ; attn proj: q+k+v+o."""
        h, i = self.cfg.hidden_size, self.cfg.intermediate_size
        kv = self.n_kv * self.hd
        ffn = 3 * h * i
        attn = 2 * h * h + 2 * h * kv   # q,o (h*h) + k,v (h*kv)
        return ffn / (ffn + attn)

    # ---- capture hooks on k_proj / v_proj ----
    def _shape_kv(self, out):
        # [B, T, n_kv*hd] -> [B, n_kv, T, hd]
        B, T, _ = out.shape
        return out.view(B, T, self.n_kv, self.hd).transpose(1, 2)

    def _mk_hook(self, layer_idx, which):
        def hook(module, inp, out):
            self._cap.setdefault(layer_idx, {})[which] = self._shape_kv(out.detach())
        return hook

    def install(self):
        if self._orig_forward is not None:
            return self
        for li, layer in enumerate(self.model.model.layers):
            self._handles.append(layer.self_attn.k_proj.register_forward_hook(self._mk_hook(li, "k")))
            self._handles.append(layer.self_attn.v_proj.register_forward_hook(self._mk_hook(li, "v")))

        orig = self.model.forward
        self._orig_forward = orig
        inst = self

        def wrapped(*a, **k):
            pre_kv = 0
            pkv = k.get("past_key_values", None)
            if pkv is not None:
                try: pre_kv = pkv.layers[0].keys.shape[-2]
                except Exception: pre_kv = 0
            upd = k.get("update_past_key_values", False)
            is_commit  = upd and not k.get("use_block_cache", False) and pre_kv > 0
            is_prefill = upd and not k.get("use_block_cache", False) and pre_kv == 0
            is_small   = (pre_kv > 0) and not upd

            if is_prefill or is_commit:            # new block -> reset per-block tracking
                inst._prev = {}
                inst._step_in_block = 0

            inst._cap = {}
            out = orig(*a, **k)                    # hooks fill inst._cap during this call

            if is_small:
                inst._process_small_step()
            return out

        self.model.forward = wrapped
        return self

    def _process_small_step(self):
        if self._verbose_left > 0 and self._cap:
            k0 = self._cap.get(0, {}).get("k")
            if k0 is not None:
                print(f"[LockInstrument] small step {self._step_in_block}: K shape {tuple(k0.shape)} "
                      f"(B,H_kv,T,D)", flush=True)
                self._verbose_left -= 1

        for li, kv in self._cap.items():
            prev = self._prev.get(li)
            if prev is not None and "k" in kv and "k" in prev and kv["k"].shape == prev["k"].shape:
                cos = cosine_staleness(kv["k"], prev["k"])   # [B,T]
                l2  = l2_staleness(kv["k"], prev["k"])       # [B,T]
                self._record("cosine", COSINE_THRS, cos)
                self._record("l2", L2_THRS, l2)
            # roll forward
            self._prev[li] = {kk: vv for kk, vv in kv.items()}

        self._step_in_block += 1

    def _record(self, metric, thrs, staleness):
        s = self._step_in_block
        for thr in thrs:
            rate = (staleness <= thr).float().mean().item()
            d = self.stats.setdefault((metric, thr), {})
            acc = d.setdefault(s, [0.0, 0])
            acc[0] += rate; acc[1] += 1

    # ---- reporting ----
    def report(self):
        print(f"\n{'='*70}\nLOCK-RATE DRY RUN  (FFN FLOP fraction ≈ {self.ffn_frac:.2f}; "
              f"mean lock rate × this ≈ theoretical FFN-freeze FLOP saving)\n{'='*70}")
        for metric, thrs in (("cosine", COSINE_THRS), ("l2", L2_THRS)):
            print(f"\n-- {metric} staleness: lock rate by step-in-block --")
            steps = sorted({s for thr in thrs for s in self.stats.get((metric, thr), {})})
            if not steps:
                print("  (no aligned steps captured — check shapes above)"); continue
            hdr = "  step " + "".join(f"{thr:>9}" for thr in thrs)
            print(hdr); print("  " + "-" * (len(hdr) - 2))
            for s in steps:
                row = f"  {s:>4} "
                for thr in thrs:
                    acc = self.stats.get((metric, thr), {}).get(s)
                    row += f"{(acc[0]/acc[1]):>9.2f}" if acc else f"{'-':>9}"
                print(row)
            # overall mean lock rate per threshold
            print("  MEAN " + "".join(
                f"{self._mean_rate(metric, thr):>9.2f}" for thr in thrs))
        print("\n  Read: high lock rate at EARLY steps = lots of skippable FFN compute (green light).")
        print("  Low/late = little to skip. Multiply mean lock rate by the FFN fraction for the")
        print("  theoretical FFN-freeze FLOP saving (wall-clock will be less — that's the next test).")

    def _mean_rate(self, metric, thr):
        d = self.stats.get((metric, thr), {})
        tot = sum(v[0] for v in d.values()); cnt = sum(v[1] for v in d.values())
        return tot / cnt if cnt else 0.0

    def uninstall(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._orig_forward is not None:
            self.model.forward = self._orig_forward
            self._orig_forward = None

    def __enter__(self): return self.install()
    def __exit__(self, *x): self.uninstall()
