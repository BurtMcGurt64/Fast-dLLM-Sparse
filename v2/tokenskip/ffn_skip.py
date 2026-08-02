"""
ACTIVE FFN-freeze skipping for Fast-dLLM v2 (freeze-target = 'ffn').

For a "locked" (stale-K) token we skip its FFN compute and reuse its cached FFN output; attention
(incl. K) is still recomputed for all tokens (that's the 'ffn' freeze-target — the one with most of
the compute). Lock decision comes from staleness.py (cosine/l2), swappable.

Correctness note (alignment): within a sub-block the small step processes the SAME T token positions
across consecutive steps, so we align caches by index. If T changes (new sub-block / block) we RESET:
recompute all tokens that step and refresh the cache. Locks are sticky within an aligned run.

WALL-CLOCK caveat: at the small step the FFN is B×8 rows → memory-bound → skipping tokens saves FLOPs
but little wall-clock until B is large (B×8 ≳ 260). Quality is measurable at any batch; measure
wall-clock at B≥32. See README.

Usage:
    sk = FFNSkipper(model, metric="cosine", threshold=0.01).install()
    ... generate ...
    print(sk.mean_lock_rate()); sk.uninstall()
"""
import sys, os, types
sys.path.insert(0, os.path.dirname(__file__))
import torch
from staleness import cosine_staleness, l2_staleness

_METRIC = {"cosine": cosine_staleness, "l2": l2_staleness}


class FFNSkipper:
    def __init__(self, model, metric="cosine", threshold=0.01, freeze="ffn"):
        assert metric in _METRIC and freeze == "ffn", "v1 supports freeze='ffn'"
        self.model = model
        self.metric = _METRIC[metric]
        self.threshold = threshold
        self.cfg = model.config
        self.n_kv = self.cfg.num_key_value_heads
        self.hd = getattr(self.cfg, "head_dim", self.cfg.hidden_size // self.cfg.num_attention_heads)
        self.active = False
        self._orig_forward = None
        self._handles = []
        self._orig_mlp = {}          # layer_idx -> original bound mlp.forward
        # per-block state (reset each block/sub-block-misalign)
        self.prev_k = {}             # li -> K [B,H_kv,T,D]
        self.locked = {}             # li -> [B,T] bool (sticky within aligned run)
        self.ffn_cache = {}          # li -> [B,T,H] last FFN output
        # stats
        self._lock_sum = 0.0; self._lock_cnt = 0

    # ---- install ----
    def install(self):
        if self._orig_forward is not None:
            return self
        for li, layer in enumerate(self.model.model.layers):
            layer.self_attn._skip_li = li
            layer.mlp._skip_li = li
            self._handles.append(layer.self_attn.k_proj.register_forward_hook(self._mk_k_hook(li)))
            self._orig_mlp[li] = layer.mlp.forward
            layer.mlp.forward = self._mk_mlp_forward(li, layer.mlp)

        orig = self.model.forward
        self._orig_forward = orig
        sk = self

        def wrapped(*a, **k):
            pre_kv = 0
            pkv = k.get("past_key_values", None)
            if pkv is not None:
                try: pre_kv = pkv.layers[0].keys.shape[-2]
                except Exception: pre_kv = 0
            upd = k.get("update_past_key_values", False)
            is_small = (pre_kv > 0) and not upd
            if not is_small:                       # prefill / commit -> new block, reset everything
                sk._reset()
            sk.active = is_small
            return orig(*a, **k)

        self.model.forward = wrapped
        return self

    def _reset(self):
        self.prev_k = {}; self.locked = {}; self.ffn_cache = {}

    # ---- lock decision (k_proj hook, runs BEFORE mlp in the same layer) ----
    def _mk_k_hook(self, li):
        def hook(module, inp, out):
            if not self.active:
                return
            B, T, _ = out.shape
            K = out.detach().view(B, T, self.n_kv, self.hd).transpose(1, 2)  # [B,H_kv,T,D]
            prev = self.prev_k.get(li)
            if prev is not None and prev.shape == K.shape:
                stale = self.metric(K, prev)                    # [B,T]
                new_lock = stale <= self.threshold
                base = self.locked.get(li)
                self.locked[li] = new_lock if base is None or base.shape != new_lock.shape \
                    else (base | new_lock)                      # sticky within aligned run
            else:
                self.locked[li] = torch.zeros(B, T, dtype=torch.bool, device=K.device)  # misalign -> no skip
                self.ffn_cache.pop(li, None)                    # stale cache invalid
            self.prev_k[li] = K
            # record lock rate (layer 0 as representative to avoid double counting per step)
            if li == 0 and self.locked[li] is not None:
                self._lock_sum += self.locked[li].float().mean().item(); self._lock_cnt += 1
        return hook

    # ---- FFN forward with gather/scatter skip ----
    def _mk_mlp_forward(self, li, mlp):
        orig = self._orig_mlp[li]
        def fwd(x):
            lock = self.locked.get(li)
            cache = self.ffn_cache.get(li)
            if (not self.active) or lock is None or lock.shape != x.shape[:2] \
               or cache is None or cache.shape != x.shape:
                out = orig(x)                          # full compute
                self.ffn_cache[li] = out.detach()
                return out
            B, T, H = x.shape
            flat_x = x.reshape(B * T, H)
            flat_lock = lock.reshape(B * T)
            unlocked = (~flat_lock).nonzero(as_tuple=True)[0]
            out = cache.reshape(B * T, H).clone()
            if unlocked.numel() > 0:
                out[unlocked] = orig(flat_x[unlocked])  # FFN only on unlocked rows
            out = out.reshape(B, T, H)
            self.ffn_cache[li] = out.detach()
            return out
        return fwd

    def mean_lock_rate(self):
        return self._lock_sum / self._lock_cnt if self._lock_cnt else 0.0

    def uninstall(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        for li, orig in self._orig_mlp.items():
            self.model.model.layers[li].mlp.forward = orig
        self._orig_mlp = {}
        if self._orig_forward is not None:
            self.model.forward = self._orig_forward
            self._orig_forward = None
        self._reset()

    def __enter__(self): return self.install()
    def __exit__(self, *x): self.uninstall()
