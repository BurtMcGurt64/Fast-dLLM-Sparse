"""
ACTIVE FFN-freeze skipping for Fast-dLLM v2 (freeze-target = 'ffn').

For a "locked" token we skip its FFN compute and reuse its cached FFN output; attention (incl. K) is
still recomputed for all tokens. Lock signal is pluggable:

  cosine / l2  : per-LAYER K-staleness (lock layer L's FFN when its K stopped changing). Same-step.
  confidence   : WHOLE-TOKEN, from the output posterior (lock a token's FFN at ALL layers once its
                 predicted-token confidence is high). One-step delayed. This is the ~FREE baseline
                 the decode loop already computes — the bar KV-staleness must beat to be worth it.

Thresholds: cosine/l2 lock when staleness <= threshold; confidence locks when conf >= threshold.

Alignment: within a sub-block the small step keeps the same T token positions, so caches align by
index; on T change (new sub-block/block) we RESET (recompute all, refresh cache). Locks sticky within
an aligned run. WALL-CLOCK: small step is B*8 rows -> memory-bound -> quality measurable at any batch,
wall-clock benefit only at large B (see README).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F
from staleness import cosine_staleness, l2_staleness

_STALENESS = {"cosine": cosine_staleness, "l2": l2_staleness}


class FFNSkipper:
    def __init__(self, model, metric="cosine", threshold=0.01, freeze="ffn"):
        assert metric in ("cosine", "l2", "confidence") and freeze == "ffn"
        self.model = model
        self.metric_name = metric
        self.staleness = _STALENESS.get(metric)     # None for confidence
        self.threshold = threshold
        self.cfg = model.config
        self.n_kv = self.cfg.num_key_value_heads
        self.hd = getattr(self.cfg, "head_dim", self.cfg.hidden_size // self.cfg.num_attention_heads)
        self.active = False
        self._orig_forward = None
        self._handles = []
        self._orig_mlp = {}
        # per-block state
        self.prev_k = {}            # li -> K [B,H_kv,T,D]   (cosine/l2)
        self.locked = {}            # li -> [B,T] bool (per-layer, cosine/l2)
        self.conf_lock = None       # [B,T] bool whole-token (confidence), from PREVIOUS step
        self.ffn_cache = {}         # li -> [B,T,H]
        self._warned = False
        self._lock_sum = 0.0; self._lock_cnt = 0

    def install(self):
        if self._orig_forward is not None:
            return self
        for li, layer in enumerate(self.model.model.layers):
            if self.metric_name in ("cosine", "l2"):
                self._handles.append(layer.self_attn.k_proj.register_forward_hook(self._mk_k_hook(li)))
            self._orig_mlp[li] = layer.mlp.forward
            layer.mlp.forward = self._mk_mlp_forward(li)

        orig = self.model.forward
        self._orig_forward = orig
        sk = self

        def wrapped(*a, **k):
            pre_kv = 0
            pkv = k.get("past_key_values", None)
            if pkv is not None:
                try: pre_kv = pkv.layers[0].keys.shape[-2]
                except Exception: pre_kv = 0
            is_small = (pre_kv > 0) and not k.get("update_past_key_values", False)
            if not is_small:
                sk._reset()
            sk.active = is_small
            out = orig(*a, **k)
            if is_small and sk.metric_name == "confidence":
                sk._update_conf_lock(out)      # capture THIS step's conf -> lock for NEXT step
            return out

        self.model.forward = wrapped
        return self

    def _reset(self):
        self.prev_k = {}; self.locked = {}; self.ffn_cache = {}; self.conf_lock = None

    # ---- confidence lock (whole-token, one-step delayed) ----
    def _update_conf_lock(self, out):
        logits = getattr(out, "logits", None)
        if logits is None or logits.dim() != 3:
            return
        B, T, V = logits.shape
        conf = F.softmax(logits.float(), dim=-1).amax(dim=-1)   # [B,T] max prob per token
        new_lock = conf >= self.threshold
        if self.conf_lock is not None and self.conf_lock.shape == new_lock.shape:
            self.conf_lock = self.conf_lock | new_lock          # sticky
        else:
            self.conf_lock = new_lock
        if not self._warned and T not in (8, 32):
            print(f"[FFNSkipper] note: conf logits T={T} (expected 8/32); verify alignment", flush=True)
            self._warned = True

    # ---- per-layer K-staleness lock (cosine/l2) ----
    def _mk_k_hook(self, li):
        def hook(module, inp, out):
            if not self.active:
                return
            B, T, _ = out.shape
            K = out.detach().view(B, T, self.n_kv, self.hd).transpose(1, 2)
            prev = self.prev_k.get(li)
            if prev is not None and prev.shape == K.shape:
                stale = self.staleness(K, prev)
                new_lock = stale <= self.threshold
                base = self.locked.get(li)
                self.locked[li] = new_lock if base is None or base.shape != new_lock.shape \
                    else (base | new_lock)
            else:
                self.locked[li] = torch.zeros(B, T, dtype=torch.bool, device=K.device)
                self.ffn_cache.pop(li, None)
            self.prev_k[li] = K
            if li == 0:
                self._lock_sum += self.locked[li].float().mean().item(); self._lock_cnt += 1
        return hook

    def _lock_for(self, li, shape_bt):
        lock = self.conf_lock if self.metric_name == "confidence" else self.locked.get(li)
        if lock is not None and li == 0 and self.metric_name == "confidence":
            self._lock_sum += lock.float().mean().item(); self._lock_cnt += 1
        return lock

    # ---- FFN forward with gather/scatter skip ----
    def _mk_mlp_forward(self, li):
        orig = self._orig_mlp[li]
        def fwd(x):
            lock = self._lock_for(li, x.shape[:2])
            cache = self.ffn_cache.get(li)
            if (not self.active) or lock is None or lock.shape != x.shape[:2] \
               or cache is None or cache.shape != x.shape:
                out = orig(x)
                self.ffn_cache[li] = out.detach()
                return out
            B, T, H = x.shape
            flat_x = x.reshape(B * T, H)
            unlocked = (~lock.reshape(B * T)).nonzero(as_tuple=True)[0]
            out = cache.reshape(B * T, H).clone()
            if unlocked.numel() > 0:
                out[unlocked] = orig(flat_x[unlocked])
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
