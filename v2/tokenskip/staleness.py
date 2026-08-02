"""
Pluggable staleness metrics for token-skipping in block diffusion.

A token is a candidate to "lock" (skip its future compute) once its K/V representation has
stopped changing across consecutive denoising steps. This module provides the SWAPPABLE metric
that decides "how much did this token change" — so the cosine-vs-L2 question is resolved by
ablation, not argument.

Conventions:
  K/V tensors have shape [B, H_kv, T, D]  (batch, kv-heads, tokens, head-dim).
  All metrics return per-token staleness of shape [B, T], where LOWER = more stable ("staler").
  Aggregation across heads is CONSERVATIVE (take the max change across heads): if ANY head's
  representation moved a lot, the token is not considered settled.

RoPE note: compare K at the SAME token position across steps — RoPE rotation is identical for a
fixed position, so it cancels and the comparison is valid. (V has no RoPE.)
"""
import torch

_EPS = 1e-6


def cosine_staleness(cur: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity per (head, token), then max across heads. [B,H,T,D] -> [B,T].
    Scale-invariant (angle only). Convention in dLLM-Cache / Learning-to-Cache."""
    cos = torch.nn.functional.cosine_similarity(cur.float(), prev.float(), dim=-1)  # [B,H,T]
    change = 1.0 - cos                                                              # [B,H,T]
    return change.amax(dim=1)                                                       # [B,T] conservative


def l2_staleness(cur: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """Relative L2 ‖Δ‖/‖prev‖ per (head, token), then max across heads. [B,H,T,D] -> [B,T].
    Captures MAGNITUDE changes that cosine misses (attention logits q·k are magnitude-sensitive)."""
    num = (cur.float() - prev.float()).norm(dim=-1)          # [B,H,T]
    den = prev.float().norm(dim=-1).clamp_min(_EPS)          # [B,H,T]
    return (num / den).amax(dim=1)                           # [B,T] conservative


METRICS = {
    "cosine": cosine_staleness,   # Xinze's default; convention-aligned
    "l2": l2_staleness,           # ablation; magnitude-aware
}


def compute_staleness(cur, prev, metric: str = "cosine", *, on: str = "k",
                      k_cur=None, k_prev=None, v_cur=None, v_prev=None) -> torch.Tensor:
    """Dispatch. `on` selects the tensor(s): 'k', 'v', or 'kv' (max of both, most conservative).
    Pass either (cur, prev) for a single tensor, or the k_*/v_* pairs when on='kv'."""
    fn = METRICS[metric]
    if on == "kv":
        assert None not in (k_cur, k_prev, v_cur, v_prev), "on='kv' needs k_* and v_* pairs"
        return torch.maximum(fn(k_cur, k_prev), fn(v_cur, v_prev))
    return fn(cur, prev)


def make_lock_mask(staleness: torch.Tensor, threshold: float,
                   already_locked: torch.Tensor = None) -> torch.Tensor:
    """Tokens with staleness <= threshold are 'settled' -> lock. Locks are STICKY: once locked,
    stays locked (SureLock/ES-dLLM behavior). Returns bool [B,T]."""
    lock = staleness <= threshold
    if already_locked is not None:
        lock = lock | already_locked
    return lock


if __name__ == "__main__":
    # sanity: identical -> ~0 staleness -> locks; orthogonal -> high; scaled -> cosine 0, l2 large
    torch.manual_seed(0)
    B, H, T, D = 2, 4, 8, 128
    prev = torch.randn(B, H, T, D)
    cur_same = prev.clone()
    cur_scaled = prev * 2.0                       # same direction, 2x magnitude
    print("cosine, identical (≈0):", cosine_staleness(cur_same, prev).max().item())
    print("l2, identical (≈0):    ", l2_staleness(cur_same, prev).max().item())
    print("cosine, 2x scaled (≈0):", cosine_staleness(cur_scaled, prev).max().item(),
          "  <- cosine BLIND to magnitude")
    print("l2, 2x scaled (≈1.0):  ", l2_staleness(cur_scaled, prev).max().item(),
          "  <- l2 catches it")
    lock = make_lock_mask(cosine_staleness(cur_same, prev), threshold=0.05)
    print("lock mask (all True):  ", lock.all().item())
