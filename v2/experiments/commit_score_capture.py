"""
CommitScoreCapture — per-commit block-importance capture for the CURRENT modeling.py.

Why this exists: the working-tree modeling.py uses SDPA and, during a commit step
(when layer.self_attn._capture_attn_scores is True), stores only the mean commit
query per layer as `_commit_query_mean` [B, H_kv, d]. It no longer materializes
`_last_attn_weights`, so the older BlockAttentionCapture (which reads that attribute)
silently records nothing — every example is skipped.

This capture reconstructs a proper per-block attention-MASS distribution from the
exact signal TopKBlockLoader ranks on:

    logits[pos]   = (q_mean · k[pos]) * scaling        # per KV head, model's own scaling
    attn[pos]     = softmax_pos(logits)                # post-softmax -> sums to 1, non-negative
    block_mass[n] = sum_{pos in block n} attn[pos]      # per head, then mean over heads & layers

So "fraction of blocks holding 90% of mass" (frac90) is well-defined, and it measures
the sparsity of the actual top-K selection signal — not a separate attention readout.

Note (methodology): q_mean is averaged over the 32 commit positions and GQA groups
before the softmax. Averaging shrinks the query norm, which slightly FLATTENS the
softmax -> this is a CONSERVATIVE bias (it makes the model look *less* concentrated
than a per-token readout would). Good for claims that survive it.

Drop-in for BlockAttentionCapture: same .install()/.uninstall()/.records interface,
where records is a list of {prefix_kv_len, n_blocks, scores(np.array[n_blocks])}.
"""

import torch

BLOCK_SIZE = 32


class CommitScoreCapture:
    def __init__(self, model, block_size=BLOCK_SIZE):
        self.model      = model
        self.block_size = block_size
        self.records    = []
        self._original_forward = None

    def install(self):
        cap      = self
        original = self.model.forward
        self._original_forward = original

        def _wrapped(*args, **kwargs):
            pre_kv_len = 0
            try:
                pkv = kwargs.get("past_key_values", None)
                if pkv is not None:
                    pre_kv_len = pkv.layers[0].keys.shape[-2]
            except Exception:
                pass

            is_commit = (kwargs.get("update_past_key_values", False)
                         and not kwargs.get("use_block_cache", False)
                         and pre_kv_len > 0)

            if is_commit:
                for layer in cap.model.model.layers:
                    layer.self_attn._capture_attn_scores = True

            output = original(*args, **kwargs)

            if is_commit:
                for layer in cap.model.model.layers:
                    layer.self_attn._capture_attn_scores = False
                try:
                    prefix_kv_len = output.past_key_values.layers[0].keys.shape[-2]
                except Exception:
                    prefix_kv_len = pre_kv_len
                scores = cap._block_mass(output.past_key_values, prefix_kv_len)
                if scores is not None:
                    cap.records.append({
                        "prefix_kv_len": prefix_kv_len,
                        "n_blocks":      len(scores),
                        "scores":        scores,
                    })
            return output

        self.model.forward = _wrapped
        return self

    def _block_mass(self, past_key_values, prefix_kv_len):
        bs = self.block_size
        n_blocks = prefix_kv_len // bs
        if n_blocks == 0:
            return None

        per_layer = []
        B = None
        with torch.no_grad():
            for i, layer in enumerate(self.model.model.layers):
                attn   = layer.self_attn
                q_mean = getattr(attn, "_commit_query_mean", None)
                attn._commit_query_mean = None
                if q_mean is None:
                    continue
                try:
                    keys = past_key_values[i][0]           # [B, H_kv, prefix_len, d]
                except Exception:
                    continue
                if B is None:
                    B = keys.shape[0]
                d       = keys.shape[3]
                scaling = getattr(attn, "scaling", d ** -0.5)

                k_sliced = keys[:, :, :n_blocks * bs, :]                    # [B, H_kv, n*bs, d]
                q        = q_mean.to(keys.dtype).unsqueeze(2)               # [B, H_kv, 1, d]
                logits   = torch.matmul(q, k_sliced.transpose(-2, -1))      # [B, H_kv, 1, n*bs]
                logits   = logits.squeeze(2).float() * scaling             # [B, H_kv, n*bs]
                attn_p   = torch.softmax(logits, dim=-1)                    # [B, H_kv, n*bs]
                block_mass = attn_p.reshape(B, -1, n_blocks, bs).sum(-1)   # [B, H_kv, n_blocks]
                per_layer.append(block_mass.mean(1))                       # [B, n_blocks]

        if not per_layer or B is None:
            return None
        block_scores = torch.stack(per_layer, 0).mean(0)                   # [B, n_blocks]
        return block_scores[0].float().cpu().numpy()                       # B=1

    def uninstall(self):
        if self._original_forward is not None:
            self.model.forward = self._original_forward
            self._original_forward = None
        for layer in self.model.model.layers:
            layer.self_attn._capture_attn_scores = False
            layer.self_attn._commit_query_mean   = None
