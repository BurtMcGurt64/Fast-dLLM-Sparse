import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import types

BLOCK_SIZE = 32


class TopKBlockLoader:
    """
    After each commit step, selects the top-K most-attended prefix blocks using
    commit-step attention scores and loads only those during subsequent small steps.

    Scoring method: query-weighted block importance.
    During the commit forward, each attention layer stores its mean query vector
    (averaged over GQA groups and query positions) as _commit_query_mean [B, H_kv, d].
    After commit, for each layer we compute:
        scores[b, n] = mean_{head, pos_in_block} q_mean[b,h] · k[b,h,block_n,pos] / sqrt(d)
    then average across layers to get a final per-block importance score.

    This is diffusion-specific: the commit step observes all 32 denoising tokens
    simultaneously, providing a richer attention signal than any single AR query.
    """

    def __init__(self, model, k=None, block_size=BLOCK_SIZE, frac=None, n_sink=2, n_recency=2,
                 selection="commit"):
        self.model      = model
        self.k          = k
        self.frac       = frac          # if set, budget = round(frac * current n_blocks) — scales with context
        self.n_sink     = n_sink        # always-kept blocks at the START (attention sink)
        self.n_recency  = n_recency     # always-kept MOST-RECENT blocks (coherence — fixes long-gen looping)
        self.block_size = block_size
        # selection = where the content blocks come from (baselines for ablation):
        #   "commit"    — OUR method: score at the commit step (clean 32-token queries) + pins
        #   "streaming" — StreamingLLM baseline: sink + recency-fill to budget, NO content selection
        #   "mage"      — MAGE-style: score at the FIRST denoising step (all-mask queries) + pins
        # commit vs streaming isolates whether content selection beats position-only at matched
        # budget; commit vs mage isolates whether the commit-step signal beats first-step scoring.
        self.selection  = selection
        self._mage_scored = False   # MAGE: scored the current block yet? (score once/block)
        self._debug = False         # set True to trace MAGE arming
        self._original_forward = None
        self.last_n_blocks  = 0
        self.last_selected  = 0
        self._n_commits     = 0
        self._sum_n_blocks  = 0
        self._sum_selected  = 0
        self._padding_n_blocks = None  # [B] number of padding blocks at start per seq
        assert k is not None or frac is not None, "provide k (fixed) or frac (dynamic)"
        assert selection in ("commit", "streaming", "mage"), f"bad selection {selection!r}"

    def _budget(self, n_blocks):
        """How many blocks to keep at this commit. frac scales the budget with the
        growing context (prompt + generated), so short-prompt long-gen tasks don't
        collapse to k=1 as generation proceeds."""
        kb = max(1, round(self.frac * n_blocks)) if self.frac is not None else self.k
        return min(kb, n_blocks)

    def _set_capture(self, value):
        for layer in self.model.model.layers:
            layer.self_attn._capture_attn_scores = value

    def _arm_structural(self, prefix_kv_len, past_key_values):
        """
        Arm with position-based structural blocks: sink (0-1) + recency fill to K.
        Safe to call immediately after prefill — no query or commit score needed.
        Sink and recency together cover 78-82% of attention mass deterministically.
        """
        n_blocks = prefix_kv_len // self.block_size
        if n_blocks == 0:
            return
        k_actual = self._budget(n_blocks)

        try:
            keys = past_key_values[0][0]
            B, device = keys.shape[0], keys.device
        except Exception:
            return

        n_sink    = min(2, k_actual)
        n_recency = k_actual - n_sink

        sink_ids = torch.arange(0, n_sink, device=device)
        if n_recency > 0:
            rec_start  = max(n_sink, n_blocks - n_recency)
            recency_ids = torch.arange(rec_start, n_blocks, device=device)
            block_ids_1d = torch.cat([sink_ids, recency_ids]).sort().values
        else:
            block_ids_1d = sink_ids

        block_ids     = block_ids_1d.unsqueeze(0).expand(B, -1).contiguous()
        offsets       = torch.arange(self.block_size, device=device)
        token_indices = (
            block_ids.unsqueeze(-1) * self.block_size + offsets
        ).reshape(B, k_actual * self.block_size)

        tail_start = n_blocks * self.block_size
        if tail_start < prefix_kv_len:
            tail = torch.arange(tail_start, prefix_kv_len, device=device).unsqueeze(0).expand(B, -1)
            token_indices = torch.cat([token_indices, tail], dim=1)

        if token_indices.shape[1] < prefix_kv_len:
            self._arm_layers(token_indices, block_ids)
            # track stats so avg_stats reports the streaming frac too
            self.last_n_blocks  = n_blocks
            self.last_selected  = k_actual
            self._n_commits    += 1
            self._sum_n_blocks += n_blocks
            self._sum_selected += k_actual

    def _detect_padding_blocks(self, input_ids):
        """Detect how many blocks at the start of each sequence are pure padding (left-pad)."""
        pad_id = getattr(self.model.config, 'eos_token_id', None)
        if pad_id is None or input_ids is None:
            return
        B = input_ids.shape[0]
        counts = []
        for b in range(B):
            ids = input_ids[b]
            non_pad = (ids != pad_id).nonzero(as_tuple=False)
            first_content = non_pad[0].item() if len(non_pad) > 0 else ids.shape[0]
            counts.append(first_content // self.block_size)
        self._padding_n_blocks = counts

    def _compute_selected_indices(self, prefix_kv_len, past_key_values):
        n_blocks = prefix_kv_len // self.block_size
        if n_blocks == 0:
            return None

        all_block_scores = []
        B = None
        device = None

        for i, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn

            # Read and immediately free the captured query mean
            q_mean = getattr(attn, '_commit_query_mean', None)
            attn._commit_query_mean = None

            try:
                keys = past_key_values[i][0]  # [B, H_kv, prefix_len, d]
            except Exception:
                continue

            if B is None:
                B      = keys.shape[0]
                device = keys.device

            H_kv = keys.shape[1]
            d    = keys.shape[3]

            k_sliced  = keys[:, :, :n_blocks * self.block_size, :]
            k_blocked = k_sliced.reshape(B, H_kv, n_blocks, self.block_size, d)

            if q_mean is not None:
                # Query-weighted: dot mean-query with each key, average over block positions
                # q_mean: [B, H_kv, d] → [B, H_kv, 1, d]
                # k_flat: [B, H_kv, d, n_blocks*block_size]
                q_flat = q_mean.to(keys.dtype).unsqueeze(2)  # [B, H_kv, 1, d]
                k_flat = k_sliced.transpose(-2, -1)           # [B, H_kv, d, n*bs]
                raw    = torch.matmul(q_flat, k_flat)          # [B, H_kv, 1, n*bs]
                raw    = raw.reshape(B, H_kv, n_blocks, self.block_size) * (d ** -0.5)
                block_scores_l = raw.mean(dim=(1, 3))          # [B, n_blocks]
            else:
                # Fallback to key-norm if query was not captured (shouldn't happen)
                block_scores_l = k_blocked.norm(dim=-1).mean(dim=(1, 3))

            all_block_scores.append(block_scores_l)

        if not all_block_scores or B is None:
            return None

        block_scores = torch.stack(all_block_scores, dim=0).mean(dim=0)  # [B, n_blocks]

        pad = self._padding_n_blocks  # keep for the sink offset below

        # Zero out scores for padding blocks (left-padded batches)
        if pad is not None:
            for b in range(B):
                pad_b = min(pad[b], n_blocks)
                if pad_b > 0:
                    block_scores[b, :pad_b] = float('-inf')

        # Pin sink + recency so top-K can NEVER drop them. Recency (the most recent
        # committed blocks = the model's own recent output) is what long-generation
        # coherence needs; without it the model loses the thread and loops.
        if self.n_recency > 0:
            block_scores[:, max(0, n_blocks - self.n_recency):] = float('inf')
        if self.n_sink > 0:
            for b in range(B):
                s = min(pad[b], n_blocks) if pad is not None else 0
                block_scores[b, s:min(s + self.n_sink, n_blocks)] = float('inf')
        self._padding_n_blocks = None

        k_actual = self._budget(n_blocks)
        # budget must at least cover the pinned sink+recency blocks
        k_actual = min(n_blocks, max(k_actual, min(self.n_sink + self.n_recency, n_blocks)))

        topk_result   = torch.topk(block_scores, k_actual, dim=1)
        topk_block_ids = topk_result.indices.sort(dim=1).values  # [B, k_actual], ascending

        offsets       = torch.arange(self.block_size, device=device)
        token_indices = (
            topk_block_ids.unsqueeze(-1) * self.block_size + offsets
        ).reshape(B, k_actual * self.block_size)  # [B, K_tokens]

        # Include tail tokens beyond the last complete block
        tail_start = n_blocks * self.block_size
        if tail_start < prefix_kv_len:
            tail = torch.arange(tail_start, prefix_kv_len, device=device)
            tail = tail.unsqueeze(0).expand(B, -1)
            token_indices = torch.cat([token_indices, tail], dim=1)

        self.last_n_blocks  = n_blocks
        self.last_selected  = k_actual
        self._n_commits    += 1
        self._sum_n_blocks += n_blocks
        self._sum_selected += k_actual

        return token_indices, topk_block_ids

    def _arm_layers(self, token_indices, block_ids):
        for layer in self.model.model.layers:
            layer.self_attn._topk_token_indices = token_indices
            layer.self_attn._topk_block_ids     = block_ids

    def _disarm_layers(self):
        for layer in self.model.model.layers:
            layer.self_attn._topk_token_indices = None
            layer.self_attn._topk_block_ids     = None

    def _clear_layers(self):
        for layer in self.model.model.layers:
            layer.self_attn._topk_token_indices  = None
            layer.self_attn._topk_block_ids      = None
            layer.self_attn._capture_attn_scores = False
            layer.self_attn._commit_query_mean   = None

    @property
    def avg_stats(self):
        if self._n_commits == 0:
            return 0, 0, 0.0
        n = self._sum_n_blocks / self._n_commits
        u = self._sum_selected / self._n_commits
        return n, u, u / n if n > 0 else 0.0

    def reset_stats(self):
        self._n_commits = self._sum_n_blocks = self._sum_selected = 0

    def install(self):
        if self._original_forward is not None:
            return self
        self.reset_stats()

        loader = self
        original_forward = self.model.forward
        self._original_forward = original_forward

        def _wrapped_forward(*args, **kwargs):
            pre_kv_len = 0
            try:
                pkv = kwargs.get('past_key_values', None)
                if pkv is not None:
                    pre_kv_len = pkv.layers[0].keys.shape[-2]
            except Exception:
                pass

            mode = loader.selection

            is_prefill = (kwargs.get('update_past_key_values', False)
                          and not kwargs.get('use_block_cache', False)
                          and pre_kv_len == 0)
            is_commit  = (kwargs.get('update_past_key_values', False)
                          and not kwargs.get('use_block_cache', False)
                          and pre_kv_len > 0)
            # A "small step" = any denoising forward (not prefill, not commit): prefix
            # present, cache not being updated. Works WITH or WITHOUT block-cache mode
            # (baseline_compare runs without block cache). MAGE scores the first such step
            # per block (guarded by _mage_scored) using the all-mask query.
            is_small_step = (pre_kv_len > 0
                             and not kwargs.get('update_past_key_values', False))

            # MAGE scores ONCE per block (first all-mask step), then keeps the selection
            # for the block's remaining steps — must NOT re-disarm on every full-block forward.
            if is_commit:
                loader._mage_scored = False          # new block coming -> allow re-score
            mage_score_now = (mode == "mage" and is_small_step
                              and not loader._mage_scored)

            # dense passes: prefill + commit always; MAGE's scoring first-step only
            if is_prefill or is_commit or mage_score_now:
                loader._disarm_layers()

            if is_prefill:
                input_ids = kwargs.get('input_ids', args[0] if args else None)
                loader._detect_padding_blocks(input_ids)

            # arm score capture where THIS mode scores (commit: at commit; mage: first step)
            if (mode == "commit" and is_commit) or mage_score_now:
                loader._set_capture(True)

            output = original_forward(*args, **kwargs)

            def _score_and_arm(prefix_len, pkv_for_scores):
                loader._set_capture(False)
                result = loader._compute_selected_indices(prefix_len, pkv_for_scores)
                if loader._debug:
                    print(f"[MAGE] score prefix_len={prefix_len} -> "
                          f"{'None' if result is None else result[0].shape[1]}", flush=True)
                if result is not None:
                    token_indices, block_ids = result
                    if token_indices.shape[1] < prefix_len:
                        loader._arm_layers(token_indices, block_ids)

            if is_commit:
                try:
                    prefix_kv_len = output.past_key_values.layers[0].keys.shape[-2]
                except Exception:
                    prefix_kv_len = 0
                if prefix_kv_len > 0:
                    if mode == "commit":
                        _score_and_arm(prefix_kv_len, output.past_key_values)   # OUR method
                    elif mode == "streaming":
                        loader._arm_structural(prefix_kv_len, output.past_key_values)  # position-only
                    # mage: selection happens at the next block's first small step, not here

            if mage_score_now and pre_kv_len > 0:
                _score_and_arm(pre_kv_len, kwargs.get('past_key_values'))
                loader._mage_scored = True

            return output

        self.model.forward = _wrapped_forward
        return self

    def uninstall(self):
        if self._original_forward is not None:
            self.model.forward = self._original_forward
            self._original_forward = None
        self._clear_layers()

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.uninstall()
