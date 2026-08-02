# tokenskip — load-bearing foundation (DO NOT strip)

This project builds on the summer's working Fast-dLLM v2 code. The following changes to the
base model are **load-bearing** — removing them silently reintroduces OOM / correctness bugs.
Verified present 2026-08-01.

| File | Change | Why it's essential |
|---|---|---|
| `v2/model/modeling.py:52` | `flex_attention(..., enable_gqa=True)` (training path) | GQA without repeat_kv |
| `v2/model/modeling.py:285` | SDPA `enable_gqa=True` (inference path) | HF only passes enable_gqa when mask is None; without this KV H_kv 4→28 → +30 GB OOM |
| `v2/model/modeling.py:661,727` | `logits_to_keep` handling | prefill with `logits_to_keep=1` avoids the 80 GB lm_head spike at B=32 |
| `v2/experiments/helper.py:16` | `DynamicCache.__getitem__` + `.layers` patch | model uses non-standard `cache.layers[i].keys/values`, not `key_cache[i]` |
| `v2/experiments/helper.py` | `load_model()` (snapshot_download + manual load_state_dict) | `from_pretrained` silently skips weight loading for this model |

Model load path (do not use `AutoModelForCausalLM.from_pretrained`):
```python
from helper import load_model            # v2/experiments/helper.py
model = load_model("Efficient-Large-Model/Fast_dLLM_v2_7B", device="cuda")
```

Env: uv, at repo root (`pyproject.toml` + `uv.lock`). Run with `uv run v2/tokenskip/<script>.py`.
`.venv/` is gitignored — rebuild on the server with `uv sync` if needed.
