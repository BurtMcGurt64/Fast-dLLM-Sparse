from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def load_model(model_id: str = DEFAULT_MODEL, device: str = "cuda:0"):
    # from_pretrained silently skips weight loading for this custom model class;
    # use snapshot_download + manual load_state_dict instead.
    snapshot = snapshot_download(model_id, local_files_only=True)
    config = Fast_dLLM_QwenConfig.from_pretrained(snapshot)
    model = Fast_dLLM_QwenForCausalLM(config).to(torch.bfloat16)

    index_path = os.path.join(snapshot, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    state_dict = {}
    for shard in sorted(set(index["weight_map"].values())):
        state_dict.update(load_file(os.path.join(snapshot, shard), device="cpu"))
    model.load_state_dict(state_dict, strict=True)

    return model.to(device).eval()