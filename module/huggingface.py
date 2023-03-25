from huggingface_hub import hf_hub_download

def import_index_from_hf(path: str):
    return hf_hub_download(
        repo_id=path,
        filename="model_index.json",
    )