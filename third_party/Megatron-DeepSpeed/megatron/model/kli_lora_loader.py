# third_party/Megatron-DeepSpeed/megatron/model/kli_lora_loader.py
import importlib.util, os

_KLI_LORA = None

def load_kli_lora():
    """Load your fork's megatron/core/transformer/lora.py by absolute path."""
    global _KLI_LORA
    if _KLI_LORA is not None:
        return _KLI_LORA

    # From this file (.../third_party/Megatron-DeepSpeed/megatron/model/)
    # go up to repo root, then into third_party/kli_megatron/
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, '../../../..'))
    lora_py = os.path.join(
        repo_root,
        'third_party', 'kli_megatron', 'megatron', 'core', 'transformer', 'lora.py'
    )
    if not os.path.exists(lora_py):
        raise FileNotFoundError(f"Cannot find forked lora.py at: {lora_py}")

    spec = importlib.util.spec_from_file_location("kli_lora", lora_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _KLI_LORA = mod
    return mod