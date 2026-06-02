from .promptir import PromptIR
from .promptir_v2 import PromptIRv2


def build_model(name: str = "promptir", **kwargs):
    if name in ("promptir", "promptir_base"):
        return PromptIR(decoder=True)
    if name in ("promptir_v2", "v2", "simplegate", "adair", "promptir_adair"):
        return PromptIRv2(decoder=True, **kwargs)
    raise ValueError(f"unknown model: {name}")

__all__ = ["PromptIR"]
