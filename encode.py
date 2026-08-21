from __future__ import annotations

import os
from typing import Callable, Optional

DEFAULT_MODEL = os.getenv("SEARCH_COLBERT_MODEL", "colbert-ir/colbertv2.0")

TokenEncoder = Callable[[str], list[list[float]]]

_encode: Optional[TokenEncoder] = None
_model_name = DEFAULT_MODEL
_load_error: Optional[str] = None


def set_encoder(fn: Optional[TokenEncoder]) -> None:
    global _encode, _load_error
    _encode = fn
    _load_error = None


def model_name() -> str:
    return _model_name


def status() -> dict:
    return {
        "plugin": "search-colbert",
        "channel": "plugin:colbert",
        "model": _model_name,
        "loaded": _encode is not None,
        "error": _load_error,
    }


def encode_tokens(text: str) -> list[list[float]]:
    encoder = _ensure_encoder()
    return encoder(text or "")


def _ensure_encoder() -> TokenEncoder:
    global _encode, _load_error
    if _encode is not None:
        return _encode
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_model_name)
        model = AutoModel.from_pretrained(_model_name)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model.to(device)
        model.eval()

        def _run(text: str) -> list[list[float]]:
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=180,
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.no_grad():
                hidden = model(**tokens).last_hidden_state.squeeze(0)
            mask = tokens["attention_mask"].squeeze(0)
            rows = []
            for vector, keep in zip(hidden.tolist(), mask.tolist()):
                if int(keep) == 1:
                    rows.append([float(x) for x in vector])
            return rows[1:-1] or rows

        _encode = _run
        _load_error = None
        return _encode
    except Exception as exc:
        _load_error = str(exc)
        raise RuntimeError(
            f"Failed to load ColBERT encoder {_model_name!r}: {exc}"
        ) from exc
