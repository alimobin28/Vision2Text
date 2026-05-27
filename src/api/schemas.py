"""
Pydantic response models — the API contract lives here and nowhere else.

Both FastAPI (validation/serialization) and the test suite import from this
module, so renaming a field surfaces as a type error in both places before
anything reaches production.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CaptionResponse(BaseModel):
    caption: str = Field(
        description="Full decoded caption string."
    )
    beam_size: int = Field(
        ge=1,
        description="Beam width used during decoding.",
    )
    inference_ms: float = Field(
        ge=0.0,
        description="Wall-clock time for the complete forward pass, in milliseconds.",
    )
    tokens: list[str] = Field(
        description=(
            "Individual BPE token strings that make up the caption. "
            "Aligned 1-to-1 with the rows of attention_weights."
        )
    )
    attention_weights: list[list[float]] = Field(
        description=(
            "Decoder cross-attention weights averaged over all decoder layers. "
            "Shape [T, 49] — one 7×7 spatial weight map per output token. "
            "tokens[i] attended to image patches described by attention_weights[i]."
        )
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
