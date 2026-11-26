"""
LLM abstraction and prompt management module for Medical RAG.
"""

from app.generation.llm import LLMProvider, get_llm_provider
from app.generation.prompts import (
    CLINICAL_RAG_SYSTEM_PROMPT,
    CLINICAL_RAG_USER_PROMPT_TEMPLATE,
    BASELINE_PROMPT_TEMPLATE,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    MEDICAL_SAFETY_DISCLAIMER,
)

__all__ = [
    "LLMProvider",
    "get_llm_provider",
    "CLINICAL_RAG_SYSTEM_PROMPT",
    "CLINICAL_RAG_USER_PROMPT_TEMPLATE",
    "BASELINE_PROMPT_TEMPLATE",
    "INSUFFICIENT_EVIDENCE_MESSAGE",
    "MEDICAL_SAFETY_DISCLAIMER",
]
