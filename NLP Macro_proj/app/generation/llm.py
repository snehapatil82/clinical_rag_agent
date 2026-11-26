"""
LLM Provider Abstraction Layer for Medical RAG.
Supports Groq, OpenAI, Google Gemini, and a deterministic offline Mock provider for zero-cost testing.
"""

from abc import ABC, abstractmethod
import os
import re
from typing import Optional

from app.config import settings
from app.generation.prompts import (
    CLINICAL_RAG_SYSTEM_PROMPT,
    INSUFFICIENT_EVIDENCE_MESSAGE,
)
from app.utils.logging_config import setup_logger

logger = setup_logger("generation.llm")


class LLMProvider(ABC):
    """Abstract Base Class for Large Language Model backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response given a user prompt and optional system prompt."""
        pass


class GroqProvider(LLMProvider):
    """Groq API integration (high-speed inference for Llama, Mixtral, etc.)."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Please set GROQ_API_KEY in .env or environment."
            )
        self.model = model or settings.LLM_MODEL or "llama-3.3-70b-versatile"
        from groq import Groq
        self.client = Groq(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        sys_p = system_prompt or CLINICAL_RAG_SYSTEM_PROMPT
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        max_t = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                max_tokens=max_t,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API Error: {e}", exc_info=True)
            raise RuntimeError(f"Groq Generation Failed: {e}")


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI / OpenAI-compatible API integration (OpenRouter, DeepSeek, LocalAI)."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in .env or environment."
            )
        self.model = model or settings.LLM_MODEL or "gpt-4o-mini"
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        sys_p = system_prompt or CLINICAL_RAG_SYSTEM_PROMPT
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        max_t = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                max_tokens=max_t,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API Error: {e}", exc_info=True)
            raise RuntimeError(f"OpenAI Generation Failed: {e}")


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY in .env or environment."
            )
        self.model_name = model or settings.LLM_MODEL or "gemini-1.5-flash"
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
                    "max_output_tokens": max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
                }
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API Error: {e}", exc_info=True)
            raise RuntimeError(f"Gemini Generation Failed: {e}")


class MockLLMProvider(LLMProvider):
    """
    Deterministic clinical reasoning simulator for offline execution and testing.
    Synthesizes answers strictly from provided context blocks with citations,
    or falls back to insufficient evidence if no relevant context is present.
    """

    def __init__(self, model_name: str = "mock-clinical-v1"):
        self.model_name = model_name
        logger.info(f"Initialized MockLLMProvider ({self.model_name})")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        # Extract context and question from prompt if formatted via template
        context_match = re.search(r"MEDICAL CONTEXT:\s*={10,}\s*(.*?)\s*={10,}", prompt, re.DOTALL)
        question_match = re.search(r"CLINICAL QUESTION:\s*(.*?)\s*(?:INSTRUCTIONS:|$)", prompt, re.DOTALL)

        if not context_match:
            # Baseline prompt without context
            q = prompt.strip()
            return (
                f"[Baseline Response - Unverified LLM Knowledge]\n"
                f"Regarding the query '{q[:60]}...': This condition involves typical physiological mechanisms, "
                f"general clinical presentations, and standard diagnostic approaches as found in general literature. "
                f"(Note: No knowledge base retrieved; information lacks source citations)."
            )

        context_text = context_match.group(1).strip()
        question_text = question_match.group(1).strip() if question_match else ""

        if not context_text or "No medical context retrieved" in context_text:
            return INSUFFICIENT_EVIDENCE_MESSAGE

        # Extract documents cited in the context blocks
        doc_citations = re.findall(r"\[DOCUMENT:\s*(.*?)\s*\|\s*PAGE:\s*(\d+)", context_text)
        unique_citations = []
        for doc, page in doc_citations:
            cite_str = f"{doc} (Page {page})"
            if cite_str not in unique_citations:
                unique_citations.append(cite_str)

        # Extract key sentences from the context that share overlap with the question keywords
        q_words = set(re.findall(r"\w+", question_text.lower())) - {"what", "are", "the", "of", "and", "in", "is", "for", "how", "to"}
        
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context_text) if s.strip() and not s.startswith("[DOCUMENT")]
        relevant_sentences = []
        for s in sentences:
            s_words = set(re.findall(r"\w+", s.lower()))
            overlap = q_words.intersection(s_words)
            if overlap:
                relevant_sentences.append((len(overlap), s))

        relevant_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s[1] for s in relevant_sentences[:4]]

        if not top_sentences:
            top_sentences = sentences[:3] if sentences else ["Evidence retrieved from medical reference."]

        body = " ".join(top_sentences)

        citations_formatted = "\n".join([f"- {c}" for c in unique_citations]) if unique_citations else "- Medical Knowledge Base"

        return (
            f"Based on the provided medical reference documents, {body}\n\n"
            f"Sources Cited:\n{citations_formatted}"
        )


def get_llm_provider(
    provider_name: Optional[str] = None,
    allow_mock_fallback: bool = True
) -> LLMProvider:
    """
    Factory function to instantiate the active LLM Provider based on config and availability.
    """
    prov = (provider_name or settings.LLM_PROVIDER).lower()

    if prov == "groq":
        try:
            return GroqProvider()
        except Exception as e:
            if allow_mock_fallback:
                logger.warning(f"Failed to initialize GroqProvider ({e}). Falling back to MockLLMProvider.")
                return MockLLMProvider()
            raise

    elif prov == "openai":
        try:
            return OpenAICompatibleProvider()
        except Exception as e:
            if allow_mock_fallback:
                logger.warning(f"Failed to initialize OpenAIProvider ({e}). Falling back to MockLLMProvider.")
                return MockLLMProvider()
            raise

    elif prov == "gemini":
        try:
            return GeminiProvider()
        except Exception as e:
            if allow_mock_fallback:
                logger.warning(f"Failed to initialize GeminiProvider ({e}). Falling back to MockLLMProvider.")
                return MockLLMProvider()
            raise

    elif prov == "mock":
        return MockLLMProvider()

    else:
        logger.warning(f"Unknown LLM provider '{prov}'. Defaulting to MockLLMProvider.")
        return MockLLMProvider()
