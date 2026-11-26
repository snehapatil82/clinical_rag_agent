"""
End-to-End Clinical RAG Pipeline Orchestrator.
Integrates safety validation, semantic retrieval, relevance thresholding,
grounded generation, and source provenance tracking.
"""

from dataclasses import dataclass, field
import re
import time
from typing import Any, Dict, List, Optional

from app.config import settings
from app.embeddings.embedder import MedicalEmbedder, get_embedder
from app.generation.llm import LLMProvider, get_llm_provider
from app.generation.prompts import (
    CLINICAL_RAG_SYSTEM_PROMPT,
    CLINICAL_RAG_USER_PROMPT_TEMPLATE,
    BASELINE_PROMPT_TEMPLATE,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    MEDICAL_SAFETY_DISCLAIMER,
    DIAGNOSTIC_REQUEST_REFUSAL,
)
from app.retrieval.retriever import ClinicalRetriever, RetrievalResult
from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.research_paper_searcher import (
    MedicalResearchPaperSearcher,
    ResearchPaper,
    get_research_paper_searcher,
)
from app.utils.logging_config import setup_logger

logger = setup_logger("rag.pipeline")


@dataclass
class RAGResponse:
    """Structured response from the Clinical RAG Pipeline."""
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    evidence: List[RetrievalResult] = field(default_factory=list)
    research_papers: List[ResearchPaper] = field(default_factory=list)
    is_grounded: bool = False
    is_external_literature: bool = False
    confidence: float = 0.0
    status: str = "SUCCESS"  # "SUCCESS", "EXTERNAL_LITERATURE_FOUND", "INSUFFICIENT_EVIDENCE", "REFUSAL_SAFETY", "ERROR"
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "evidence": [e.to_dict() for e in self.evidence],
            "research_papers": [p.to_dict() for p in self.research_papers],
            "is_grounded": self.is_grounded,
            "is_external_literature": self.is_external_literature,
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "latency_ms": round(self.latency_ms, 2),
            "metadata": self.metadata,
        }


class ClinicalRAGPipeline:
    """
    Main pipeline for Retrieval-Augmented Clinical Question Answering.
    """

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        embedder: Optional[MedicalEmbedder] = None,
        llm_provider: Optional[LLMProvider] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ):
        self.embedder = embedder or get_embedder()
        self.vector_store = vector_store or FAISSVectorStore()

        # Load FAISS index if index is not in memory
        if self.vector_store.index is None and self.vector_store.is_indexed():
            self.vector_store.load()

        self.retriever = ClinicalRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            top_k=top_k or settings.TOP_K,
            similarity_threshold=similarity_threshold or settings.SIMILARITY_THRESHOLD,
        )
        self.llm = llm_provider or get_llm_provider()

    def _check_safety_guardrails(self, query: str) -> Optional[str]:
        """
        Check for dangerous diagnostic or prescription requests.
        Returns a polite refusal message if triggered, otherwise None.
        """
        q_lower = query.lower()
        # Specific diagnostic command phrases
        prescriptive_patterns = [
            r"\bdiagnose me\b",
            r"\bdo i have\b",
            r"\bprescribe\b",
            r"\bwhat dosage should i take\b",
            r"\bshould i stop taking\b",
            r"\bchange my medication\b",
            r"\bam i dying\b",
        ]
        for pattern in prescriptive_patterns:
            if re.search(pattern, q_lower):
                return (
                    f"{DIAGNOSTIC_REQUEST_REFUSAL}\n\n"
                    f"*{MEDICAL_SAFETY_DISCLAIMER}*"
                )
        return None

    def _clean_answer(self, text: str) -> str:
        """Strip raw heading labels, cited sources lists, and bracketed citation tags."""
        if not text:
            return ""
        cleaned = text.strip()
        # Remove leading **Answer** or Answer:
        cleaned = re.sub(r'^\s*(\*\*Answer:?\*\*|Answer:?|#+\s*Answer)\s*', '', cleaned, flags=re.IGNORECASE)
        # Remove trailing Cited sources / References section
        cleaned = re.split(r'\n+\s*(?:\*\*)?(?:Cited sources|References|Sources cited|Citations|Sources)(?:\*\*)?:?', cleaned, flags=re.IGNORECASE)[0]
        # Remove bracketed document citations like [doc_name.pdf, p.23] or 【9789240074033-eng.pdf, p.7, chunk 5】
        cleaned = re.sub(r'[\[【][^\]】]*\.(?:pdf|md|txt)[^\]】]*[\]】]', '', cleaned)
        # Clean up any duplicate newlines or trailing whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        return cleaned

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> RAGResponse:
        """
        Execute full RAG pipeline for a given medical query.

        Args:
            question: Clinical question string.
            top_k: Number of chunks to retrieve.
            similarity_threshold: Cosine similarity cutoff.

        Returns:
            RAGResponse object.
        """
        start_time = time.perf_counter()

        if not question or not question.strip():
            return RAGResponse(
                question="",
                answer="Please enter a valid medical question.",
                status="ERROR",
                latency_ms=0.0,
            )

        clean_q = question.strip()

        # Step 1: Safety validation
        safety_refusal = self._check_safety_guardrails(clean_q)
        if safety_refusal:
            latency = (time.perf_counter() - start_time) * 1000
            return RAGResponse(
                question=clean_q,
                answer=safety_refusal,
                status="REFUSAL_SAFETY",
                is_grounded=False,
                confidence=0.0,
                latency_ms=latency,
            )

        # Step 2: Semantic Retrieval
        k = top_k if top_k is not None else self.retriever.top_k
        thresh = similarity_threshold if similarity_threshold is not None else self.retriever.similarity_threshold

        retrieved_evidence = self.retriever.retrieve(clean_q, top_k=k)

        # Step 3: Check Local Evidence vs External Web Research Fallback
        if not retrieved_evidence or not self.retriever.is_evidence_sufficient(retrieved_evidence, threshold=thresh):
            top_score = retrieved_evidence[0].score if retrieved_evidence else 0.0
            logger.info(
                f"Local evidence below threshold for '{clean_q[:40]}' (top_score={top_score:.4f} < {thresh:.4f}). "
                "Triggering External Web Medical Research Paper Search..."
            )

            # Search external published biomedical research papers from web
            searcher = MedicalResearchPaperSearcher(embedder=self.embedder)
            papers = searcher.search_papers(clean_q, max_results=5)

            fallback_ans = "I don't have any documents in the knowledge base related to this question, but here are some research papers you can refer to:"

            if papers:
                latency = (time.perf_counter() - start_time) * 1000
                paper_citations = [f"{p.title} - {p.journal} ({p.pub_year})" for p in papers]

                return RAGResponse(
                    question=clean_q,
                    answer=fallback_ans,
                    sources=paper_citations,
                    evidence=retrieved_evidence,
                    research_papers=papers,
                    is_grounded=True,
                    is_external_literature=True,
                    confidence=papers[0].similarity_score if papers else top_score,
                    status="EXTERNAL_LITERATURE_FOUND",
                    latency_ms=latency,
                )

            # Fallback if paper search returned nothing
            latency = (time.perf_counter() - start_time) * 1000
            return RAGResponse(
                question=clean_q,
                answer=fallback_ans,
                sources=[],
                evidence=retrieved_evidence,
                research_papers=[],
                is_grounded=False,
                is_external_literature=False,
                confidence=top_score,
                status="INSUFFICIENT_EVIDENCE",
                latency_ms=latency,
            )

        # Step 4: Construct Grounded Medical Context
        context_block = self.retriever.format_context_for_llm(retrieved_evidence)

        # Step 5: Format Prompt
        prompt = CLINICAL_RAG_USER_PROMPT_TEMPLATE.format(
            context=context_block,
            question=clean_q,
        )

        # Step 6: Generate Grounded Answer via LLM
        try:
            raw_answer = self.llm.generate(
                prompt=prompt,
                system_prompt=CLINICAL_RAG_SYSTEM_PROMPT,
            )
            clean_answer = self._clean_answer(raw_answer)
        except Exception as gen_err:
            logger.error(f"Generation error in RAG pipeline: {gen_err}", exc_info=True)
            latency = (time.perf_counter() - start_time) * 1000
            return RAGResponse(
                question=clean_q,
                answer=f"An error occurred while generating the answer: {gen_err}",
                sources=[],
                evidence=retrieved_evidence,
                status="ERROR",
                latency_ms=latency,
            )

        # Step 6b: Detect if LLM found the retrieved local context insufficient for this query
        is_refusal = bool(re.search(
            r"(could not find sufficient|insufficient evidence|do not have (sufficient|any) (evidence|documents|information)|not enough information|cannot find sufficient|unable to answer based on the provided)",
            clean_answer,
            re.IGNORECASE
        ))

        if is_refusal:
            logger.info(f"LLM indicated insufficient context for '{clean_q[:40]}'. Triggering External Research Search...")
            searcher = MedicalResearchPaperSearcher(embedder=self.embedder)
            papers = searcher.search_papers(clean_q, max_results=5)
            fallback_ans = "I don't have any documents in the knowledge base related to this question, but here are some research papers you can refer to:"

            if papers:
                latency = (time.perf_counter() - start_time) * 1000
                paper_citations = [f"{p.title} - {p.journal} ({p.pub_year})" for p in papers]
                return RAGResponse(
                    question=clean_q,
                    answer=fallback_ans,
                    sources=paper_citations,
                    evidence=retrieved_evidence,
                    research_papers=papers,
                    is_grounded=True,
                    is_external_literature=True,
                    confidence=papers[0].similarity_score,
                    status="EXTERNAL_LITERATURE_FOUND",
                    latency_ms=latency,
                )

        # Step 7: Extract Citations & Calculate Confidence
        unique_sources = []
        for e in retrieved_evidence:
            citation = e.to_citation()
            if citation not in unique_sources:
                unique_sources.append(citation)

        top_confidence = retrieved_evidence[0].score if retrieved_evidence else 0.0
        latency = (time.perf_counter() - start_time) * 1000

        return RAGResponse(
            question=clean_q,
            answer=clean_answer,
            sources=unique_sources,
            evidence=retrieved_evidence,
            is_grounded=True,
            confidence=top_confidence,
            status="SUCCESS",
            latency_ms=latency,
        )

    def query_baseline(self, question: str) -> str:
        """
        Execute standard LLM generation without RAG context (for academic baseline comparison).
        """
        prompt = BASELINE_PROMPT_TEMPLATE.format(question=question)
        return self.llm.generate(
            prompt=prompt,
            system_prompt="You are a general knowledge medical assistant.",
        )
