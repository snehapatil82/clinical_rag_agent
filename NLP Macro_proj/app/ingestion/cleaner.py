"""
Medical-safe text cleaning module.
Performs non-destructive normalization while strictly preserving medical terminology,
clinical abbreviations, drug dosages, units of measurement, and anatomical references.
"""

import re
import unicodedata
from typing import List

from app.ingestion.loaders import Document
from app.utils.logging_config import setup_logger

logger = setup_logger("ingestion.cleaner")


class MedicalTextCleaner:
    """
    Cleans raw document text while preserving domain-critical medical information.
    Ensures numerical values, units, abbreviations, and clinical entities remain intact.
    """

    def __init__(self):
        # Common extraction artifacts to remove
        self._bullet_pattern = re.compile(r"^[\s•\-\*■◆○►]+\s*", re.MULTILINE)
        self._whitespace_pattern = re.compile(r"[ \t]+")
        self._multi_newline_pattern = re.compile(r"\n{3,}")
        self._null_bytes_pattern = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
        self._page_num_footer_pattern = re.compile(
            r"^(?:page\s+\d+(?:\s+of\s+\d+)?|\d+\s*/\s*\d+|\b\d+\b)$",
            re.IGNORECASE | re.MULTILINE
        )

    def clean_text(self, raw_text: str) -> str:
        """
        Normalize text preserving clinical semantics.

        Args:
            raw_text: Uncleaned extracted text string.

        Returns:
            Normalized and cleaned text.
        """
        if not raw_text:
            return ""

        # 1. Normalize unicode characters (NFKC)
        text = unicodedata.normalize("NFKC", raw_text)

        # 2. Strip null and unprintable control characters
        text = self._null_bytes_pattern.sub("", text)

        # 3. Normalize quotes and dashes
        text = (
            text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("—", " - ")
            .replace("–", " - ")
        )

        # 4. Remove isolated page number footers on single lines
        lines = text.splitlines()
        cleaned_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if self._page_num_footer_pattern.match(stripped) and len(stripped) < 15:
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # 5. Normalize repeated whitespace within lines
        text = self._whitespace_pattern.sub(" ", text)

        # 6. Normalize excessive blank lines (max 2 consecutive newlines)
        text = self._multi_newline_pattern.sub("\n\n", text)

        return text.strip()

    def clean_document(self, doc: Document) -> Document:
        """
        Clean a single Document object, returning a new Document instance.
        """
        cleaned_text = self.clean_text(doc.text)
        return Document(text=cleaned_text, metadata=dict(doc.metadata))

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean a batch of Document objects.
        """
        cleaned_docs: List[Document] = []
        for doc in documents:
            cleaned = self.clean_document(doc)
            if cleaned.text.strip():
                cleaned_docs.append(cleaned)
        logger.info(f"Cleaned {len(cleaned_docs)}/{len(documents)} documents successfully.")
        return cleaned_docs
