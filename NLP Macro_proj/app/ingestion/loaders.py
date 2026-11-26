"""
Document loader module for medical guidelines and clinical literature.
Supports PDF, TXT, and Markdown formats with rich metadata preservation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import uuid

from app.utils.logging_config import setup_logger

logger = setup_logger("ingestion.loaders")


@dataclass
class Document:
    """Represents an extracted document or document page with metadata."""
    text: str
    metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)

    @property
    def document_name(self) -> str:
        return str(self.metadata.get("document_name", "unknown_document"))

    @property
    def page_number(self) -> int:
        return int(self.metadata.get("page_number", 1))

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "Medical Knowledge Base"))

    @property
    def document_id(self) -> str:
        return str(self.metadata.get("document_id", ""))


class DocumentLoader:
    """
    Robust multi-format document loader for medical knowledge sources.
    Extracts text per page (PDF) or section (Markdown/TXT) preserving metadata.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

    def __init__(self, raw_dir: Optional[Union[str, Path]] = None):
        self.raw_dir = Path(raw_dir) if raw_dir else None

    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load and parse a single document file.

        Args:
            file_path: Path to the target document.

        Returns:
            List of Document objects representing pages or sections.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            return []

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file format '{ext}' for file: {path.name}")
            return []

        doc_id = f"doc_{uuid.uuid5(uuid.NAMESPACE_DNS, path.name).hex[:8]}"

        try:
            if ext == ".pdf":
                return self._load_pdf(path, doc_id)
            elif ext in {".txt", ".md", ".markdown"}:
                return self._load_text_file(path, doc_id)
            else:
                return []
        except Exception as e:
            logger.error(f"Unexpected error loading file {path.name}: {e}", exc_info=True)
            return []

    def _load_pdf(self, path: Path, doc_id: str) -> List[Document]:
        """Extract text from PDF page-by-page using pypdf with fitz fallback."""
        docs: List[Document] = []

        # Attempt 1: pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            total_pages = len(reader.pages)
            for idx, page in enumerate(reader.pages):
                page_num = idx + 1
                try:
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append(Document(
                            text=text,
                            metadata={
                                "document_name": path.name,
                                "source": f"{path.name} (Page {page_num})",
                                "page_number": page_num,
                                "total_pages": total_pages,
                                "document_id": doc_id,
                                "file_type": "pdf",
                            }
                        ))
                except Exception as page_err:
                    logger.warning(f"Error extracting page {page_num} from {path.name}: {page_err}")
            if docs:
                logger.info(f"Loaded {len(docs)} pages from PDF: {path.name} via pypdf")
                return docs
        except Exception as pdf_err:
            logger.warning(f"pypdf failed for {path.name}, trying PyMuPDF fallback: {pdf_err}")

        # Attempt 2: PyMuPDF (fitz) fallback
        try:
            import fitz
            doc = fitz.open(str(path))
            total_pages = len(doc)
            for idx, page in enumerate(doc):
                page_num = idx + 1
                text = page.get_text() or ""
                if text.strip():
                    docs.append(Document(
                        text=text,
                        metadata={
                            "document_name": path.name,
                            "source": f"{path.name} (Page {page_num})",
                            "page_number": page_num,
                            "total_pages": total_pages,
                            "document_id": doc_id,
                            "file_type": "pdf",
                        }
                    ))
            doc.close()
            logger.info(f"Loaded {len(docs)} pages from PDF: {path.name} via PyMuPDF")
            return docs
        except Exception as fitz_err:
            logger.error(f"Failed to extract PDF {path.name} using all engines: {fitz_err}")
            return []

    def _load_text_file(self, path: Path, doc_id: str) -> List[Document]:
        """Extract text from TXT or Markdown file."""
        encodings = ["utf-8", "latin-1", "cp1252"]
        text_content = ""

        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    text_content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if not text_content.strip():
            logger.warning(f"Empty or unreadable text file: {path.name}")
            return []

        doc = Document(
            text=text_content,
            metadata={
                "document_name": path.name,
                "source": path.name,
                "page_number": 1,
                "total_pages": 1,
                "document_id": doc_id,
                "file_type": path.suffix.lstrip(".").lower(),
            }
        )
        logger.info(f"Loaded text document: {path.name} ({len(text_content)} chars)")
        return [doc]

    def load_text(
        self,
        text: str,
        document_name: str = "custom_clinical_note.txt",
        source: str = "Direct Note Entry"
    ) -> List[Document]:
        """
        Load direct text content as a Document.

        Args:
            text: Raw clinical or guideline text.
            document_name: Identifier or title for the document.
            source: Origin or source description.

        Returns:
            List with a single Document object.
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to load_text.")
            return []

        doc_id = f"doc_{uuid.uuid5(uuid.NAMESPACE_DNS, document_name).hex[:8]}"
        doc = Document(
            text=text.strip(),
            metadata={
                "document_name": document_name,
                "source": source or document_name,
                "page_number": 1,
                "total_pages": 1,
                "document_id": doc_id,
                "file_type": Path(document_name).suffix.lstrip(".").lower() or "txt",
            }
        )
        logger.info(f"Loaded direct text document: {document_name} ({len(text)} chars)")
        return [doc]

    def load_bytes(self, data: bytes, filename: str) -> List[Document]:
        """
        Load document from in-memory byte buffer (e.g. from file uploader).

        Args:
            data: Raw file bytes.
            filename: Original file name.

        Returns:
            List of Document objects extracted from the byte stream.
        """
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file format '{ext}' for uploaded file: {filename}")
            return []

        doc_id = f"doc_{uuid.uuid5(uuid.NAMESPACE_DNS, filename).hex[:8]}"

        if ext in {".txt", ".md", ".markdown"}:
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    text = data.decode(enc)
                    return self.load_text(text=text, document_name=filename, source=filename)
                except UnicodeDecodeError:
                    continue
            return []

        elif ext == ".pdf":
            import io
            docs: List[Document] = []

            # Attempt 1: pypdf
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(data))
                total_pages = len(reader.pages)
                for idx, page in enumerate(reader.pages):
                    page_num = idx + 1
                    try:
                        text = page.extract_text() or ""
                        if text.strip():
                            docs.append(Document(
                                text=text,
                                metadata={
                                    "document_name": filename,
                                    "source": f"{filename} (Page {page_num})",
                                    "page_number": page_num,
                                    "total_pages": total_pages,
                                    "document_id": doc_id,
                                    "file_type": "pdf",
                                }
                            ))
                    except Exception as page_err:
                        logger.warning(f"Error extracting page {page_num} from {filename} via pypdf: {page_err}")
                if docs:
                    logger.info(f"Loaded {len(docs)} pages from uploaded PDF: {filename} via pypdf")
                    return docs
            except Exception as pdf_err:
                logger.warning(f"pypdf in-memory extraction failed for {filename}, trying PyMuPDF fallback: {pdf_err}")

            # Attempt 2: PyMuPDF (fitz) fallback
            try:
                import fitz
                pdf_stream = io.BytesIO(data)
                doc = fitz.open(stream=pdf_stream, filetype="pdf")
                total_pages = len(doc)
                for idx, page in enumerate(doc):
                    page_num = idx + 1
                    text = page.get_text() or ""
                    if text.strip():
                        docs.append(Document(
                            text=text,
                            metadata={
                                "document_name": filename,
                                "source": f"{filename} (Page {page_num})",
                                "page_number": page_num,
                                "total_pages": total_pages,
                                "document_id": doc_id,
                                "file_type": "pdf",
                            }
                        ))
                doc.close()
                if docs:
                    logger.info(f"Loaded {len(docs)} pages from uploaded PDF: {filename} via PyMuPDF")
                    return docs
            except Exception as fitz_err:
                logger.error(f"Failed to extract uploaded PDF {filename} using all engines: {fitz_err}")

            return docs

    def load_directory(self, dir_path: Optional[Union[str, Path]] = None) -> List[Document]:
        """
        Recursively scan and load all supported documents in a directory.

        Args:
            dir_path: Target directory path (defaults to self.raw_dir).

        Returns:
            List of all extracted Document objects across all files.
        """
        target_dir = Path(dir_path) if dir_path else self.raw_dir
        if not target_dir or not target_dir.exists():
            logger.error(f"Directory does not exist: {target_dir}")
            return []

        all_docs: List[Document] = []
        for file_path in target_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                docs = self.load_file(file_path)
                all_docs.extend(docs)

        logger.info(f"Successfully loaded total of {len(all_docs)} document units from {target_dir}")
        return all_docs
