import logging
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
}


class CvParser:
    async def parse(self, file_path: str, mime_type: str) -> str:
        fmt = SUPPORTED_MIME_TYPES.get(mime_type)
        if fmt is None:
            raise ValueError(f"Unsupported CV mime type: {mime_type}")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if fmt == "pdf":
                return self._parse_pdf(path)
            elif fmt == "docx":
                return self._parse_docx(path)
            else:
                return path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("CV parse failed for %s: %s", file_path, e)
            raise RuntimeError(f"Failed to parse CV: {e}") from e

    def _parse_pdf(self, path: Path) -> str:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    def _parse_docx(self, path: Path) -> str:
        doc = DocxDocument(str(path))
        return "\n".join(para.text for para in doc.paragraphs)


cv_parser = CvParser()
