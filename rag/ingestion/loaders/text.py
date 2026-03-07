from pathlib import Path
from rag.core.interfaces import DocumentLoader
from rag.core.models import Document, DocumentMetadata

class TextFileLoader(DocumentLoader):
    def load(self, file_path: str) -> Document:
        file_obj = Path(file_path)
        text = file_obj.read_text(encoding="utf-8")
        return Document(
            content=text,
            metadata=DocumentMetadata(
                source=str(file_path),
                doc_id=file_obj.stem,
                format=file_obj.suffix
            )
        )