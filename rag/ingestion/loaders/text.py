from pathlib import Path
from rag.core.interfaces import Document, DocumentLoader

class TextFileLoader(DocumentLoader):
    def load(self, file_path: str) -> Document:
        file_obj = Path(file_path)
        text = file_obj.read_text(encoding="utf-8")
        return Document(content=text, 
                        metadata={
                            "source": str(file_path),
                            "doc_id": file_obj.stem,
                            "format": file_obj.suffix
                        })