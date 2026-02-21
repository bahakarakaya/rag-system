import pathlib
from rag.core.interfaces import Document, DocumentLoader

class TextFileLoader(DocumentLoader):
    def __init__(self, file_path: str):
        self.file_path = pathlib.Path(file_path)

    def load(self) -> Document:
        text = self.file_path.read_text(encoding="utf-8")
        yield Document(content=text, 
                        metadata={
                            "source": str(self.file_path),
                            "doc_id": self.file_path.stem,
                            "format": self.file_path.suffix
                        })


class DirectoryLoader(DocumentLoader):
    """
    Loads all .txt files from a specified directory and yields them as Document objects.
    """
    def __init__(self, directory_path: str, pattern: str = "*.txt"):
        self.directory_path = pathlib.Path(directory_path)
        self.pattern = pattern

    def load(self) -> Document:
        for file_path in self.directory_path.glob(self.pattern):
            text = file_path.read_text(encoding="utf-8")
            yield Document(content=text, 
                            metadata={
                                "source": str(file_path),
                                "doc_id": file_path.stem,
                                "format": file_path.suffix
                            })