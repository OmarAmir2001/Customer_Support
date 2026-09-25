import json
import os

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from customer_support.helpers.logging_config import get_logger
from customer_support.models.enums import ProcessingEnum

from .BaseController import BaseController
from .ProjectController import ProjectController

# Which heading levels start a new chunk, and the metadata key each one lands in.
# Three levels because the handbooks use all three (CS: 1 h1 / 9 h2 / 35 h3), and a
# section path built from all of them is unique across the document — which it has
# to be, since that path is the key the Section 1 re-sync deletes on.
HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# Handbook name -> department. Both halves of this mapping are load-bearing at
# retrieval time: RetrievalController ranks by `source` (handbook chunks outrank
# instructor-resolved ones) and filters by `department` (a CS student is never
# answered out of the IS handbook). Neither works unless ingestion writes them, so
# every chunk gets tagged here, at the one place chunks are created.
HANDBOOK_DEPARTMENTS = {
    "CS_2023": "CS",
    "IS_2023": "IS",
}


class ProcessController(BaseController):
    def __init__(self, project_id: str):
        super().__init__()

        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)
        self.logger = get_logger(__name__)

    def identify_handbook(self, file_id: str):
        """Map a stored file id back to (source, department).

        Stored ids carry a random prefix — `3EsFAHwA7Z8L_CS_2023.md` — and a loader's
        own `source` is the full path on disk. Either would make
        `source.startswith("CS_2023")` false, which silently drops handbook chunks
        BELOW instructor-resolved ones in the ranking. So the handbook name is
        recovered here and written as a clean `source`.
        """
        for handbook, department in HANDBOOK_DEPARTMENTS.items():
            if handbook.lower() in (file_id or "").lower():
                return handbook, department
        return None, None

    def _chunk_metadata(self, file_id: str, base: dict = None) -> dict:
        """Normalise one chunk's metadata: clean `source`, plus `department`."""
        metadata = dict(base or {})
        source, department = self.identify_handbook(file_id)

        if source is not None:
            metadata["source"] = source
            metadata["department"] = department
        else:
            # Not a recognised handbook: keep whatever source the loader gave, and
            # leave department unset so the chunk is treated as shared content.
            metadata.setdefault("source", file_id)
            metadata.setdefault("department", None)

        return metadata

    def get_file_extension(self, file_id: str):
        """
        Get the file extension of the specified file in the project directory.
        """
        return os.path.splitext(file_id)[-1]  # Returns the file extension without the dot

    def get_file_loader(self, file_id: str):
        """
        Get the appropriate file loader based on the file extension.
        """
        file_path = os.path.join(self.project_path, file_id)
        file_ext = self.get_file_extension(file_id)

        if not os.path.exists(file_path):
            return None

        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding="utf-8")
        if file_ext == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path)
        if file_ext == ProcessingEnum.MD.value:
            return TextLoader(file_path, encoding="utf-8")
        return None

    def get_file_content(self, file_id: str):
        """
        Get the content of the specified file in the project directory.
        """
        loader = self.get_file_loader(file_id=file_id)
        return loader.load() if loader else None

    def process_file_content(
        self, file_content: list, file_id: str, chunk_size: int = 1000, overlap: int = 50
    ):
        """
        Process the content of the specified file in the project directory.

        Markdown is split on its OWN headings; anything else falls back to a
        fixed-size split. A handbook's headings are semantic boundaries an author
        chose, so splitting on them keeps a rule intact instead of cutting it in
        half at character 1000 — and it is what gives each chunk a real `section`
        value, which the answer prompt cites and the Section 1 re-sync keys on.
        """
        if self.get_file_extension(file_id=file_id) == ProcessingEnum.MD.value:
            return self._chunk_markdown(
                file_content=file_content,
                file_id=file_id,
                chunk_size=chunk_size,
                overlap=overlap,
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )
        # Split the file content into chunks
        file_content_text = [doc.page_content for doc in file_content]

        # Split the file content metadata into chunks, tagged so retrieval can rank and
        # filter them. Without this the loader's `source` is a file path.
        file_content_metadata = [
            self._chunk_metadata(file_id, doc.metadata) for doc in file_content
        ]

        # Create documents from the file content and metadata
        chunks = text_splitter.create_documents(file_content_text, metadatas=file_content_metadata)

        return chunks

    def _chunk_markdown(
        self, file_content: list, file_id: str, chunk_size: int = 1000, overlap: int = 50
    ):
        """Split on headings, then sub-split only the sections that are too long.

        Headings alone are not enough: one section of a regulations handbook can run
        to several thousand characters, which would produce a chunk too large to
        retrieve usefully. Headings alone are also not too *many* — a 200-character
        heading-only section is fine, so nothing is merged upward.

        Every sub-chunk keeps its parent section label, so a section that splits into
        four parts still deletes and re-inserts as one unit.
        """
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            # Keep the heading text in the chunk body. It is the single most
            # informative line for embedding a section, and dropping it makes a
            # chunk about "withdrawal" indistinguishable from one about "transfer".
            strip_headers=False,
        )
        overflow_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )

        chunks = []
        for document in file_content:
            for section in header_splitter.split_text(document.page_content):
                metadata = self._chunk_metadata(file_id, document.metadata)
                metadata["section"] = self._section_label(section.metadata)

                if len(section.page_content) <= chunk_size:
                    chunks.append(Document(page_content=section.page_content, metadata=metadata))
                    continue

                # Oversized section: split it, and give every part the same section
                # label so the re-sync still treats them as one logical unit.
                for part in overflow_splitter.split_text(section.page_content):
                    chunks.append(Document(page_content=part, metadata=dict(metadata)))

        return chunks

    @staticmethod
    def _section_label(header_metadata: dict) -> str:
        """Build a readable, citable section path from the headings above a chunk.

        "Chapter 4 > Withdrawal > Deadlines" rather than just "Deadlines", because a
        deadline means nothing without knowing what it is a deadline for — and
        because the label is the key the re-sync deletes on, so it has to be unique
        across the document.
        """
        parts = [header_metadata.get(key) for _, key in HEADERS_TO_SPLIT_ON]
        return " > ".join(part for part in parts if part) or "(untitled)"

    def process_json_content(self, file_id: str):
        """
        Load a pre-chunked JSON handbook (list of {text, source, section}).
        Skips extraction + splitting — the chunks are already made.
        Returns LangChain Documents so the rest of the pipeline is unchanged.
        """
        file_path = os.path.join(self.project_path, file_id)
        if not os.path.exists(file_path):
            return None
        with open(file_path, encoding="utf-8") as f:
            raw_chunks = json.load(f)

        documents = []
        for i, chunk in enumerate(raw_chunks):
            if "text" not in chunk:
                self.logger.error("json_chunk_missing_text", file_id=file_id, chunk_index=i)
                return None
            documents.append(
                Document(
                    page_content=chunk["text"],
                    metadata=self._chunk_metadata(
                        chunk.get("source") or file_id,
                        {"source": chunk.get("source"), "section": chunk.get("section")},
                    ),
                )
            )
        return documents
