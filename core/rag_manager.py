"""RAG Manager - ChromaDB-backed cold memory tier for the persona engine."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import yaml

import chromadb
from chromadb.utils import embedding_functions


class RAGManager:
    """
    ChromaDB-backed retrieval-augmented generation manager.

    Handles document ingestion (text, markdown, YAML), chunking, embedding,
    and semantic retrieval.  Used as the cold memory tier in PersonaOrchestrator:
    relevant knowledge-base chunks are automatically injected into the system
    prompt each turn based on the user's message.

    Storage is fully local — no external server required.
    """

    def __init__(
        self,
        knowledge_dir: str | Path = "knowledge",
        persist_dir: str | Path = ".chroma",
        collection_name: str = "persona_knowledge",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
    ):
        self.knowledge_dir = Path(knowledge_dir)
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._ef = embedding_functions.DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_directory(self, directory: str | Path | None = None) -> int:
        """Ingest all supported documents from *directory* (default: knowledge_dir).

        Returns the total number of chunks added to the collection.
        """
        target = Path(directory) if directory else self.knowledge_dir
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            return 0

        total = 0
        for path in sorted(target.rglob("*")):
            if path.suffix.lower() in (".txt", ".md", ".yaml", ".yml"):
                total += self.ingest_file(path)
        return total

    def ingest_file(self, file_path: str | Path) -> int:
        """Ingest a single file. Returns number of chunks added."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return 0

        text = self._load_file(path)
        if not text:
            return 0

        chunks = self._chunk_text(text, str(path))
        if not chunks:
            return 0

        # Remove stale chunks for this source before re-adding
        existing = self._collection.get(where={"source": str(path)})
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])

        self._collection.add(
            documents=[c["content"] for c in chunks],
            ids=[c["id"] for c in chunks],
            metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks],
        )
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, n_results: int | None = None) -> list[str]:
        """Return the top-k most relevant document chunks for *query*."""
        k = n_results or self.top_k
        count = self._collection.count()
        if count == 0:
            return []
        k = min(k, count)

        results = self._collection.query(query_texts=[query], n_results=k)
        return results["documents"][0] if results["documents"] else []

    def build_context_block(self, query: str, n_results: int | None = None) -> str:
        """Build a formatted context block ready for prompt injection.

        Returns an empty string when the collection has no relevant chunks.
        """
        chunks = self.retrieve(query, n_results)
        if not chunks:
            return ""
        joined = "\n---\n".join(chunks)
        return f"## Knowledge Base\n{joined}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> str:
        """Read a file and return its text content."""
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, OSError):
            return ""

        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                data = yaml.safe_load(text)
                text = self._yaml_to_text(data)
            except yaml.YAMLError:
                pass  # fall through to raw text

        return text.strip()

    def _yaml_to_text(self, data: Any, depth: int = 0) -> str:
        """Recursively flatten YAML data to human-readable text."""
        indent = "  " * depth
        if isinstance(data, dict):
            lines = []
            for k, v in data.items():
                child = self._yaml_to_text(v, depth + 1)
                lines.append(f"{indent}{k}: {child}")
            return "\n".join(lines)
        elif isinstance(data, list):
            items = [f"{indent}- {self._yaml_to_text(i, depth + 1)}" for i in data]
            return "\n" + "\n".join(items)
        else:
            return str(data) if data is not None else ""

    def _chunk_text(self, text: str, source: str) -> list[dict[str, Any]]:
        """Split *text* into overlapping word-count chunks."""
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        words = text.split()
        if not words:
            return []

        # Sanitise source path into a valid Chroma ID prefix (max 40 chars)
        source_slug = re.sub(r"[^a-z0-9]", "_", source.lower())
        source_slug = re.sub(r"_+", "_", source_slug).strip("_")[-40:]

        chunks: list[dict[str, Any]] = []
        start = 0
        idx = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(
                {
                    "content": " ".join(words[start:end]),
                    "source": source,
                    "chunk_index": idx,
                    "id": f"{source_slug}_{idx}",
                }
            )
            if end == len(words):
                break
            start += self.chunk_size - self.chunk_overlap
            idx += 1

        return chunks

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        """Number of chunks currently stored in the collection."""
        return self._collection.count()
