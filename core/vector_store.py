"""Индексация юридических материалов и семантический поиск в ChromaDB."""

from hashlib import sha256
from pathlib import Path
from typing import TypedDict

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError

from config import CHROMA_PERSIST_DIR, PRECEDENTS_DIR, TEMPLATES_DIR, load_settings
from core.parser import extract_text_from_file

_TEXT_FORMATS = {".txt", ".md"}
_DOCUMENT_FORMATS = {".pdf", ".docx"}
_TEMPLATES_COLLECTION = "contract_templates"
_PRECEDENTS_COLLECTION = "legal_precedents"


class SearchResult(TypedDict):
    """Фрагмент базы знаний и его близость к запросу."""

    text: str
    source: str
    doc_type: str
    category: str
    distance: float
    score: float


class LegalVectorStore:
    """Хранить шаблоны договоров и обзоры юридических рисков."""

    def __init__(self) -> None:
        """Открыть локальную базу и настроить модель эмбеддингов OpenAI."""
        settings = load_settings()
        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key=settings.openai_api_key
        )
        self._client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        self._templates = self._open_collection(_TEMPLATES_COLLECTION)
        self._precedents = self._open_collection(_PRECEDENTS_COLLECTION)
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    @staticmethod
    def knowledge_base_counts() -> dict[str, int]:
        """Прочитать число фрагментов в существующей базе без ключа OpenAI."""
        counts = {_TEMPLATES_COLLECTION: 0, _PRECEDENTS_COLLECTION: 0}
        if not (CHROMA_PERSIST_DIR / "chroma.sqlite3").is_file():
            return counts
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        for name in counts:
            try:
                counts[name] = client.get_collection(name, embedding_function=None).count()
            except chromadb.errors.NotFoundError:
                continue
        return counts

    def index_knowledge_base(self, force_reindex: bool = False) -> dict[str, int]:
        """Вернуть число записанных фрагментов по коллекциям, включая 0 при пропуске."""
        indexed: dict[str, int] = {}
        for collection, directory, doc_type, prefix in (
            (self._templates, TEMPLATES_DIR, "template", "template_"),
            (self._precedents, PRECEDENTS_DIR, "precedent", "precedent_"),
        ):
            indexed[collection.name] = 0
            if collection.count() and not force_reindex:
                continue
            if not directory.is_dir():
                raise ValueError(f"Папка базы знаний не найдена: {directory}")

            documents: list[str] = []
            ids: list[str] = []
            metadata: list[dict[str, str]] = []
            for path in sorted(directory.rglob("*")):
                extension = path.suffix.lower()
                if not path.is_file() or extension not in _TEXT_FORMATS | _DOCUMENT_FORMATS:
                    continue
                source = path.relative_to(directory).as_posix()
                try:
                    text = (
                        path.read_text(encoding="utf-8-sig")
                        if extension in _TEXT_FORMATS
                        else extract_text_from_file(path, path.name)
                    )
                    if not text.strip():
                        raise ValueError("Документ пуст.")
                except (OSError, ValueError) as exc:
                    raise ValueError(f"Ошибка чтения {collection.name}/{source}: {exc}") from exc

                category = path.stem.removeprefix(prefix)
                for index, chunk in enumerate(self._splitter.split_text(text)):
                    digest = sha256(f"{source}\0{index}\0{chunk}".encode()).hexdigest()
                    ids.append(digest)
                    documents.append(chunk)
                    metadata.append({"source": source, "doc_type": doc_type, "category": category})

            # Получаем эмбеддинги до записи, чтобы сбой API сохранил прежний индекс.
            embeddings = self._embed(documents) if documents else []
            old_ids = set(collection.get(include=[])["ids"]) if force_reindex else set()
            batch_size = self._client.get_max_batch_size()
            try:
                for start in range(0, len(ids), batch_size):
                    batch = slice(start, start + batch_size)
                    collection.upsert(
                        ids=ids[batch],
                        documents=documents[batch],
                        embeddings=embeddings[batch],
                        metadatas=metadata[batch],
                    )
            except chromadb.errors.ChromaError:
                self._delete_ids(collection, set(ids) - old_ids, batch_size)
                raise
            self._delete_ids(collection, old_ids - set(ids), batch_size)
            indexed[collection.name] = len(ids)
        return indexed

    def search_precedents(self, query_text: str, top_k: int = 4) -> list[SearchResult]:
        """Найти релевантные фрагменты о рисках и судебной практике."""
        return self._search(self._precedents, query_text, top_k)

    def search_templates(self, query_text: str, top_k: int = 3) -> list[SearchResult]:
        """Найти релевантные формулировки из эталонных договоров."""
        return self._search(self._templates, query_text, top_k)

    def _open_collection(self, name: str) -> Collection:
        """Открыть коллекцию с косинусным расстоянием и внешними эмбеддингами."""
        return self._client.get_or_create_collection(
            name=name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
        )

    @staticmethod
    def _delete_ids(collection: Collection, ids: set[str], batch_size: int) -> None:
        """Удалить записи с учётом ограничения Chroma на размер пакета."""
        ordered_ids = sorted(ids)
        for start in range(0, len(ordered_ids), batch_size):
            collection.delete(ids=ordered_ids[start : start + batch_size])

    def _embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """Получить эмбеддинги и перевести ошибки OpenAI в понятные сообщения."""
        try:
            if query:
                return [self._embeddings.embed_query(texts[0])]
            return self._embeddings.embed_documents(texts)
        except AuthenticationError:
            raise RuntimeError("OpenAI отклонил ключ. Проверьте OPENAI_API_KEY.") from None
        except RateLimitError:
            raise RuntimeError("Достигнут лимит OpenAI. Проверьте квоту или повторите позже.") from None
        except APIConnectionError:
            raise RuntimeError("Не удалось подключиться к OpenAI. Проверьте соединение.") from None
        except APIError as exc:
            status = getattr(exc, "status_code", None)
            detail = f" (HTTP {status})" if status is not None else ""
            raise RuntimeError(f"OpenAI не смог построить эмбеддинги{detail}.") from None

    def _search(self, collection: Collection, query_text: str, top_k: int) -> list[SearchResult]:
        """Вернуть результаты в порядке возрастания косинусного расстояния."""
        query_text = query_text.strip()
        if not query_text:
            raise ValueError("Введите текст поискового запроса.")
        if top_k < 1:
            raise ValueError("Количество результатов должно быть положительным.")
        count = min(top_k, collection.count())
        if not count:
            return []

        matches = collection.query(
            query_embeddings=self._embed([query_text], query=True),
            n_results=count,
            include=["documents", "metadatas", "distances"],
        )
        results: list[SearchResult] = []
        for text, meta, distance in zip(
            (matches["documents"] or [[]])[0],
            (matches["metadatas"] or [[]])[0],
            (matches["distances"] or [[]])[0],
            strict=True,
        ):
            fields = meta or {}
            results.append(
                {
                    "text": text or "",
                    "source": str(fields.get("source", "")),
                    "doc_type": str(fields.get("doc_type", "")),
                    "category": str(fields.get("category", "")),
                    "distance": distance,
                    "score": 1.0 - distance,
                }
            )
        return results
