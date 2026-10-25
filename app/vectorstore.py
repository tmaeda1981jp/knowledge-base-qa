"""埋め込みモデル（FastEmbed・ローカル）と Chroma の初期化・追加・検索。"""

from __future__ import annotations

import logging
import os
import uuid
from collections import Counter
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge"


class FastEmbedEmbeddings(Embeddings):
    """fastembed（ONNX）を LangChain の Embeddings として使う薄いラッパー。APIキー不要。"""

    def __init__(self, model_name: str, cache_dir: str) -> None:
        # Hugging Face のキャッシュもプロジェクト配下に置く（Docker のボリュームに残すため）
        os.environ.setdefault("HF_HOME", os.path.join(cache_dir, "hf"))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from fastembed import TextEmbedding  # 起動を速くするため遅延 import

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    settings.fastembed_cache_path.mkdir(parents=True, exist_ok=True)
    logger.info("埋め込みモデルを読み込み中: %s", settings.embedding_model)
    return FastEmbedEmbeddings(settings.embedding_model, str(settings.fastembed_cache_path))


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _chunk_id(doc: Document) -> str:
    m = doc.metadata
    key = f"{m.get('source')}|{m.get('page', 0)}|{m.get('chunk', 0)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def add_chunks(chunks: list[Document]) -> int:
    """チャンクを保存する。同名ファイルの既存チャンクは先に消して差し替える。"""
    if not chunks:
        return 0
    vs = get_vectorstore()
    for source in {c.metadata["source"] for c in chunks}:
        remove_source(source)
    vs.add_documents(chunks, ids=[_chunk_id(c) for c in chunks])
    from app import retriever  # 循環 import を避けるため遅延 import

    retriever.invalidate_index()
    return len(chunks)


def remove_source(source: str) -> None:
    vs = get_vectorstore()
    existing = vs.get(where={"source": source}, include=[])
    if existing["ids"]:
        vs.delete(ids=existing["ids"])


def search(query: str, k: int | None = None) -> list[Document]:
    return get_vectorstore().similarity_search(query, k=k or settings.retrieve_k)


def count_chunks() -> int:
    return get_vectorstore()._collection.count()


def list_documents() -> list[dict]:
    data = get_vectorstore().get(include=["metadatas"])
    counter = Counter(m["source"] for m in data["metadatas"])
    return [{"file": f, "chunks": n} for f, n in sorted(counter.items())]
