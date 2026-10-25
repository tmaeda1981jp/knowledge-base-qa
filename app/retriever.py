"""ハイブリッド検索: ベクトル検索（意味）＋ BM25（文字の一致）を RRF で合成する。

OCR された表形式の資料や、固有名詞のカタカナ／英語表記のように「意味」より「その文字が
書いてあるか」が重要なケースを、BM25 側が確実に拾う。日本語は形態素解析器を使わず
文字 2-gram で分割する（辞書不要・固有名詞に強い）。
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app import vectorstore
from app.config import settings

logger = logging.getLogger(__name__)

RRF_K = 60  # Reciprocal Rank Fusion の定数（一般的な既定値）
CANDIDATES_PER_SOURCE = 3  # 各検索から k の何倍を候補に取るか

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")
_CJK_RUN_RE = re.compile(r"[぀-ヿ㐀-鿿豈-﫿ｦ-ﾟ]+")


_VOICING_MARKS = {"\u3099", "\u309a"}  # 結合文字の濁点・半濁点


def normalize_kana(text: str) -> str:
    """濁点・半濁点を落とす（OCR が「ペ」を「ベ」と読む等の揺れを吸収する）。"""
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if ch not in _VOICING_MARKS)


def tokenize(text: str) -> list[str]:
    """英数字は単語、日本語は文字 2-gram（1文字の塊は 1-gram）に分ける。"""
    text = unicodedata.normalize("NFKC", text).lower()
    text = unicodedata.normalize("NFC", normalize_kana(text))
    tokens = _WORD_RE.findall(text)
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass
class _Index:
    docs: list[Document]
    bm25: BM25Okapi
    size: int  # 構築時のチャンク数（増減したら作り直す）


_index: _Index | None = None
_lock = threading.Lock()


def _build_index() -> _Index:
    data = vectorstore.get_vectorstore().get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(data["documents"], data["metadatas"], strict=True)
    ]
    logger.info("BM25 索引を構築: %d チャンク", len(docs))
    return _Index(
        docs=docs, bm25=BM25Okapi([tokenize(d.page_content) for d in docs]), size=len(docs)
    )


def get_index() -> _Index:
    global _index
    with _lock:
        if _index is None or _index.size != vectorstore.count_chunks():
            _index = _build_index()
        return _index


def invalidate_index() -> None:
    global _index
    with _lock:
        _index = None


def _key(doc: Document) -> str:
    m = doc.metadata
    return f"{m.get('source')}|{m.get('page')}|{m.get('chunk')}"


def keyword_search(query: str, k: int) -> list[Document]:
    index = get_index()
    if not index.docs:
        return []
    scores = index.bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [index.docs[i] for i in ranked[:k] if scores[i] > 0]


def _merge_overlap(acc: str, nxt: str, max_overlap: int) -> str:
    """acc の末尾と nxt の先頭の重なり（チャンクの overlap）を取り除いて連結する。"""
    for n in range(min(max_overlap, len(acc), len(nxt)), 19, -1):
        if acc.endswith(nxt[:n]):
            return acc + nxt[n:]
    return acc + "\n" + nxt


def expand_to_pages(docs: list[Document]) -> list[Document]:
    """取得チャンクを「同じファイルの同じページ全体」に広げる（順序は最初に出た位置を維持）。

    スキャン PDF の表は、結合セルの見出しと行が別チャンクに分かれることがあるため、
    ページ単位にまとめてから判定・回答に渡す。ページを持たない資料は前後 1 チャンクを足す。
    """
    index = get_index()
    by_source: dict[str, list[Document]] = {}
    for d in index.docs:
        by_source.setdefault(d.metadata.get("source", ""), []).append(d)
    for lst in by_source.values():
        lst.sort(key=lambda d: d.metadata.get("chunk", 0))

    expanded: list[Document] = []
    seen: set[tuple] = set()
    for d in docs:
        source, page, chunk = (
            d.metadata.get("source", ""),
            d.metadata.get("page"),
            d.metadata.get("chunk", 0),
        )
        if page is not None:
            group = [x for x in by_source.get(source, []) if x.metadata.get("page") == page]
            key = (source, page)
        else:
            group = [
                x for x in by_source.get(source, []) if abs(x.metadata.get("chunk", 0) - chunk) <= 1
            ]
            key = (source, "chunk", chunk)
        if key in seen:
            continue
        seen.add(key)
        if not group:
            expanded.append(d)
            continue
        text = group[0].page_content
        for x in group[1:]:
            text = _merge_overlap(text, x.page_content, settings.chunk_overlap + 40)
        meta = dict(d.metadata)
        meta["chunk_span"] = [group[0].metadata.get("chunk", 0), group[-1].metadata.get("chunk", 0)]
        expanded.append(Document(page_content=text, metadata=meta))
    return expanded


def hybrid_search(query: str, k: int | None = None) -> list[Document]:
    """ベクトル検索と BM25 の順位を RRF で合成し、上位 k 件を返す（必要ならページに展開）。"""
    k = k or settings.retrieve_k
    n = k * CANDIDATES_PER_SOURCE
    ranked_lists = [vectorstore.search(query, n), keyword_search(query, n)]

    fused: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs):
            key = _key(doc)
            by_key.setdefault(key, doc)
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)

    top = sorted(fused, key=fused.get, reverse=True)[:k]
    docs = [by_key[key] for key in top]
    return expand_to_pages(docs) if settings.expand_to_page else docs
