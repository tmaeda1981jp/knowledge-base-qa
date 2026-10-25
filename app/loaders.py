"""資料の読み込みとチャンク分割。

対応形式: PDF（ページ番号つき）/ Markdown / テキスト。
根拠表示に使うため、metadata には必ず source（ファイル名）を入れ、PDF には page も入れる。
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}

# 日本語の文章を自然な位置で切るための区切り（優先順）
JA_SEPARATORS = ["\n\n", "\n", "。", "．", "！", "？", "、", " ", ""]


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def load_file(path: Path) -> list[Document]:
    """1ファイルを Document のリストにする。PDF は1ページ=1 Document。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".md", ".markdown", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            logger.warning("空のファイルをスキップ: %s", path.name)
            return []
        return [Document(page_content=text, metadata={"source": path.name})]
    raise ValueError(f"未対応の形式です: {path.suffix}")


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    docs: list[Document] = []
    empty_pages = 0
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            empty_pages += 1
            continue
        docs.append(Document(page_content=text, metadata={"source": path.name, "page": i}))
    if empty_pages:
        logger.warning(
            "%s: 文字を取り出せないページが %d ページありました（スキャン画像の可能性）",
            path.name,
            empty_pages,
        )
    return docs


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """チャンクに分割し、ファイル内の通し番号 chunk を metadata に付ける。"""
    splitter = RecursiveCharacterTextSplitter(
        separators=JA_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = idx
    return chunks


def list_supported_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and is_supported(p))
