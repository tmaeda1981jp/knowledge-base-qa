"""資料の取り込みCLI。

使い方:
  python -m app.ingest            # docs/ を取り込む（空なら sample_docs/ にフォールバック）
  python -m app.ingest path/to/x  # ファイルまたはディレクトリを指定して取り込む
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.loaders import list_supported_files, load_file, split_documents
from app.vectorstore import add_chunks

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    source_dir: str
    files: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return sum(f["chunks"] for f in self.files)


def resolve_source_dir() -> Path:
    """docs/ に対応ファイルがあればそれを、無ければ sample_docs/ を使う。"""
    if list_supported_files(settings.docs_dir):
        return settings.docs_dir
    logger.info("docs/ が空のため sample_docs/ を取り込みます")
    return settings.sample_docs_dir


def ingest_file(path: Path) -> int:
    """1ファイルを取り込み、保存したチャンク数を返す。"""
    docs = load_file(path)
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    n = add_chunks(chunks)
    logger.info("取り込み: %s → %d チャンク", path.name, n)
    return n


def ingest_directory(directory: Path) -> IngestResult:
    result = IngestResult(source_dir=str(directory))
    files = list_supported_files(directory)
    if not files:
        result.warnings.append(f"{directory} に取り込める資料（PDF/MD/TXT）がありません")
        return result
    for path in files:
        try:
            n = ingest_file(path)
            result.files.append({"file": path.name, "chunks": n})
            if n == 0:
                result.warnings.append(f"{path.name}: 文字を取り出せませんでした")
        except Exception as e:  # 1ファイルの失敗で全体を止めない
            logger.exception("取り込み失敗: %s", path.name)
            result.warnings.append(f"{path.name}: {type(e).__name__}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="資料をベクトルDBに取り込む")
    parser.add_argument("path", nargs="?", help="ファイルまたはディレクトリ（省略時は docs/）")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    target = Path(args.path) if args.path else resolve_source_dir()
    if target.is_file():
        n = ingest_file(target)
        print(f"完了: {target.name} → {n} チャンク")
        return 0

    result = ingest_directory(target)
    print(f"取り込み元: {result.source_dir}")
    for f in result.files:
        print(f"  {f['file']}: {f['chunks']} チャンク")
    for w in result.warnings:
        print(f"  [警告] {w}")
    print(f"合計: {len(result.files)} ファイル / {result.total_chunks} チャンク")
    return 0 if result.files else 1


if __name__ == "__main__":
    sys.exit(main())
