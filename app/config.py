"""環境変数の読み込み。

.env を読み込んだうえで Settings に束ねる。APIキーはここ以外で参照しない。
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# ChatAnthropic / TypeSafeClient も環境変数からキーを読むため、先に os.environ へ反映する
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Claude（生成）---
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5-5"

    # --- Jev（判断）---
    typesafe_api_key: str = ""
    jev_model: str = "jev-latest"
    grade_threshold: float = 0.6
    verify_threshold: float = 0.7
    verify_enabled: bool = True

    # --- 検索・取り込み ---
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    retrieve_k: int = 8
    expand_to_page: bool = True  # 取得チャンクを同じページ全体に広げる（スキャン資料の表対策）
    chunk_size: int = 400
    chunk_overlap: int = 80

    # --- パス ---
    docs_dir: Path = Path("docs")
    sample_docs_dir: Path = Path("sample_docs")
    chroma_dir: Path = Path("data/chroma")
    fastembed_cache_path: Path = Path("data/fastembed_cache")

    @property
    def use_jev(self) -> bool:
        """Jev のキーがあるときだけ Jev を使う（無ければ Claude が判定を代行）。"""
        return bool(self.typesafe_api_key)


settings = Settings()
