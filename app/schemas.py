"""API の入出力と、Claude 構造化出力のスキーマ（Pydantic）。"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 共通
# ---------------------------------------------------------------------------


class Source(BaseModel):
    """回答の根拠。画面の根拠カードにそのまま使う。"""

    file: str
    page: int | None = None
    excerpt: str
    relevance: float | None = Field(default=None, description="grade で得た関係ありの確率")


class Step(BaseModel):
    """LangGraph の通過ノードと判定内容。画面で「どう考えたか」を見せる。"""

    node: str
    detail: str


# ---------------------------------------------------------------------------
# /api/chat
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    found: bool = Field(description="資料に根拠が見つかり、裏付けも確認できたか")
    sources: list[Source] = []
    rewritten_question: str | None = None
    groundedness: float | None = Field(default=None, description="verify で得た裏付けの確率")
    steps: list[Step] = []
    judge: str = Field(description="判定に使ったモデル: jev / claude")


# ---------------------------------------------------------------------------
# 取り込み系
# ---------------------------------------------------------------------------


class IngestedFile(BaseModel):
    file: str
    chunks: int


class IngestResponse(BaseModel):
    source_dir: str
    files: list[IngestedFile]
    chunks: int
    warnings: list[str] = []


class UploadResponse(BaseModel):
    file: str
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[IngestedFile]


class HealthResponse(BaseModel):
    status: str
    chunks: int
    judge: str
    model: str


# ---------------------------------------------------------------------------
# Claude に判定を代行させるときの構造化出力
# ---------------------------------------------------------------------------


class GradeResult(BaseModel):
    scores: list[float] = Field(
        description="各チャンクが質問に答える情報を含む確率（0〜1）。チャンクと同じ順・同じ個数"
    )


class VerifyResult(BaseModel):
    score: float = Field(description="回答の主張がすべて根拠に裏付けられている確率（0〜1）")
