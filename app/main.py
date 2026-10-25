"""FastAPI: チャット API と画面（静的HTML）の配信。"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app import graph as graph_module
from app import ingest as ingest_module
from app import vectorstore
from app.config import settings
from app.loaders import SUPPORTED_SUFFIXES
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentsResponse,
    HealthResponse,
    IngestedFile,
    IngestResponse,
    UploadResponse,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(
    title="ナレッジベース Q&A（RAG）",
    description=(
        "LangChain + LangGraph + Claude + Jev。"
        "資料に根拠がある範囲で答え、無ければ「分かりません」と言う。"
    ),
    version="0.1.0",
)


def _judge_name() -> str:
    return "jev" if settings.use_jev else "claude"


# ---------------------------------------------------------------------------
# 画面
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        chunks=vectorstore.count_chunks(),
        judge=_judge_name(),
        model=settings.anthropic_model,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if vectorstore.count_chunks() == 0:
        raise HTTPException(
            status_code=409,
            detail=(
                "資料が取り込まれていません。"
                "先に /api/ingest を実行するかファイルをアップロードしてください。"
            ),
        )
    try:
        result = graph_module.ask(req.question.strip())
    except Exception:
        # APIキー等を含み得る例外メッセージはそのまま返さない
        logger.exception("回答の生成に失敗")
        raise HTTPException(
            status_code=502, detail="回答の生成に失敗しました。サーバのログを確認してください。"
        ) from None

    return ChatResponse(
        answer=result["answer"],
        found=result.get("found", False),
        sources=result.get("sources", []),
        rewritten_question=result.get("rewritten_question"),
        groundedness=result.get("groundedness"),
        steps=result.get("steps", []),
        judge=result.get("judge") or _judge_name(),
    )


@app.post("/api/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    source_dir = ingest_module.resolve_source_dir()
    result = ingest_module.ingest_directory(source_dir)
    return IngestResponse(
        source_dir=result.source_dir,
        files=[IngestedFile(**f) for f in result.files],
        chunks=result.total_chunks,
        warnings=result.warnings,
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile) -> UploadResponse:
    name = Path(file.filename or "").name
    if not name or Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400, detail=f"対応形式は {', '.join(sorted(SUPPORTED_SUFFIXES))} です"
        )
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="ファイルが大きすぎます（上限 20MB）")

    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.docs_dir / name
    dest.write_bytes(data)
    try:
        chunks = ingest_module.ingest_file(dest)
    except Exception:
        logger.exception("アップロード資料の取り込みに失敗: %s", name)
        raise HTTPException(status_code=422, detail="ファイルを読み取れませんでした") from None
    return UploadResponse(file=name, chunks=chunks)


@app.get("/api/documents", response_model=DocumentsResponse)
def documents() -> DocumentsResponse:
    return DocumentsResponse(documents=[IngestedFile(**d) for d in vectorstore.list_documents()])
