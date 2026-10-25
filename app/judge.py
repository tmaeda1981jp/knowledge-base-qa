"""判断レイヤー。

「チャンクは質問に関係あるか（grade）」「回答は根拠に裏付けられているか（verify）」を
確率（0〜1）で返す。実装は2つ:

- JevJudge   : TypeSafe AI の判断特化モデル Jev（Noul 質問）。速くて安い
- ClaudeJudge: Claude の構造化出力で同じ質問に答えさせる（Jev キー未設定時／Jev 障害時）

LangGraph のノードは Judge プロトコルだけに依存し、どちらが使われたかは `name` で知る。
"""

from __future__ import annotations

import logging
from typing import Protocol

from app import prompts
from app.config import settings
from app.schemas import GradeResult, VerifyResult

logger = logging.getLogger(__name__)


class Judge(Protocol):
    name: str

    def grade(self, question: str, chunks: list[str]) -> list[float]:
        """各チャンクが質問に答える情報を含む確率を返す（chunks と同じ順）。"""
        ...

    def verify(self, question: str, answer: str, chunks: list[str]) -> float:
        """回答が根拠チャンクに裏付けられている確率を返す。"""
        ...


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ---------------------------------------------------------------------------
# Jev
# ---------------------------------------------------------------------------


class JevJudge:
    name = "jev"

    def __init__(self, api_key: str, model: str) -> None:
        from typesafe_sdk import TypeSafeClient

        self._client = TypeSafeClient(api_key=api_key, model=model)

    def grade(self, question: str, chunks: list[str]) -> list[float]:
        from typesafe_sdk import Noul

        if not chunks:
            return []
        state = {
            "question": question,
            "chunks": [{"id": i, "text": text} for i, text in enumerate(chunks)],
        }
        # k 個の Noul 質問を1リクエストに同梱する（Jev は質問を並列に評価するので時間は増えない）
        questions = {
            f"chunk_{i}": Noul(instructions=prompts.GRADE_INSTRUCTIONS_JEV.format(idx=i))
            for i in range(len(chunks))
        }
        resp = self._client.system_one(state=state, questions=questions)
        return [_clamp(resp.answers[f"chunk_{i}"].noul) for i in range(len(chunks))]

    def verify(self, question: str, answer: str, chunks: list[str]) -> float:
        from typesafe_sdk import Noul

        state = {"question": question, "answer": answer, "evidence": chunks}
        resp = self._client.system_one(
            state=state,
            questions={"grounded": Noul(instructions=prompts.VERIFY_INSTRUCTIONS_JEV)},
        )
        return _clamp(resp.answers["grounded"].noul)


# ---------------------------------------------------------------------------
# Claude（フォールバック）
# ---------------------------------------------------------------------------


class ClaudeJudge:
    name = "claude"

    def __init__(self, model: str) -> None:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=model, max_tokens=1024)
        # json_schema 方式: Claude の構造化出力を使う（tool_choice 強制は Opus 5.5 で使えないため）
        self._grader = llm.with_structured_output(GradeResult, method="json_schema")
        self._verifier = llm.with_structured_output(VerifyResult, method="json_schema")

    def grade(self, question: str, chunks: list[str]) -> list[float]:
        if not chunks:
            return []
        body = "\n\n".join(f"[チャンク {i}]\n{text}" for i, text in enumerate(chunks))
        result: GradeResult = self._grader.invoke(
            [
                ("system", prompts.GRADE_SYSTEM),
                ("human", f"# 質問\n{question}\n\n# チャンク（{len(chunks)}個）\n{body}"),
            ]
        )
        scores = [_clamp(s) for s in result.scores][: len(chunks)]
        # 個数が足りなければ 0 で埋める（判定不能は「関係なし」扱い）
        scores += [0.0] * (len(chunks) - len(scores))
        return scores

    def verify(self, question: str, answer: str, chunks: list[str]) -> float:
        evidence = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(chunks))
        result: VerifyResult = self._verifier.invoke(
            [
                ("system", prompts.VERIFY_SYSTEM),
                ("human", f"# 質問\n{question}\n\n# 回答\n{answer}\n\n# 根拠の抜粋\n{evidence}"),
            ]
        )
        return _clamp(result.score)


# ---------------------------------------------------------------------------
# フォールバック付きラッパー
# ---------------------------------------------------------------------------


class FallbackJudge:
    """primary（Jev）で失敗したら secondary（Claude）で判定し直す。

    直近の呼び出しで実際に使った実装名を `name` に残す（steps に記録するため）。
    """

    def __init__(self, primary: Judge, secondary: Judge) -> None:
        self._primary = primary
        self._secondary = secondary
        self.name = primary.name

    def _call(self, method: str, *args):
        try:
            result = getattr(self._primary, method)(*args)
            self.name = self._primary.name
            return result
        except Exception as e:  # 新しい外部サービスなので、障害時はデモを止めず Claude に切り替える
            logger.warning(
                "%s の %s に失敗。%s で代行します: %s",
                self._primary.name,
                method,
                self._secondary.name,
                e,
            )
            result = getattr(self._secondary, method)(*args)
            self.name = f"{self._secondary.name} (fallback)"
            return result

    def grade(self, question: str, chunks: list[str]) -> list[float]:
        return self._call("grade", question, chunks)

    def verify(self, question: str, answer: str, chunks: list[str]) -> float:
        return self._call("verify", question, answer, chunks)


def build_judge() -> Judge:
    """環境変数に応じて判定器を組み立てる。"""
    claude = ClaudeJudge(settings.anthropic_model)
    if settings.use_jev:
        logger.info("判定モデル: Jev (%s) / フォールバック: Claude", settings.jev_model)
        return FallbackJudge(JevJudge(settings.typesafe_api_key, settings.jev_model), claude)
    logger.info("判定モデル: Claude (%s)（TYPESAFE_API_KEY 未設定）", settings.anthropic_model)
    return claude
