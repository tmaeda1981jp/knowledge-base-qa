"""LangGraph の定義。

START → retrieve → grade ─┬─ 関係あり ──→ generate → verify ─┬─ 裏付けあり → END
                          │                                   └─ 裏付けなし → no_answer → END
                          ├─ 関係なし & 再試行前 → rewrite → retrieve（最大1回）
                          └─ 関係なし & 再試行済 → no_answer → END

依存（LLM・判定器・検索）は build_graph() に注入する。テストではモックに差し替える。
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Callable
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph

from app import prompts
from app.config import settings
from app.judge import Judge
from app.schemas import Source, Step

logger = logging.getLogger(__name__)

MAX_RETRIES = 1  # 言い換えによる再検索は最大1回
EXCERPT_CHARS = 300

Retriever = Callable[[str, int], list[Document]]


class GraphState(TypedDict, total=False):
    question: str  # 現在の検索に使う質問（rewrite で上書き）
    original_question: str  # 利用者が入力した元の質問
    documents: list[Document]
    relevance_scores: list[float]
    is_relevant: bool
    retry_count: int
    rewritten_question: str | None
    answer: str
    is_grounded: bool
    groundedness: float | None
    sources: list[Source]
    found: bool
    judge: str
    steps: Annotated[list[Step], operator.add]  # 各ノードが追記する


def _text(msg: BaseMessage) -> str:
    """AIMessage から本文テキストだけを取り出す（thinking ブロック等を除く）。"""
    return str(msg.text).strip()


def _source_of(doc: Document, relevance: float | None) -> Source:
    content = doc.page_content.strip()
    excerpt = content if len(content) <= EXCERPT_CHARS else content[:EXCERPT_CHARS] + "…"
    return Source(
        file=doc.metadata.get("source", "unknown"),
        page=doc.metadata.get("page"),
        excerpt=excerpt,
        relevance=relevance,
    )


def _format_context(docs: list[Document]) -> str:
    items = []
    for i, d in enumerate(docs, start=1):
        page = f" p.{d.metadata['page']}" if d.metadata.get("page") else ""
        items.append(
            prompts.CONTEXT_ITEM_TEMPLATE.format(
                idx=i, source=d.metadata.get("source", "unknown"), page=page, text=d.page_content
            )
        )
    return "\n\n".join(items)


def build_graph(
    *,
    judge: Judge,
    llm: BaseChatModel,
    retriever: Retriever,
    retrieve_k: int = 4,
    grade_threshold: float = 0.6,
    verify_threshold: float = 0.7,
    verify_enabled: bool = True,
):
    # ---- ノード ------------------------------------------------------------

    def retrieve(state: GraphState) -> dict:
        docs = retriever(state["question"], retrieve_k)
        return {
            "documents": docs,
            "steps": [
                Step(node="retrieve", detail=f"「{state['question']}」で検索 → {len(docs)}件")
            ],
        }

    def grade(state: GraphState) -> dict:
        docs = state.get("documents", [])
        scores = judge.grade(state["question"], [d.page_content for d in docs]) if docs else []
        kept = [(d, s) for d, s in zip(docs, scores, strict=True) if s >= grade_threshold]
        detail = " / ".join(f"{s:.2f}" for s in scores) or "候補なし"
        verdict = f"{len(kept)}件を採用" if kept else "関係ありなし"
        return {
            "documents": [d for d, _ in kept],
            "relevance_scores": [s for _, s in kept],
            "is_relevant": bool(kept),
            "judge": judge.name,
            "steps": [
                Step(
                    node="grade",
                    detail=f"[{judge.name}] 関係あり確率 {detail}"
                    f"（閾値 {grade_threshold}）→ {verdict}",
                )
            ],
        }

    def rewrite(state: GraphState) -> dict:
        msg = llm.invoke(
            [("system", prompts.REWRITE_SYSTEM), ("human", state["original_question"])]
        )
        new_q = _text(msg) or state["original_question"]
        return {
            "question": new_q,
            "rewritten_question": new_q,
            "retry_count": state.get("retry_count", 0) + 1,
            "steps": [Step(node="rewrite", detail=f"質問を言い換え → 「{new_q}」")],
        }

    def generate(state: GraphState) -> dict:
        docs = state["documents"]
        scores = state.get("relevance_scores") or [None] * len(docs)
        msg = llm.invoke(
            [
                ("system", prompts.GENERATE_SYSTEM),
                (
                    "human",
                    prompts.GENERATE_USER_TEMPLATE.format(
                        question=state["original_question"], context=_format_context(docs)
                    ),
                ),
            ]
        )
        return {
            "answer": _text(msg),
            "sources": [_source_of(d, s) for d, s in zip(docs, scores, strict=True)],
            "steps": [Step(node="generate", detail=f"{len(docs)}件の根拠から回答を生成")],
        }

    def verify(state: GraphState) -> dict:
        if not verify_enabled:
            return {
                "is_grounded": True,
                "groundedness": None,
                "found": True,
                "steps": [Step(node="verify", detail="スキップ（VERIFY_ENABLED=false）")],
            }
        chunks = [d.page_content for d in state["documents"]]
        g = judge.verify(state["original_question"], state["answer"], chunks)
        ok = g >= verify_threshold
        return {
            "is_grounded": ok,
            "groundedness": g,
            "found": ok,
            "judge": judge.name,
            "steps": [
                Step(
                    node="verify",
                    detail=f"[{judge.name}] 裏付け確率 {g:.2f}（閾値 {verify_threshold}）→ "
                    + ("合格" if ok else "不合格"),
                )
            ],
        }

    def no_answer(state: GraphState) -> dict:
        if state.get("answer") and not state.get("is_grounded", True):
            # verify 不合格: 生成した回答は捨て、根拠候補だけ残して利用者に確認を促す
            return {
                "answer": prompts.NOT_GROUNDED_MESSAGE,
                "found": False,
                "steps": [Step(node="no_answer", detail="裏付けを確認できず回答を破棄")],
            }
        return {
            "answer": prompts.NO_ANSWER_MESSAGE,
            "sources": [],
            "found": False,
            "steps": [Step(node="no_answer", detail="関係する資料が見つからず「分かりません」")],
        }

    # ---- 分岐 --------------------------------------------------------------

    def route_after_grade(state: GraphState) -> Literal["generate", "rewrite", "no_answer"]:
        if state.get("is_relevant"):
            return "generate"
        if state.get("retry_count", 0) < MAX_RETRIES:
            return "rewrite"
        return "no_answer"

    def route_after_verify(state: GraphState) -> Literal["end", "no_answer"]:
        return "end" if state.get("is_grounded") else "no_answer"

    # ---- グラフ組み立て ------------------------------------------------------

    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)
    g.add_node("verify", verify)
    g.add_node("no_answer", no_answer)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade)
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"end": END, "no_answer": "no_answer"})
    g.add_edge("no_answer", END)
    return g.compile()


def initial_state(question: str) -> GraphState:
    return {
        "question": question,
        "original_question": question,
        "retry_count": 0,
        "rewritten_question": None,
        "steps": [],
    }


# ---------------------------------------------------------------------------
# 本番用の組み立て（遅延初期化）
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_graph():
    from langchain_anthropic import ChatAnthropic

    from app.judge import build_judge
    from app.retriever import hybrid_search

    llm = ChatAnthropic(model=settings.anthropic_model, max_tokens=4096)
    return build_graph(
        judge=build_judge(),
        llm=llm,
        retriever=hybrid_search,
        retrieve_k=settings.retrieve_k,
        grade_threshold=settings.grade_threshold,
        verify_threshold=settings.verify_threshold,
        verify_enabled=settings.verify_enabled,
    )


def ask(question: str) -> GraphState:
    """質問に答える。API と CLI の共通入口。"""
    return get_graph().invoke(initial_state(question))


if __name__ == "__main__":  # 動作確認用: python -m app.graph "質問"
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    q = " ".join(sys.argv[1:]) or "黒滝蒸留所の見学は何分かかりますか？"
    result = ask(q)
    print("\n=== 回答 ===\n" + result["answer"])
    print("\n=== 根拠 ===")
    for s in result.get("sources", []):
        print(f"- {s.file} p.{s.page} ({s.relevance}): {s.excerpt[:60]}…")
    print("\n=== 経路 ===")
    for st in result["steps"]:
        print(f"- {st.node}: {st.detail}")
