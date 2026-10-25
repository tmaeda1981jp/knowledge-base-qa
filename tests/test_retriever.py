"""ハイブリッド検索のテスト（Chroma・埋め込みは差し替える）。"""

from langchain_core.documents import Document

from app import retriever
from app.retriever import expand_to_pages, hybrid_search, keyword_search, normalize_kana, tokenize


def doc(text, source="a.pdf", page=1, chunk=0):
    return Document(page_content=text, metadata={"source": source, "page": page, "chunk": chunk})


def test_tokenize_splits_words_and_japanese_bigrams():
    tokens = tokenize("Diageo が所有する ディアジオ 2024")
    assert "diageo" in tokens and "2024" in tokens
    # 濁点は落として照合する（ディ→テイ、ジオ→シオ）
    assert "ティ" in tokens and "アシ" in tokens and "シオ" in tokens
    assert "か所" in tokens  # 日本語は 2-gram（濁点は落ちる）
    assert tokenize("Ｄｉａｇｅｏ") == ["diageo"]  # 全角→半角・小文字に正規化
    assert tokenize("酒") == ["酒"]  # 1文字は 1-gram


def test_keyword_search_finds_literal_katakana(monkeypatch):
    docs = [
        doc("蒸留所一覧 Ardross アードロス グレンウィヴィス", chunk=0),
        doc("Tamdhu タムデュー 所有者 ディアジオ 稼働中", chunk=1),
        doc("Glen Moray グレンマレイ ラ・マルティニケーズ", chunk=2),
        doc("Old Parr オールドパー Diageo", chunk=3),
    ]
    monkeypatch.setattr(
        retriever,
        "_build_index",
        lambda: retriever._Index(
            docs=docs, bm25=retriever.BM25Okapi([tokenize(d.page_content) for d in docs]), size=4
        ),
    )
    monkeypatch.setattr(retriever.vectorstore, "count_chunks", lambda: 4)
    retriever.invalidate_index()

    assert keyword_search("ディアジオ", k=3)[0].metadata["chunk"] == 1
    assert keyword_search("diageo", k=3)[0].metadata["chunk"] == 3
    assert keyword_search("存在しない語", k=3) == []  # スコア 0 は返さない


def test_hybrid_fuses_vector_and_keyword_ranks(monkeypatch):
    vec = [doc("v1", chunk=1), doc("shared", chunk=9), doc("v3", chunk=3)]
    kw = [doc("shared", chunk=9), doc("k2", chunk=2)]
    monkeypatch.setattr(retriever.vectorstore, "search", lambda q, n: vec)
    monkeypatch.setattr(retriever, "keyword_search", lambda q, n: kw)
    monkeypatch.setattr(retriever.settings, "expand_to_page", False)

    result = hybrid_search("q", k=3)

    # 両方の検索に出た chunk=9 が最上位。重複は1件にまとまる
    assert [d.metadata["chunk"] for d in result] == [9, 1, 2]
    assert len(result) == 3


def test_kana_voicing_is_ignored_for_ocr_noise():
    assert normalize_kana("ペルノリカール") == normalize_kana("ベルノリカール") == "ヘルノリカール"
    assert tokenize("ペルノリカール") == tokenize("ベルノリカール")


def test_expand_to_pages_merges_chunks_of_same_page(monkeypatch):
    all_docs = [
        doc("A" * 30 + "B" * 20, page=1, chunk=0),
        doc("B" * 20 + "C" * 30, page=1, chunk=1),  # 先頭 20 字が前チャンクと重なる
        doc("D" * 50, page=2, chunk=2),
        doc("E" * 50, source="memo.md", page=None, chunk=0),
        doc("F" * 50, source="memo.md", page=None, chunk=1),
        doc("G" * 50, source="memo.md", page=None, chunk=2),
    ]
    monkeypatch.setattr(
        retriever,
        "_build_index",
        lambda: retriever._Index(
            docs=all_docs,
            bm25=retriever.BM25Okapi([tokenize(d.page_content) for d in all_docs]),
            size=6,
        ),
    )
    monkeypatch.setattr(retriever.vectorstore, "count_chunks", lambda: 6)
    monkeypatch.setattr(retriever.settings, "chunk_overlap", 20)
    retriever.invalidate_index()

    result = expand_to_pages([all_docs[1], all_docs[0], all_docs[5]])

    # 同じページの2チャンクは1件に統合され、重なりは除かれる
    assert len(result) == 2
    assert result[0].page_content == "A" * 30 + "B" * 20 + "C" * 30
    assert result[0].metadata["page"] == 1 and result[0].metadata["chunk_span"] == [0, 1]
    # ページを持たない資料は前後1チャンク（ここでは chunk 1,2）を含む
    assert result[1].page_content == "F" * 50 + "\n" + "G" * 50
