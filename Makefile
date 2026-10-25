.PHONY: help up build down logs ingest reset test lint dev dev-ingest

help:
	@echo "make up          Docker で起動（http://localhost:8000）"
	@echo "make ingest      docs/（空なら sample_docs/）を取り込み（Docker）"
	@echo "make reset       取り込んだデータ（data/chroma）を削除"
	@echo "make test        テスト実行（ローカル .venv）"
	@echo "make dev         ローカルで開発サーバ起動（ホットリロード）"
	@echo "make dev-ingest  ローカルで取り込み"

# ---- Docker ----
up:
	docker compose up --build -d
	@echo "→ http://localhost:8000"

build:
	docker compose build

down:
	docker compose down

logs:
	docker compose logs -f app

ingest:
	docker compose run --rm app python -m app.ingest

reset:
	docker compose down
	rm -rf data/chroma
	@echo "data/chroma を削除しました。make up && make ingest で取り込み直してください"

# ---- ローカル開発（uv）----
dev:
	uv run uvicorn app.main:app --reload --port 8000

dev-ingest:
	uv run python -m app.ingest

test:
	uv run pytest

lint:
	uv run ruff check app tests
