install:
	pip install -r requirements.txt

dev:
	uvicorn acfr.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	bash scripts/lint.sh

check:
	ruff check .
	ruff format --check .

format:
	ruff format .

ingest-sample:
	python scripts/ingest_sample.py
