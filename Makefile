.PHONY: install train test serve lint docker compose clean

install:
	pip install -r requirements.txt

train:
	python -m src.train

test:
	pytest -q

lint:
	ruff check src tests

serve:
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

docker:
	python -m src.train
	docker build -f docker/Dockerfile -t fraud-scoring-service:latest .

compose:
	python -m src.train
	docker compose up --build

clean:
	rm -rf models predictions.db data/*.csv .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
