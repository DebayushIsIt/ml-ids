.PHONY: setup data test lint up

setup:
	python3.11 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

data:
	python -m ids.data.clean --raw data/raw/MachineLearningCVE --out data/processed --benign-frac 0.2

test:
	pytest -q

lint:
	ruff check src tests

up:
	docker compose up --build
