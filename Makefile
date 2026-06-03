.PHONY: up notebooks datasets train register serve load-adapters api test mlflow clean

PYTHON ?= python
ENV_FILE ?= .env
DOCKER_COMPOSE ?= docker compose

up:
	$(DOCKER_COMPOSE) --env-file $(ENV_FILE) up -d mlflow minio

notebooks:
	$(PYTHON) -m jupyter lab --ip $${JUPYTER_HOST:-0.0.0.0} --port $${JUPYTER_PORT:-8888} --ServerApp.token=$${JUPYTER_TOKEN:-local-dev}

datasets:
	$(PYTHON) training/generate_synthetic.py

train:
	$(PYTHON) training/train_lora.py

register:
	$(PYTHON) training/register_mlflow.py

serve:
	$(DOCKER_COMPOSE) --env-file $(ENV_FILE) up vllm

load-adapters:
	$(PYTHON) scripts/load_adapters.py

api:
	$(PYTHON) -m uvicorn serving.gateway:app --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8080}

test:
	$(PYTHON) scripts/test_inference.py
	$(PYTHON) evaluation/evaluate.py

mlflow:
	$(PYTHON) -m mlflow server --backend-store-uri ./mlruns --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000

clean:
	$(PYTHON) scripts/clean_generated.py
