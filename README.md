Fine-tuning with LoRA LLMOps Demo on HPE Private Cloud AI

This repository demonstrates a notebook-driven workflow for creating, tracking, deploying, routing, and testing standalone PEFT LoRA adapters for `Qwen/Qwen2.5-7B-Instruct`.

## What It Builds

```mermaid
flowchart LR
    A[Jupyter notebooks] --> B[Synthetic domain datasets]
    B --> C[LoRA fine-tuning]
    C --> D[adapters/finance]
    C --> E[adapters/legal]
    C --> F[adapters/healthcare]
    C --> G[MLflow runs and registry entries]
    D --> H[vLLM OpenAI-compatible endpoint]
    E --> H
    F --> H
    H --> I[FastAPI gateway]
    I --> J[Qwen classifier]
    J --> K{Domain}
    K --> L[finance adapter]
    K --> M[legal adapter]
    K --> N[healthcare adapter]
    K --> O[base model]
    P[Open WebUI pipeline] --> I
```

The default routed domains are:

- `finance`
- `legal`
- `healthcare`
- `general`

`finance`, `legal`, and `healthcare` requests are routed to matching standalone PEFT LoRA adapters. `general` requests are routed to the base model.

## Repository Layout

- `notebooks/`: the main end-to-end workflow for data generation, training, MLflow registration, vLLM startup, adapter loading, and inference tests.
- `training_data/`: generated JSON chat datasets for each domain.
- `adapters/`: trained adapter output directories.
- `training/generate_synthetic.py`: deterministic synthetic dataset generation.
- `training/train_lora.py`: PEFT LoRA training and MLflow logging.
- `training/register_mlflow.py`: registration of existing local adapters in MLflow.
- `evaluation/evaluate.py`: lightweight adapter response evaluation.
- `scripts/load_adapters.py`: calls the vLLM LoRA runtime loading endpoint.
- `scripts/test_inference.py`: OpenAI-compatible smoke test against vLLM.
- `llmops_demo/settings.py`: environment-backed project configuration.
- `gateway/`: FastAPI gateway that classifies requests and forwards them to vLLM.
- `gateway/tests/`: gateway unit tests for routing behavior and API key forwarding.
- `k8s/`: Kubernetes deployment and service manifests for the gateway.
- `lora_gateway.py`: Open WebUI pipeline that sends chat requests to the gateway.

## Notebook Workflow

Run the notebooks in order:

1. `01_generate_datasets.ipynb`
2. `02_train_finance_lora.ipynb`
3. `03_train_legal_lora.ipynb`
4. `04_train_healthcare_lora.ipynb`
5. `05_mlflow_tracking.ipynb`
6. `06_start_vllm.ipynb`
7. `07_load_adapters.ipynb`
8. `08_test_inference.ipynb`
9. `09_llmops.ipynb`

### 1. Generate Datasets

`01_generate_datasets.ipynb` creates deterministic synthetic chat datasets for each domain using templates in `training/generate_synthetic.py`.

Outputs:

- `training_data/finance.json`
- `training_data/legal.json`
- `training_data/healthcare.json`

Each record contains OpenAI-style `messages` with a system prompt, user instruction, and assistant response.

### 2. Train LoRA Adapters

`02_train_finance_lora.ipynb`, `03_train_legal_lora.ipynb`, and `04_train_healthcare_lora.ipynb` each call `training.train_lora.train_adapter(...)`.

Training loads the base model, applies a PEFT LoRA configuration, formats the JSON chat records through the tokenizer chat template, and saves one adapter directory per domain.

Outputs:

- `adapters/finance/`
- `adapters/legal/`
- `adapters/healthcare/`

Training also logs parameters, tags, and adapter artifacts to MLflow.

### 3. Track and Register in MLflow

`05_mlflow_tracking.ipynb` configures MLflow, registers any local adapters that already exist, and displays recent experiment runs.

Registered model names use this pattern:

```text
qwen2_5_7b_lora_<adapter>
```

For example:

- `qwen2_5_7b_lora_finance`
- `qwen2_5_7b_lora_legal`
- `qwen2_5_7b_lora_healthcare`

The MLflow model wrapper points to the standalone adapter artifact. It does not package a merged model.

### 4. Start vLLM

`06_start_vllm.ipynb` documents the vLLM serving configuration used for LoRA runtime loading.

For MLIS or another vLLM deployment, configure vLLM environment with:

```text
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

and advanced settings with LoRA support, for example:

```text
--served-model-name base --enable-lora --max-model-len 4096
```

### 5. Load Adapters into vLLM

`07_load_adapters.ipynb` downloads adapter artifacts from MLflow, places them where the vLLM server can read them, and loads each adapter through:

```text
POST /v1/load_lora_adapter
```

The helper implementation is in `scripts/load_adapters.py`.

The loaded model names match the adapter names:

- `finance`
- `legal`
- `healthcare`

### 6. Test Direct Inference

`08_test_inference.ipynb` uses the OpenAI Python client against the vLLM OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(
    base_url=f"{settings_cfg.vllm_base_url}/v1",
    api_key=settings_cfg.vllm_api_key,
)

### 7. LLMOps

`09_llmops.ipynb` triggers vLLM adapter update when a new adapter is promoted to `prod`.

## Gateway Routing

The gateway exposes an OpenAI-compatible chat completion endpoint:

```text
POST /v1/chat/completions
```

For each request, the gateway:

1. Sends the latest user message to the base model with a classifier prompt.
2. Parses the classifier response as JSON with `domain` and `confidence`.
3. Maps the domain to a LoRA adapter.
4. Forwards the original chat request to vLLM.
5. Adds routing metadata to the response.

The routing map in `gateway/app.py` is:

```text
finance    -> finance adapter
legal      -> legal adapter
healthcare -> healthcare adapter
general    -> base model
```

Adapter-routed requests are forwarded to vLLM with:

```json
{
  "extra_body": {
    "adapter_name": "finance"
  }
}
```

The gateway response includes the normal chat completion payload plus:

- `adapter`
- `confidence`
- `routed_domain`

### Gateway Kubernetes Deployment

The gateway manifests are:

- `k8s/gateway-deployment.yaml`
- `k8s/gateway-service.yaml`

The deployment expects a `vllm-api` secret containing an `api-key` key:

```bash
kubectl create secret generic vllm-api --from-literal=api-key=<your-api-key>
kubectl apply -f k8s/gateway-deployment.yaml
kubectl apply -f k8s/gateway-service.yaml
```

Update `VLLM_URL`, `BASE_MODEL`, image name, namespace, and service exposure to match your cluster.

## Open WebUI Integration

`lora_gateway.py` is an Open WebUI pipeline for end-user chat integration.

The pipeline flow is:

```text
Open WebUI -> Gateway -> Qwen classifier -> selected LoRA adapter or base model -> response
```

The pipeline sends non-streaming chat requests to:

```text
<GATEWAY_ENDPOINT>/chat/completions
```

The default `GATEWAY_ENDPOINT` is:

```text
http://llm-gateway.project-user-vinchar.svc.cluster.local:9000/v1
```

Adjust the pipeline valve to match your Open WebUI environment. For local testing against the gateway above, use:

```text
http://localhost:9000/v1
```

When the gateway returns routing metadata, the pipeline appends the selected adapter and confidence to the assistant response.

## Configuration

Most notebook settings are read from `.env` through `llmops_demo/settings.py`. Gateway settings are read directly by `gateway/router.py` and `gateway/app.py`.

Common settings:

```text
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
DATA_DIR=training_data
ADAPTER_DIR=adapters
OUTPUT_DIR=outputs
ADAPTERS=finance,legal,healthcare

MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=llmops-lora-demo
MLFLOW_REGISTERED_MODEL_PREFIX=qwen2_5_7b_lora

TRAIN_EPOCHS=1
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=0.0002
MAX_SEQ_LENGTH=1024
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

VLLM_BASE_URL=http://localhost:8000
VLLM_URL=http://localhost:8000
VLLM_API_KEY=local-dev
VLLM_CHAT_TIMEOUT_SECONDS=120

API_BASE_URL=http://localhost:9000
```

Use `VLLM_BASE_URL` for notebook and script clients. Use `VLLM_URL` for the gateway's upstream vLLM endpoint.

## Optional Script Entry Points

The notebooks are the primary interface, but the same backend code can be called directly:

```bash
python training/generate_synthetic.py
python training/train_lora.py --adapter finance
python training/register_mlflow.py
python scripts/load_adapters.py --base-url http://localhost:8000
python scripts/test_inference.py --adapter finance
python evaluation/evaluate.py
```

These scripts are useful for smoke tests and automation, but the expected demo flow is to run the notebooks first, then deploy the gateway and Open WebUI pipeline.

## Hardware Notes

Training and vLLM serving are intended for a CUDA-capable environment.

Recommended baseline:
- Notebook: 1 vGPU with 16 GB VRAM minimum for small LoRA experiments.
- vLLM: 1 vGPU with 24 GB VRAM or more for smoother local serving.

The gateway and Open WebUI pipeline do not require a GPU. They only need network access to the vLLM endpoint.

## Expected End State

After completing the workflow, you should have:

- Three JSON training datasets under `training_data/`.
- Three PEFT LoRA adapter directories under `adapters/`.
- MLflow runs and registered adapter model entries.
- A vLLM endpoint serving the base model and the loaded `finance`, `legal`, and `healthcare` adapters.
- A FastAPI gateway exposing one OpenAI-compatible chat endpoint.
- Request classification that routes domain prompts to the matching adapter and general prompts to the base model.
- An Open WebUI pipeline that lets end users chat through the gateway.
