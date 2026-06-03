# Project Spec: Notebook-first Local LLMOps Demo

Build a complete notebook-based LLMOps demo repository that runs locally in JupyterLab first, then transfers trained LoRA adapters and serving configuration to an HPE MLIS + vLLM server for a live demo.

## Architecture

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Fine-tuning: LoRA using PEFT
- Adapters:
  - `finance`
  - `legal`
  - `healthcare`
- Experiment tracking and registry: MLflow
- Artifact storage for local services: MinIO
- Serving: vLLM with OpenAI-compatible APIs and `--enable-lora`
- Gateway: FastAPI adapter routing by domain

## Import Safety

Training data must live under `training_data/`, not `datasets/`, to avoid shadowing the Hugging Face `datasets` package. Hugging Face imports must remain unchanged:

```python
from datasets import load_dataset
from datasets import Dataset
```

## Notebook Requirements

Create the project primarily as Jupyter notebooks under `notebooks/`:

1. `01_generate_datasets.ipynb`: generate synthetic training data and save to `training_data/`.
2. `02_train_finance_lora.ipynb`: train finance LoRA adapter and save to `adapters/finance/`.
3. `03_train_legal_lora.ipynb`: train legal LoRA adapter and save to `adapters/legal/`.
4. `04_train_healthcare_lora.ipynb`: train healthcare LoRA adapter and save to `adapters/healthcare/`.
5. `05_mlflow_tracking.ipynb`: track runs, register adapters, and compare experiments.
6. `06_start_vllm.ipynb`: show commands for starting vLLM with LoRA enabled.
7. `07_load_adapters.ipynb`: dynamically load adapters into vLLM and verify model registration.
8. `08_fastapi_gateway.ipynb`: build and run adapter routing through FastAPI.
9. `09_test_inference.ipynb`: test all adapters, compare outputs, and demonstrate specialization.
10. `10_end_to_end_demo.ipynb`: run the full workflow and dynamic adapter switching demo.

Each notebook must use:

```python
from training.config import DEFAULT_CONFIG
```

The default notebook config is:

```python
DEFAULT_CONFIG = {
    "data_dir": "../training_data",
    "output_dir": "../adapters",
    "experiment_name": "llmops-demo",
}
```

## Additional Requirements

1. Add `requirements.txt`.
2. Add `docker-compose.yml` for `mlflow`, `minio`, and `vllm`.
3. Add Makefile targets:
   - `make up`
   - `make notebooks`
   - `make serve`
4. Keep adapters as standalone PEFT adapters.
5. Do not merge adapters into the base model.
6. Use OpenAI-compatible vLLM APIs.
7. Add README covering local notebook workflow, MLflow setup, MLIS migration, and serving architecture.

## Repository Structure

```text
training_data/
  finance.json
  legal.json
  healthcare.json
training/
  config.py
  generate_synthetic.py
  train_lora.py
  register_mlflow.py
adapters/
notebooks/
serving/
evaluation/
docker/
```

The final repository must be runnable locally in JupyterLab and transferable to a remote MLIS/vLLM environment with minimal changes.
