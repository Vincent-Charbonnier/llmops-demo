# HPE Private Cloud AI LoRA LLMOps Demo

This repository is a demo for training standalone PEFT LoRA adapters in JupyterLab, tracking them in MLflow, serving them with vLLM.

## Architecture

```mermaid
flowchart LR
    A[Jupyter notebooks] --> B[Synthetic training data]
    B --> C[PEFT LoRA training]
    C --> D[adapters/finance]
    C --> E[adapters/legal]
    C --> F[adapters/healthcare]
    C --> G[MLflow tracking and registry]
    G --> H[S3 artifact store]
    D --> I[vLLM OpenAI API]
    E --> I
    F --> I
    I --> K[Demo clients]
```

Base model: `Qwen/Qwen2.5-7B-Instruct`

Adapters:

- `finance`
- `legal`
- `healthcare`

The adapters stay as standalone PEFT LoRA adapters. They are not merged into the base model.

## Notebook Workflow

Run notebooks in order:

1. `notebooks/01_generate_datasets.ipynb`
2. `notebooks/02_train_finance_lora.ipynb`
3. `notebooks/03_train_legal_lora.ipynb`
4. `notebooks/04_train_healthcare_lora.ipynb`
5. `notebooks/05_mlflow_tracking.ipynb`
6. `notebooks/06_start_vllm.ipynb`
7. `notebooks/07_load_adapters.ipynb`
8. `notebooks/08_test_inference.ipynb`

Each notebook includes markdown explanations, runnable cells, architecture diagrams, example prompts, and expected outputs.

## GPU Notes

Practical local training and vLLM serving require a CUDA GPU.

Recommended baseline:
- 16 GB VRAM minimum for small quantized LoRA experiments
- 24 GB or more VRAM for smoother local serving
