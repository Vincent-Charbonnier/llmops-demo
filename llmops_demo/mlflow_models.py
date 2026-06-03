from __future__ import annotations

import mlflow


class AdapterArtifactModel(mlflow.pyfunc.PythonModel):
    """MLflow registry wrapper that points to a standalone PEFT adapter artifact."""

    def predict(self, context, model_input):  # type: ignore[no-untyped-def]
        return {
            "adapter_path": context.artifacts.get("adapter"),
            "message": "This MLflow model entry points to a standalone PEFT LoRA adapter.",
        }

