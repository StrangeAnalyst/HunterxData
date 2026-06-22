"""register — log and register the assembled MarkovKernel as an MLflow pyfunc.

This is the single point where the 5 row models + per-row calibrators + state
contract are serialised into the deployable :class:`MarkovKernel` and logged as a
pyfunc model with a signature, then promoted in the model registry. Keeping this
in one place guarantees the scoring job loads exactly what training produced.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..common.action_logger import ActionLogger
from .markov_kernel import MarkovKernel, MarkovKernelPyfunc


def log_and_register_kernel(
    kernel: MarkovKernel,
    config: Mapping[str, object],
    input_example: pd.DataFrame,
    *,
    register: bool = True,
    logger: ActionLogger | None = None,
) -> str:  # pragma: no cover - exercises the MLflow runtime
    """Serialise the kernel, log it as a pyfunc, and (optionally) register it.

    Args:
        kernel: The fully assembled :class:`MarkovKernel`.
        config: Loaded config (reads the ``mlflow`` / ``scoring`` blocks).
        input_example: A small feature frame for signature inference.
        register: Whether to register the model in the MLflow registry.
        logger: Optional :class:`ActionLogger`.

    Returns:
        The logged model URI (``runs:/<run_id>/markov_kernel``).
    """
    import cloudpickle
    import mlflow
    from mlflow.models.signature import infer_signature

    logger = logger or ActionLogger("kernel.register")
    registered_name = str(config["mlflow"]["registered_model_name"])  # type: ignore[index]

    with tempfile.TemporaryDirectory() as tmp:
        artifact_path = Path(tmp) / "kernel.pkl"
        with open(artifact_path, "wb") as handle:
            cloudpickle.dump(kernel, handle)

        example_out = kernel.predict(input_example)
        signature = infer_signature(input_example, example_out)

        logged = mlflow.pyfunc.log_model(
            artifact_path="markov_kernel",
            python_model=MarkovKernelPyfunc.build(),
            artifacts={"kernel": str(artifact_path)},
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_name if register else None,
        )

    logger.info(
        "kernel.logged",
        model_uri=logged.model_uri,
        registered=register,
        registered_name=registered_name if register else None,
    )
    return logged.model_uri
