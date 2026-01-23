from __future__ import annotations

import pathlib
import time
from typing import Literal, Optional

import joblib


def train_MLTensor_model(
    replay_memory_location: Optional[pathlib.Path],
    model_location: Optional[pathlib.Path],
    model_class: Literal["MLP"] = "MLP",
) -> None:
    """Train the tensor-based model.

    Replay format per line:
        "<tensor_literal> || <target_literal>"

    X is a 4x4x5 tensor

    IMPORTANT (consistency with ml_binary):
    y is a 22-dim one-hot encoding of the *move actually played* in that state,
    but only for states coming from games the data bot won.

    We train an MLPRegressor to output 22 scores; at inference we pick the legal move with max score.
    """

    if replay_memory_location is None:
        replay_memory_location = pathlib.Path("ML_replay_memories") / "tensor_replay_memory.txt"
    if model_location is None:
        model_location = pathlib.Path("ML_models") / "tensor_model.joblib"

    replay_memory_location = pathlib.Path(replay_memory_location)
    model_location = pathlib.Path(model_location)

    if not replay_memory_location.exists():
        raise ValueError(f"Dataset was not found at: {replay_memory_location} !")
    if model_location.exists():
        raise ValueError(f"Model at {model_location} exists already. Refusing to overwrite.")

    model_location.parent.mkdir(parents=True, exist_ok=True)

    import ast
    import numpy as np
    from sklearn.neural_network import MLPRegressor

    X_list = []
    y_list = []

    with replay_memory_location.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("||")
            if len(parts) < 2:
                continue
            x_s = parts[0].strip()
            y_s = "||".join(parts[1:]).strip()

            if not x_s or not y_s:
                continue

            x = ast.literal_eval(x_s) if x_s.startswith("[") else None
            y = ast.literal_eval(y_s) if y_s.startswith("[") else None
            if x is None or y is None:
                continue

            x_arr = np.asarray(x, dtype=np.float32)
            y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

            if x_arr.shape != (4, 4, 5):
                continue
            if y_arr.shape[0] != 22:
                continue

            X_list.append(x_arr.reshape(-1))
            y_list.append(y_arr)

    if not X_list:
        raise ValueError(f"No training samples could be parsed from {replay_memory_location}")

    X = np.stack(X_list, axis=0)
    Y = np.stack(y_list, axis=0)

    if model_class != "MLP":
        raise AssertionError("Unknown model_class")

    learner = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=75,
        random_state=0,
        verbose=False,
    )

    start = time.time()
    model = learner.fit(X, Y)
    joblib.dump(model, model_location)
    end = time.time()
    print("MLTensor model trained in", (end - start) / 60, "minutes")
