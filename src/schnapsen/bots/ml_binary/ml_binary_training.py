from typing import Optional, Literal
import pathlib
import time
import joblib
import ast


def train_ML_model(
    replay_memory_location: Optional[pathlib.Path],
    model_location: Optional[pathlib.Path],
    model_class: Literal["NN", "LR"] = "LR",
) -> None:
    """
    Train the ML model for the MLPlayingBot based on replay memory stored by the MLDataBot.

    This repo supports two replay formats:
      1) binary pipeline: "<features> || <targets>" where each side can be a Python list literal or CSV.
      2) legacy tests:   "<comma-separated-features>  <label>" (two+ spaces), where label is a single int.

    The LR version trains a MultiOutputRegressor(LinearRegression), matching the playing bot's regression-style usage.
    """

    if replay_memory_location is None:
        replay_memory_location = pathlib.Path('ML_replay_memories') / 'test_replay_memory'
    if model_location is None:
        model_location = pathlib.Path("ML_models") / 'test_model'
    assert model_class in ("NN", "LR"), "Unknown model class"

    if not replay_memory_location.exists():
        raise ValueError(f"Dataset was not found at: {replay_memory_location} !")
    if model_location.exists():
        raise ValueError(
            f"Model at {model_location} exists already and overwrite is set to False. \nNo new model will be trained, process terminates"
        )

    model_location.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.multioutput import MultiOutputRegressor

    X: list[list[int]] = []
    y: list[list[int]] = []

    def parse_part(part: str) -> list[int]:
        s = part.strip()
        if not s:
            return []
        if s[0] == '[':
            parsed = ast.literal_eval(s)
            return [int(v) for v in parsed]
        return [int(v.strip()) for v in s.split(',') if v.strip() != '']

    with open(replay_memory_location, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "||" in line:
                parts = line.split('||')
                if len(parts) < 2:
                    continue
                feature_string = parts[0]
                target_string = '||'.join(parts[1:])
                features = parse_part(feature_string)
                targets = parse_part(target_string)
            else:
                # Legacy format used in tests: "f1,f2,...  label" (two+ spaces)
                parts = line.split()
                if len(parts) < 2:
                    continue
                features = parse_part(parts[0])
                targets = [int(parts[1])]

            if features and targets:
                X.append(features)
                y.append(targets)

    X_arr = np.asarray(X, dtype=np.float32)
    Y_arr = np.asarray(y, dtype=np.float32)

    if model_class == 'LR':
        print("Training a Simple (Linear Logistic Regression) model")
        learner = MultiOutputRegressor(LinearRegression())
    else:
        raise AssertionError("Unknown model class")

    start = time.time()
    print("Starting training phase...")

    model = learner.fit(X_arr, Y_arr)
    joblib.dump(model, model_location)
    end = time.time()
    print('The model was trained in ', (end - start) / 60, 'minutes.')
