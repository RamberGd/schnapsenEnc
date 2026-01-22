from typing import Optional, Literal
import pathlib
import time
import joblib

def train_ML_model(replay_memory_location: Optional[pathlib.Path],
                   model_location: Optional[pathlib.Path],
                   model_class: Literal["NN", "LR"] = "LR"
                   ) -> None:
    """
    Train the ML model for the MLPlayingBot based on replay memory stored byt the MLDataBot.
    This implementation has the option to train a neural network model or a model based on linear regression.
    The model classes used in this implementation are not necessarily optimal.

    :param replay_memory_location: Location of the games stored by MLDataBot, default pathlib.Path('ML_replay_memories') / 'test_replay_memory'
    :param model_location: Location where the model will be stored, default pathlib.Path("ML_models") / 'test_model'
    :param model_class: The machine learning model class to be used, either 'NN' for a neural network, or 'LR' for a linear regression.
    :param overwrite: Whether to overwrite a possibly existing model.
    """
    if replay_memory_location is None:
        replay_memory_location = pathlib.Path('ML_replay_memories') / 'test_replay_memory'
    if model_location is None:
        model_location = pathlib.Path("ML_models") / 'test_model'
    assert model_class == 'NN' or model_class == 'LR', "Unknown model class"

    # check that the replay memory dataset is found at the specified location
    if not replay_memory_location.exists():
        raise ValueError(f"Dataset was not found at: {replay_memory_location} !")

    # Check if model exists already
    if model_location.exists():
        raise ValueError(
            f"Model at {model_location} exists already and overwrite is set to False. \nNo new model will be trained, process terminates")

    # check if directory exists, and if not, then create it
    model_location.parent.mkdir(parents=True, exist_ok=True)

    data: list[list[int]] = []
    targets: list[int] = []

    # Only context
    import numpy as np

    X = []
    y = []

    with open(replay_memory_location, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Support two possible replay formats:
            # 1) CSV: "f1,f2,... || t1,t2,..." (legacy MLDataBot output)
            # 2) Python list literals: "[f1, f2, ...] || [t1, t2, ...]" (ml_binary_data_bot output)
            def parse_part(part: str) -> list[int]:
                s = part.strip()
                if not s:
                    return []
                # If the part looks like a Python list literal, use ast.literal_eval
                if s[0] == '[':
                    parsed = ast.literal_eval(s)
                    # Ensure booleans become ints and numeric strings are converted
                    return [int(x) for x in parsed]
                # Otherwise assume CSV of numbers
                return [int(v.strip()) for v in s.split(',') if v.strip() != '']

            parts = line.split('||')
            if len(parts) < 2:
                # malformed line; skip
                continue
            feature_string = parts[0]
            # join the rest as target in case there are extra '||' tokens
            target_string = '||'.join(parts[1:])

            features = parse_part(feature_string)
            targets = parse_part(target_string)

            if features and targets:
                X.append(features)
                y.append(targets)

    X = np.array(X, dtype=np.float32)
    Y = np.array(y, dtype=np.float32)
    """
    print("Dataset Statistics:")
    samples_of_wins = sum(targets)
    samples_of_losses = len(targets) - samples_of_wins
    print("Samples of wins:", samples_of_wins)
    print("Samples of losses:", samples_of_losses)

    """
    # What type of model will be used depends on the value of the parameter use_neural_network

    if model_class == 'LR':
        # Train a simpler Linear Logistic Regression model
        # learn more about the model or how to use better use it by checking out its documentation
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
        print("Training a Simple (Linear Logistic Regression) model")

        # Usually there is no reason to change the hyperparameters of such a simple model but fill free to experiment:
        learner = MultiOutputRegressor(LinearRegression())
    else:
        raise AssertionError("Unknown model class")

    start = time.time()
    print("Starting training phase...")

    model = learner.fit(X, Y)
    # Save the model in a file
    joblib.dump(model, model_location)
    end = time.time()
    print('The model was trained in ', (end - start) / 60, 'minutes.')



