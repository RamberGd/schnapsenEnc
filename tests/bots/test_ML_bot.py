from unittest import TestCase
import tempfile
import pathlib
import random
from schnapsen.bots import MLDataBot, train_ML_model
from schnapsen.bots import RandBot


class MLBotTrainOnlyTest(TestCase):
    def test_train_only(self) -> None:
        # Create a temporary directory for replay memory and model output
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            replay_file = tmpdir / "replay.mem"
            model_file = tmpdir / "test_model.joblib"

            # create a simple MLDataBot wrapping a random bot (we won't play any games)
            randbot = RandBot(random.Random(42), "randbot")
            ml_databot = MLDataBot(randbot, replay_file)

            # Write a small synthetic replay-memory dataset directly to the file.
            # Each line must follow the format written by MLDataBot.notify_game_end:
            # "<comma-separated-features> || <label>\n"
            # Use fixed-length feature vectors (expected length 262) and mixed labels.
            num_features = 262
            samples = []
            for i in range(40):
                # keep values small integers; LogisticRegression can handle these as numeric features
                features = [str((i + j) % 2) for j in range(num_features)]
                label = "1" if i % 2 == 0 else "0"
                samples.append(f"{','.join(features)} || {label}\n")

            replay_file.write_text(''.join(samples))

            # Ensure the replay file exists
            assert replay_file.exists()

            # Train a logistic regression model (fast) using the synthetic dataset
            train_ML_model(replay_memory_location=replay_file, model_location=model_file, model_class='LR')

            # After training, the model file should exist
            assert model_file.exists()
