from typing import Optional
import pathlib

import joblib

from schnapsen.game import Bot, Move, PlayerPerspective, SchnapsenDeckGenerator
from schnapsen.bots.ml_binary.ml_binary_helpers import get_state_feature_vector


class MLPlayingBot(Bot):
    """Loads a trained ML model and uses it to play.

    This bot is intended to be used with models trained by `ml_binary_training.py` on
    replay data created by `MLBinaryDataBot`.

    Training format recap (per sample):
    - X: state feature vector from `get_state_feature_vector(perspective)`
    - y: 22-dim action/trick vector (20 cards + trump-exchange + marriage)

    At inference time we:
    1) compute the same state feature vector X
    2) ask the model for a 22-dim prediction
    3) pick the *legal* move whose corresponding index has the highest predicted value
    """

    def __init__(self, model_location: pathlib.Path, name: Optional[str] = None) -> None:
        super().__init__(name)
        assert model_location.exists(), f"Model could not be found at: {model_location}"
        self.__model = joblib.load(model_location)

        # Determine how many features the loaded model expects.
        # Some wrappers (e.g., MultiOutputRegressor) store it on the first estimator.
        self.__expected_n_features: Optional[int] = None
        if hasattr(self.__model, "n_features_in_"):
            self.__expected_n_features = int(getattr(self.__model, "n_features_in_"))
        elif hasattr(self.__model, "estimators_") and getattr(self.__model, "estimators_", None):
            first = getattr(self.__model, "estimators_")[0]
            if hasattr(first, "n_features_in_"):
                self.__expected_n_features = int(getattr(first, "n_features_in_"))

    @staticmethod
    def _move_to_index(move: Move, deck: list) -> int:
        """Map a Move to the same 22-dim index that MLBinaryDataBot uses for targets."""
        if move.is_trump_exchange():
            return 20
        if move.is_marriage():
            return 21
        if move.is_regular_move():
            for i, c in enumerate(deck):
                if move.card == c:
                    return i
        raise ValueError(f"Unsupported/unmappable move: {move}")

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        import numpy as np

        state_features = get_state_feature_vector(perspective)

        # `get_state_feature_vector` historically returned a cleaned string in this repo.
        # For model input we need a list[int]. Accept both for robustness.
        if isinstance(state_features, str):
            # string looks like: "0, 1, 0, ..." (no brackets)
            state_vec = [int(v.strip()) for v in state_features.split(",") if v.strip() != ""]
        else:
            state_vec = [int(v) for v in state_features]

        if self.__expected_n_features is not None and len(state_vec) != self.__expected_n_features:
            raise ValueError(
                f"State vector length {len(state_vec)} does not match model expectation {self.__expected_n_features}. "
                "Make sure the model was trained with the same feature extractor."
            )

        valid_moves = perspective.valid_moves()
        assert valid_moves, "No valid moves available"

        deck = list(SchnapsenDeckGenerator().get_initial_deck())

        # Predict a 22-dim score vector for this state.
        y_hat = self.__model.predict(np.asarray([state_vec], dtype=np.float32))
        y_arr = np.asarray(y_hat, dtype=np.float32)

        # Normalize possible shapes to a flat 22-d vector.
        if y_arr.ndim == 3:
            # Some wrappers can return shape (1, 22, 1)
            y_arr = y_arr.reshape(y_arr.shape[0], -1)
        if y_arr.ndim != 2 or y_arr.shape[0] != 1:
            raise ValueError(f"Unexpected model prediction shape: {y_arr.shape}")

        action_scores = y_arr[0].reshape(-1)
        if action_scores.shape[0] < 22:
            raise ValueError(
                f"Model returned only {action_scores.shape[0]} outputs; expected at least 22 for action vector."
            )

        move_scores: list[float] = []
        for mv in valid_moves:
            idx = self._move_to_index(mv, deck)
            move_scores.append(float(action_scores[idx]))

        best_idx = int(np.argmax(np.asarray(move_scores, dtype=np.float32)))
        return valid_moves[best_idx]
