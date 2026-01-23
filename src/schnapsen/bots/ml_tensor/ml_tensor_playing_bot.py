from __future__ import annotations

from typing import Optional
import pathlib

import joblib

from schnapsen.game import Bot, Move, PlayerPerspective, SchnapsenDeckGenerator

from schnapsen.bots.ml_binary.ml_binary_helpers import _move_to_action_index
from schnapsen.bots.ml_tensor.ml_tensor_helpers import get_state_feature_tensor


class MLTensorPlayingBot(Bot):
    """Tensor-based playing bot.

    Mirrors `MLPlayingBot` (ml_binary):
    - compute X from perspective (but as tensor -> flattened)
    - model predicts 22 scores
    - pick best legal action among `perspective.valid_moves()`

    The final returned value is a normal `Move` object, identical to the binary pipeline.
    """

    def __init__(self, model_location: pathlib.Path, name: Optional[str] = None) -> None:
        super().__init__(name)
        assert model_location.exists(), f"Model could not be found at: {model_location}"
        self.__model = joblib.load(model_location)

        self.__expected_n_features: Optional[int] = None
        if hasattr(self.__model, "n_features_in_"):
            self.__expected_n_features = int(getattr(self.__model, "n_features_in_"))

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        import numpy as np

        x_tensor = get_state_feature_tensor(perspective, leader_move=leader_move)
        x_arr = np.asarray(x_tensor, dtype=np.float32).reshape(-1)

        if self.__expected_n_features is not None and x_arr.shape[0] != self.__expected_n_features:
            raise ValueError(
                f"State tensor flattened length {x_arr.shape[0]} does not match model expectation {self.__expected_n_features}."
            )

        valid_moves = perspective.valid_moves()
        assert valid_moves, "No valid moves available"

        deck = list(SchnapsenDeckGenerator().get_initial_deck())

        y_hat = self.__model.predict(np.asarray([x_arr], dtype=np.float32))
        y_arr = np.asarray(y_hat, dtype=np.float32)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(1, -1)
        if y_arr.ndim != 2 or y_arr.shape[0] != 1:
            raise ValueError(f"Unexpected model prediction shape: {y_arr.shape}")

        scores = y_arr[0].reshape(-1)
        if scores.shape[0] < 22:
            raise ValueError(f"Model returned {scores.shape[0]} outputs; expected at least 22")

        move_scores = []
        for mv in valid_moves:
            idx = _move_to_action_index(mv, deck)
            move_scores.append(float(scores[idx]))

        best = int(np.argmax(np.asarray(move_scores, dtype=np.float32)))
        return valid_moves[best]
