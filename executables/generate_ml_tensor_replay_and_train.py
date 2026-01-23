#!/usr/bin/env python3
"""Generate tensor replay-memory and train a tensor model.

This is the ml_tensor analogue of `executables/generate_ml_replay.py`.

It can either:
- generate N games and train once, or
- auto-pick a reasonable dataset size by training on growing prefixes and choosing the best validation score.

Replay format:
  "<4x4x5 tensor literal> || <22-dim move mask literal>"

Usage examples:
  PYTHONPATH=src python3 executables/generate_ml_tensor_replay_and_train.py --games 2000 --overwrite
  PYTHONPATH=src python3 executables/generate_ml_tensor_replay_and_train.py --auto --max-games 2000 --step 250 --overwrite

Note: "ideal" dataset size is approximated by validation performance on the move-mask prediction task.
"""

from __future__ import annotations

import argparse
import pathlib
import random
from typing import Optional

import joblib

from schnapsen.bots import RandBot
from schnapsen.bots.bully_bot import BullyBot
from schnapsen.bots.rdeep import RdeepBot
from schnapsen.bots.ml_tensor.ml_tensor_data_bot import MLTensorDataBot
from schnapsen.bots.ml_tensor.ml_tensor_helpers import parse_replay_part
from schnapsen.game import SchnapsenGamePlayEngine


def _load_replay_as_arrays(replay_path: pathlib.Path):
    import numpy as np

    X_list = []
    y_list = []

    with replay_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("||")
            if len(parts) < 2:
                continue
            x = parse_replay_part(parts[0])
            y = parse_replay_part("||".join(parts[1:]))

            x_arr = np.asarray(x, dtype=np.float32)
            y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

            if x_arr.shape != (7, 4, 5) or y_arr.shape[0] != 22:
                continue

            X_list.append(x_arr.reshape(-1))
            y_list.append(y_arr)

    if not X_list:
        raise ValueError(f"No samples parsed from {replay_path}")

    X = np.stack(X_list, axis=0)
    Y = np.stack(y_list, axis=0)
    return X, Y


def _train_mlp(X_train, y_train, random_state: int):
    from sklearn.neural_network import MLPRegressor

    # Keep this similar to the default ml_tensor_training, but a bit more stable.
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=75,
        random_state=random_state,
        verbose=False,
    )
    return model.fit(X_train, y_train)


def _validation_score(y_true, y_pred) -> float:
    """Score predictions against multi-hot move masks.

    We use average ROC-AUC over the 22 action dimensions (macro) as a stable proxy.
    If a dimension has only one class in y_true, we skip it.
    """

    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    aucs = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if np.all(col == col[0]):
            continue
        try:
            aucs.append(float(roc_auc_score(col, y_pred[:, j])))
        except Exception:
            continue

    return float(np.mean(aucs)) if aucs else float("nan")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate ml_tensor replay memory and train a model")

    parser.add_argument(
        "--replay-out",
        type=str,
        default="ML_replay_memories/ml_tensor_replay.txt",
        help="Output replay memory file",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="ML_replay_memories/ml_tensor_model.joblib",
        help="Output model file",
    )
    parser.add_argument("--seed", type=int, default=1, help="Base RNG seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    # Simple mode:
    parser.add_argument("--games", type=int, default=None, help="Number of games to play (simple mode)")

    # Auto mode:
    parser.add_argument("--auto", action="store_true", help="Auto-select a good dataset size")
    parser.add_argument("--max-games", type=int, default=2000, help="Max games to generate (auto mode)")
    parser.add_argument("--step", type=int, default=250, help="Evaluate every N games (auto mode)")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction (auto mode)")

    args = parser.parse_args(argv)

    replay_path = pathlib.Path(args.replay_out)
    model_path = pathlib.Path(args.model_out)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        if replay_path.exists():
            replay_path.unlink()
        if model_path.exists():
            model_path.unlink()

    if args.auto and args.games is not None:
        raise ValueError("Use either --auto or --games, not both")

    # Decide how many games to generate.
    games_to_generate = args.max_games if args.auto else (args.games or 200)

    engine = SchnapsenGamePlayEngine()

    # Fast, deterministic baselines.
    # Per request: Rdeep(10,2,Random(42)) vs BullyBot(Random(42))
    # Note: Give each bot its own RNG instance so they don't interfere.
    base1 = RdeepBot(10, 2, random.Random(42), name="rdeep_10_2")
    base2 = BullyBot(random.Random(42), name="bully_42")

    bot1 = MLTensorDataBot(base1, replay_memory_location=replay_path)
    bot2 = MLTensorDataBot(base2, replay_memory_location=replay_path)

    for i in range(games_to_generate):
        engine.play_game(bot1, bot2, random.Random(args.seed + 1000 + i))

    # Load parsed samples.
    X, Y = _load_replay_as_arrays(replay_path)

    # Auto-pick a dataset size by evaluating growing prefixes.
    if args.auto:
        import numpy as np

        n = X.shape[0]
        if n < 100:
            raise ValueError(f"Too few samples ({n}) for auto selection. Increase --max-games")

        rng = np.random.RandomState(args.seed)
        idx = rng.permutation(n)
        X = X[idx]
        Y = Y[idx]

        val_n = max(1, int(n * float(args.val_frac)))
        X_val, Y_val = X[:val_n], Y[:val_n]
        X_pool, Y_pool = X[val_n:], Y[val_n:]

        # Candidate sizes in samples, not games. We step in roughly equal chunks.
        # We bound by the available pool size.
        step = max(200, int((X_pool.shape[0]) * (args.step / max(1, games_to_generate))))
        candidates = list(range(step, X_pool.shape[0] + 1, step))
        if not candidates:
            candidates = [X_pool.shape[0]]

        best_score = float("-inf")
        best_k = candidates[0]

        for k in candidates:
            model = _train_mlp(X_pool[:k], Y_pool[:k], random_state=args.seed)
            pred = model.predict(X_val)
            score = _validation_score(Y_val, pred)
            print(f"[auto] k={k:5d} samples -> val_auc={score:.4f}")
            if score == score and score > best_score:  # not NaN
                best_score = score
                best_k = k

        print(f"[auto] selected k={best_k} samples (val_auc={best_score:.4f})")

        # Train final model using pool[:best_k] + validation set.
        X_final = np.concatenate([X_val, X_pool[:best_k]], axis=0)
        Y_final = np.concatenate([Y_val, Y_pool[:best_k]], axis=0)
        final_model = _train_mlp(X_final, Y_final, random_state=args.seed)
    else:
        # Simple mode: train on everything.
        final_model = _train_mlp(X, Y, random_state=args.seed)

    joblib.dump(final_model, model_path)
    print(f"Replay saved to: {replay_path}")
    print(f"Model saved to: {model_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
