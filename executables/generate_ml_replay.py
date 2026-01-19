#!/usr/bin/env python3
"""
Generate a replay-memory file by playing a number of games using MLDataBot wrapping a RandBot.
This mirrors the behavior of the CLI `ml create_replay_memory_dataset` but provides a small, convenient script
for quickly producing a dataset file suitable for `train_ML_model`.

Usage examples:
  PYTHONPATH=src python3 executables/generate_ml_replay.py --games 20 --out ML_replay_memories/test_replay.mem --seed 42 --overwrite
  PYTHONPATH=src python3 executables/generate_ml_replay.py --games 1000 --out ML_replay_memories/random_random_1k.txt

Options:
  --games N        Number of games to play (default: 100)
  --out PATH       Output replay memory file (default: ML_replay_memories/random_random_100.txt)
  --seed S         Base random seed (default: 1)
  --overwrite      Overwrite existing output file if present
  --train          After generating data, train a LR model and save alongside the replay file
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
from typing import Optional

from schnapsen.bots import MLDataBot, RandBot, train_ML_model
from schnapsen.game import SchnapsenGamePlayEngine

from schnapsen.bots import BullyBot


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate ML replay-memory by playing games")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    parser.add_argument("--out", type=str, default="ML_replay_memories/random_random_100.txt",
                        help="Output replay memory file path")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    parser.add_argument("--train", action="store_true", help="Train a LR model after dataset generation")
    parser.add_argument("--model-out", type=str, default=None, help="Where to save the trained model (if --train)")
    args = parser.parse_args(argv)

    out_path = pathlib.Path(args.out)
    # ensure directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        if args.overwrite:
            print(f"Overwriting existing replay memory file at {out_path}")
            out_path.unlink()
        else:
            print(f"Appending to existing replay memory file at {out_path}")

    engine = SchnapsenGamePlayEngine()

    # create two MLDataBots wrapping random bots with deterministic seeds so dataset is reproducible
    bot1 = MLDataBot(BullyBot(random.Random(args.seed + 11), name=f"randbot_{args.seed+11}"), replay_memory_location=out_path)
    bot2 = MLDataBot(RandBot(random.Random(args.seed + 22), name=f"randbot_{args.seed+22}"), replay_memory_location=out_path)

    print(f"Starting generation of {args.games} games (this will append to {out_path})\n\n")
    for i in range(1, args.games + 1):
        # use a deterministic different seed per game
        seed = args.seed + random.randint(0, 1000)
        engine.play_game(bot1, bot2, random.Random(seed))
        if i % 100 == 0:
            print(f"Played {i}/{args.games} games")

    print(f"Dataset generation finished. Replay memory stored at: {out_path}")

    if args.train:
        model_out = pathlib.Path(args.model_out) if args.model_out is not None else out_path.parent / (out_path.stem + "_model.joblib")
        print(f"Training LR model from {out_path} and writing model to {model_out}\n\n")
        train_ML_model(replay_memory_location=out_path, model_location=model_out, model_class='LR')
        print("Training finished.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

