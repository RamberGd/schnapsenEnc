from __future__ import annotations

import pathlib
from typing import Optional, cast

from schnapsen.game import Bot, Move, PlayerPerspective, Trick, SchnapsenDeckGenerator

from schnapsen.bots.ml_binary.ml_binary_helpers import _move_to_action_index
from schnapsen.bots.ml_tensor.ml_tensor_helpers import append_replay_record, get_state_feature_tensor


class MLTensorDataBot(Bot):
    """Dataset writer bot for the tensor pipeline.

    IMPORTANT (consistency with ml_binary):
    - We only store samples from games this bot won.
    - The training target y is a 22-dim one-hot encoding of *the move actually played* in that
      historical state. This matches the ml_binary "what moves win games" logic.

    X is a (4x4x5) ownership/suit/rank tensor.
    """

    def __init__(self, bot: Bot, replay_memory_location: pathlib.Path) -> None:
        self.bot = bot
        self.replay_memory_file_path = replay_memory_location

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        return self.bot.get_move(perspective=perspective, leader_move=leader_move)

    def notify_game_end(self, won: bool, perspective: PlayerPerspective) -> None:
        game_history = cast(list[tuple[PlayerPerspective, Trick]], perspective.get_game_history()[:-1])
        if not bool(won):
            return

        deck = list(SchnapsenDeckGenerator().get_initial_deck())

        for round_perspective, round_trick in game_history:
            # X
            x_tensor = get_state_feature_tensor(round_perspective)

            # y: encode the move played by *this* player in that trick.
            # The perspective stored in history is *from this bot's POV*, so:
            # - if we were leader then our move is trick.leader_move
            # - else our move is trick.follower_move (partial trick for followers in phase two is handled by history reconstruction)
            y_one_hot = [0] * 22

            my_move: Optional[Move] = None

            if round_perspective.am_i_leader():
                my_move = round_trick.leader_move
            else:
                # Exchange follower is still follower; for exchange trick, "our move" is the exchange.
                if round_trick.is_trump_exchange():
                    my_move = round_trick.as_trump_exchange()
                else:
                    my_move = round_trick.follower_move



            y_one_hot[_move_to_action_index(my_move, deck)] = 1


            append_replay_record(self.replay_memory_file_path, x_tensor, y_one_hot, True)
