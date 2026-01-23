from __future__ import annotations

import ast
import pathlib
from typing import Optional

from schnapsen.deck import Card
from schnapsen.game import GamePhase, Move, PlayerPerspective, SchnapsenDeckGenerator

# Reuse the canonical 22-action mapping from ml_binary so both pipelines stay consistent.
from schnapsen.bots.ml_binary.ml_binary_helpers import _move_to_action_index


def append_replay_record(
    replay_path: pathlib.Path,
    state_tensor: list[list[list[int]]],
    action_vec: list[int],
    won_label: bool,
) -> None:
    """Append one training sample to a replay-memory file.

    Format mirrors ml_binary, but X is a 3D tensor literal:
        "<state_tensor_literal> || <action_vec_literal>\n"

    This repo's ml_binary pipeline keeps only winning samples; we do the same here.
    """

    if not won_label:
        return

    replay_path = pathlib.Path(replay_path)
    line = f"{state_tensor} || {action_vec}\n"
    with replay_path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def parse_replay_part(part: str):
    """Parse one '||' part of a replay line (list literal or CSV)."""

    s = part.strip()
    if not s:
        return []
    if s.startswith("["):
        return ast.literal_eval(s)
    return [int(v.strip()) for v in s.split(",") if v.strip() != ""]


def card_to_suit_index(card: Card) -> int:
    """Map suit to an index (0..3) with a stable order."""

    name = card.suit.name
    if name == "DIAMONDS":
        return 0
    if name == "SPADES":
        return 1
    if name == "CLUBS":
        return 2
    if name == "HEARTS":
        return 3
    raise ValueError(f"Unknown suit: {card.suit}")


def card_to_rank_index(card: Card) -> int:
    """Map Schnapsen rank to an index (0..4) with a stable order."""

    name = card.rank.name
    if name == "JACK":
        return 0
    if name == "QUEEN":
        return 1
    if name == "KING":
        return 2
    if name == "TEN":
        return 3
    if name == "ACE":
        return 4
    raise ValueError(f"Unknown rank: {card.rank}")


def _ownership_bucket_for_card(perspective: PlayerPerspective, card: Card) -> int:
    """Return ownership bucket for a card (0..3), consistent with ml_binary."""

    if card in perspective.get_hand().cards:
        return 0  # on player's hand
    if card in perspective.get_won_cards().get_cards() or card in perspective.get_opponent_won_cards().get_cards():
        return 1  # out of the game
    if card in perspective.get_known_cards_of_opponent_hand():
        return 2  # known to be on opponent's hand
    return 3  # unknown (deck/opponent's hand)


def get_state_feature_tensor(
    perspective: PlayerPerspective,
    leader_move: Optional[Move] = None,
) -> list[list[list[int]]]:
    """Create the tensor state representation.

    Output shape: [channels=7][suit=4][rank=5]

    Channels:
      0..3: ownership buckets (same as ml_binary)
      4   : trump suit indicator (1 for cards of trump suit, else 0)
      5   : leader-move one-hot projected into suit/rank (only set when we are follower and leader_move is a regular move)
      6   : global context channel filled with constants:
              [*,0,0]=am_i_leader (0/1)
              [*,0,1]=is_phase_two (0/1)
            (repeated across suits just to keep tensor shape consistent)

    This keeps the core ownership×suit×rank encoding but brings back critical context
    that the binary pipeline has and the earlier tensor version lacked.
    """

    tensor = [[[0 for _ in range(5)] for _ in range(4)] for _ in range(7)]
    deck = list(SchnapsenDeckGenerator().get_initial_deck())

    # Ownership channels 0..3
    for c in deck:
        ch = _ownership_bucket_for_card(perspective, c)
        s = card_to_suit_index(c)
        r = card_to_rank_index(c)
        tensor[ch][s][r] = 1

    # Trump suit channel 4
    trump = perspective.get_trump_suit()
    for c in deck:
        s = card_to_suit_index(c)
        r = card_to_rank_index(c)
        tensor[4][s][r] = 1 if c.suit == trump else 0

    # Leader move channel 5 (only when follower and leader_move is a regular move)
    if (not perspective.am_i_leader()) and leader_move is not None and leader_move.is_regular_move():
        c = leader_move.as_regular_move().card
        tensor[5][card_to_suit_index(c)][card_to_rank_index(c)] = 1

    # Global context channel 6 packed into a couple of rank slots (repeated across suits)
    am_i_leader = 1 if perspective.am_i_leader() else 0
    is_phase_two = 1 if perspective.get_phase() == GamePhase.TWO else 0
    for s in range(4):
        tensor[6][s][0] = am_i_leader
        tensor[6][s][1] = is_phase_two

    return tensor


def get_action_mask(perspective: PlayerPerspective) -> list[int]:
    """Return the canonical 22-dim legal-move mask for this perspective."""

    deck = list(SchnapsenDeckGenerator().get_initial_deck())
    mask = [0] * 22
    for mv in perspective.valid_moves():
        try:
            mask[_move_to_action_index(mv, deck)] = 1
        except ValueError:
            continue
    return mask
