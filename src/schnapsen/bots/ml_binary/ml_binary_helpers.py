import pathlib
import ast
from schnapsen.game import Bot, PlayerPerspective, SchnapsenDeckGenerator, Move, Trick, GamePhase, Hand, Previous
from schnapsen.deck import Suit, Rank, Card


def clean_up_replay_entry(replay_entry: list[int]) -> str:
    """Legacy helper to stringify a list of ints/booleans as comma separated 0/1."""
    return str(replay_entry).replace("True", "1").replace("False", "0").replace("[", "").replace("]", "")


def append_replay_record(replay_path: pathlib.Path, state_features: list[int], action_vec: list[int], won_label: bool) -> None:
    """Append a single replay record to the replay file.

    Binary (ml_binary) replay format:
        "<state_features_as_python_list> || <action_vec_as_python_list>\n"

    Note: In this repo version, we intentionally keep only winning samples (won_label==True).
    """
    replay_path = pathlib.Path(replay_path)
    if not won_label:
        return
    line = f"{state_features} || {action_vec}\n"
    with replay_path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def map_cards_to_ownership(perspective: PlayerPerspective) -> dict[Card, int]:

    """
    Function that maps each card in the deck to the ownership status according to the perspective parameter.

    :param perspective: The PlayerPerspective of the bot - will determine which cards were possible to be seen at the call of the function

    :return: A dictionary mapping each card to an integer representing its ownership status:
             0 - on player's hand
             1 - out of the game (won cards)
             2 - known to be on opponent's hand (from marriages or trump exchanges)
             3 - unknown (deck/opponent's hand)

    """
    ownership: dict[Card, int] = {}
    for card in SchnapsenDeckGenerator().get_initial_deck():

        if card in perspective.get_hand().cards:
            ownership[card] = 0  # on player's hand
        elif card in perspective.get_won_cards().get_cards() or card in perspective.get_opponent_won_cards().get_cards():
            ownership[card] = 1  # out of the game
        elif card in perspective.get_known_cards_of_opponent_hand():
            ownership[card] = 2  # Known to be on opponent's hand
        else:
            ownership[card] = 3  # unknown (deck/opponent's hand)

    return ownership


def convert_replay_memory_to_binary(self) -> None:
    """
    Converts the replay memory file from boolean list representation to binary (0/1) representation for model input.
    Args:
        self:

    Returns:

    """
    with open(self.replay_memory_file_path, "r") as f:
        lines = f.readlines()

    with open(self.replay_memory_file_path, "w") as out:
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split("||")

            converted_parts = []

            for part in parts:
                part = part.strip()
                bool_list = ast.literal_eval(part)
                binary = ", ".join("1" if x else "0" for x in bool_list)
                converted_parts.append(binary)

            out.write(" || ".join(converted_parts) + "\n")


def _move_to_action_index(move: Move, deck: list[Card]) -> int:
    """Map a Move to the canonical 22-dim action index.

    Indices:
      0..19 - regular move: play that card (deck order)
      20    - trump exchange
      21    - marriage
    """
    if move.is_trump_exchange():
        return 20
    if move.is_marriage():
        return 21
    if move.is_regular_move():
        for i, c in enumerate(deck):
            if move.card == c:
                return i
    raise ValueError(f"Unsupported/unmappable move: {move}")


def get_state_feature_vector(perspective: PlayerPerspective, leader_move: Move | None = None) -> list[int]:
    """Create the model's state feature vector.

    Feature contract (stable):
      0      : am_i_leader (0/1)
      1      : is_phase_two (0/1)
      2..81  : card ownership as 4 blocks of 20 (in_hand, out_of_game, known_opponent, unknown)
      82..101: is_trump_suit for each card (20)
      102..123: leader_move_one_hot (22) -- all zeros if I'm leader; otherwise encodes leader_move (0..19/20/21)
      124..145: legal_action_multi_hot (22) for this state

    Total length: 146

    IMPORTANT:
    - leader_move_one_hot must reflect the *current trick's* leader move when we are the follower.
      Do NOT derive it from previous-trick history.
    """

    deck: list[Card] = list(SchnapsenDeckGenerator().get_initial_deck())

    state_features: list[int] = []
    state_features.append(int(perspective.am_i_leader()))
    state_features.append(int(perspective.get_phase() == GamePhase.TWO))

    ownership = map_cards_to_ownership(perspective)

    for c in deck:
        state_features.append(int(ownership[c] == 0))
    for c in deck:
        state_features.append(int(ownership[c] == 1))
    for c in deck:
        state_features.append(int(ownership[c] == 2))
    for c in deck:
        state_features.append(int(ownership[c] == 3))

    trump_suit = perspective.get_trump_suit()
    for c in deck:
        state_features.append(int(c.suit == trump_suit))

    # Leader move context (22)
    leader_one_hot = [0] * 22
    if not perspective.am_i_leader():
        if leader_move is not None:
            try:
                leader_one_hot[_move_to_action_index(leader_move, deck)] = 1
            except ValueError:
                # stay all-zeros if we somehow can't map it
                pass
    state_features.extend(leader_one_hot)

    # Legal action mask (22) - no duplicated flags.
    legal_action_vec = [0] * 22
    for mv in perspective.valid_moves():
        try:
            legal_action_vec[_move_to_action_index(mv, deck)] = 1
        except ValueError:
            continue
    state_features.extend(legal_action_vec)

    return state_features
