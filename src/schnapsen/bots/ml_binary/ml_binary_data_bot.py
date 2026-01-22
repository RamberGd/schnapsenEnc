from schnapsen.game import Bot, PlayerPerspective, SchnapsenDeckGenerator, Move, Trick, GamePhase, Hand, Previous
from schnapsen.deck import Suit, Rank, Card
import pathlib
from typing import *
import ast
from schnapsen.bots.ml_binary.ml_binary_helpers import get_state_feature_vector, append_replay_record, clean_up_replay_entry


class MLBinaryDataBot(Bot):

    """
    This class is defined to allow the creation of a training schnapsen bot dataset, that allows us to train a Machine Learning (ML) Bot
    Practically, it helps us record how the game plays out according to a provided Bot behaviour; build what is called a "replay memory"
    In more detail, we create one training sample for each decision the bot makes within a game, where a decision is an action selection for a specific game state.
    Then we relate each decision with the outcome of the game, i.e. whether this bot won or not.
    This way we can then train a bot according to the assumption that:
        "decisions in earlier games that ended up in victories should be preferred over decisions that lead to lost games"
    This class only records the decisions and game outcomes of the provided bot, according to its own perspective - incomplete game state knowledge.
    """

    def __init__(self, bot: Bot, replay_memory_location: pathlib.Path) -> None:
        """
        :param bot: the provided bot that will actually play the game and make decisions
        :param replay_memory_location: the filename under which the replay memory records will be
        """

        self.bot: Bot = bot
        self.replay_memory_file_path: pathlib.Path = replay_memory_location

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        """
            This function simply calls the get_move of the provided bot
        """
        return self.bot.get_move(perspective=perspective, leader_move=leader_move)

    def notify_game_end(self, won: bool, perspective: PlayerPerspective) -> None:
        """
        When the game ends, this function retrieves the game history and more specifically all the replay memories that can
        be derived from it, and stores them in the form of state-actions vector representations and the corresponding outcome of the game

        :param won: Did this bot win the game?
        :param state: The final state of the game.
        """

        # we retrieve the game history while actually discarding the last useless history record (which is after the game has ended),
        # we know none of the Tricks can be None because that is only for the last record
        game_history: list[tuple[PlayerPerspective, Trick]] = cast(list[tuple[PlayerPerspective, Trick]], perspective.get_game_history()[:-1])
        # we also save the training label "won or lost"
        won_label = bool(won)

        # Iterate over all the rounds of the history and append one record per round in a single append operation
        for round_player_perspective, round_trick in game_history:

            # Build state features using the existing function (kept unchanged)
            state_features = get_state_feature_vector(round_player_perspective)

            # Build trick list representation (22-length boolean list)
            trick_list = [0 for _ in range(22)]



            if round_trick.is_trump_exchange():
                trick_list[20] = 1
            else:
                # Convert the trick cards to a list to inspect their count
                cards = list(round_trick.cards)
                if len(cards) >= 3:
                    # Marriage trick
                    trick_list[21] = 1
                else:
                    trick_card = cards[0]
                    counter = 0
                    for card in SchnapsenDeckGenerator().get_initial_deck():
                        if trick_card is not None and card == trick_card:
                            trick_list[counter] = 1
                        else:
                            trick_list[counter] = 0
                        counter += 1

            trick_list = clean_up_replay_entry(trick_list)

            # Append a single record: state_features || trick_list || won_label

            append_replay_record(self.replay_memory_file_path, state_features, trick_list, won_label)

        # Note: we intentionally do not call convert_replay_memory_to_binary here to avoid rewriting the whole file
        # on every game end. The conversion utility can be run as a separate maintenance step when needed.
