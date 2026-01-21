from schnapsen.game import Bot, PlayerPerspective, SchnapsenDeckGenerator, Move, Trick, GamePhase, Hand
from typing import Optional, cast, Literal
from schnapsen.deck import Suit, Rank, Card
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import joblib
import time
import pathlib
from typing import *


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
            ownership[card] = 2 # Known to be on opponent's hand
        else:
            ownership[card] = 3  # unknown (deck/opponent's hand)

    return ownership


class MLPlayingBot(Bot):
    """
    This class loads a trained ML model and uses it to play
    """

    def __init__(self, model_location: pathlib.Path, name: Optional[str] = None) -> None:
        """
        Create a new MLPlayingBot which uses the model stored in the model_location.

        :param model_location: The file containing the model.
        """
        super().__init__(name)
        model_location = model_location
        assert model_location.exists(), f"Model could not be found at: {model_location}"
        # load model
        self.__model = joblib.load(model_location)

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        # get the sate feature representation
        state_representation = get_state_feature_vector(perspective)
        # get the leader's move representation, even if it is None
        leader_move_representation = get_move_feature_vector(leader_move)
        # get all my valid moves
        my_valid_moves = perspective.valid_moves()
        # get the feature representations for all my valid moves
        my_move_representations: list[list[int]] = []
        for my_move in my_valid_moves:
            my_move_representations.append(get_move_feature_vector(my_move))

        # create all model inputs, for all bot's valid moves
        action_state_representations: list[list[int]] = []

        if perspective.am_i_leader():
            follower_move_representation = get_move_feature_vector(None)
            for my_move_representation in my_move_representations:
                action_state_representations.append(
                    state_representation + my_move_representation + follower_move_representation)
        else:
            for my_move_representation in my_move_representations:
                action_state_representations.append(
                    state_representation + leader_move_representation + my_move_representation)

        model_output = self.__model.predict(action_state_representations)
        winning_probabilities_of_moves = [outcome_prob[1] for outcome_prob in model_output]
        highest_value: float = -1
        best_move = None
        for index, value in enumerate(winning_probabilities_of_moves):
            if value > highest_value:
                highest_value = value
                best_move = my_valid_moves[index]
        assert best_move is not None, "We went over all the moves, selecting the one we expect to lead to the highest average score. Simce there must have been at least one move at the start, this can never be None"
        return best_move


class MLDataBot(Bot):
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
        won_label = won

        feature_list: List[List[bool]] = []
        trick_list: List[Trick] = []
        """
        For each round
        1. Is leading
        2. Phase (0 = first, 1 = second)
        3-23. For each card in the deck: is_in_hand (20)
        24-44. For each card in the deck: is_out (20)
        45-65. For each card in the deck: is_unknown (20)
        66-86. For each card in the deck: is_known_opponent (20)
        87-107. For each card in the deck: is_trump (20)
        108-130. For each card in the deck + Trump exchange + Marriage: is_legal (22)
        131-151 + Trump exchange + Marriage: led_card (22) - to add
        """
        round_number = 0
        # Iterate over all the rounds of the history
        for round_player_perspective, round_trick in game_history:

            # For each round, create a new feature list
            if round_number >= len(feature_list):
                feature_list.append([])

            # 1-2 are whether the bot was leading/following and the phase of the game
            feature_list[round_number].append(bool(round_player_perspective.am_i_leader()))
            feature_list[round_number].append(bool(round_player_perspective.get_phase() == GamePhase.TWO))

            ownership_values = list(map_cards_to_ownership(round_player_perspective).values())

            """
            For each round, append the ownership information of each card in the deck
            is_in_hand for all 20 (3-23); is_out for all 20 (24-44), is_known_opponent for all 20 (45-65), is_unknown (66-86) each batch consecutively
            """

            for value in ownership_values:
                feature_list[round_number].append(value == 0)

            for value in ownership_values:
                feature_list[round_number].append(value == 1)

            for value in ownership_values:
                feature_list[round_number].append(value == 2)

            for value in ownership_values:
                feature_list[round_number].append(value == 3)


            # Add trump suit mask (87-107)
            for card in SchnapsenDeckGenerator().get_initial_deck():
                if card.suit == round_player_perspective.get_trump_suit():
                    feature_list[round_number].append(True)
                else:
                    feature_list[round_number].append(False)

            # Add led card mask (if not leading) -- 20 + Marriage + Trump exchange (108-130)
            is_marriage, is_trump_exchange, led_card = False, False, None
            if not round_player_perspective.am_i_leader():
                if round_trick.is_trump_exchange():
                    is_trump_exchange = True

                elif next(round_trick.cards) is not None and next(round_trick.cards) and next(round_trick.cards):
                    is_marriage = True

                else:
                    led_card = next(round_trick.cards)



            for card in SchnapsenDeckGenerator().get_initial_deck():
                led_active = (not is_marriage and not is_trump_exchange)
                feature_list[round_number].append(led_active and (led_card == card))
                feature_list[round_number].append(bool(is_marriage))
                feature_list[round_number].append(bool(is_trump_exchange))

            #Legal moves mask (131-153)
            valid_moves = round_player_perspective.valid_moves()
            is_marriage_possible, is_trump_exchange_possible = False, False
            for card in SchnapsenDeckGenerator().get_initial_deck():
                is_legal = False
                for move in valid_moves:
                    if move.is_marriage():
                        is_marriage_possible = True
                    if move.is_trump_exchange():
                        is_trump_exchange_possible = True
                    if move.is_regular_move():
                        if move.card == card:
                            is_legal = True
                            break
                feature_list[round_number].append(is_legal)
                feature_list[round_number].append(is_marriage_possible)
                feature_list[round_number].append(is_trump_exchange_possible)



            # append replay memory to file - write the constructed feature vector (as ints)
            with open(file=self.replay_memory_file_path, mode="a") as replay_memory_file:
                if int(won_label):
                    replay_memory_file.write(f"{str(feature_list[round_number])}")
            round_number += 1




            trick_list: [List[bool]] = [False for _ in range(22)]
            if round_trick.is_trump_exchange():
                trick_list[20] = True
            else:
                # round_trick.cards is an iterable; check for exactly two items
                cards_iter = iter(round_trick.cards)
                first = next(cards_iter, None)
                second = next(cards_iter, None)
                third = next(cards_iter, None)
                if first is not None and second is not None and third is not None:
                    trick_list[21] = True
                else:
                    trick_card = next(round_trick.cards)

                    counter = 0

                    for card in SchnapsenDeckGenerator().get_initial_deck():
                        if trick_card is not None and card == trick_card:
                            trick_list[counter] = True
                        else:
                            trick_list[counter] = False
                        counter += 1
            with open(file=self.replay_memory_file_path, mode="a") as replay_memory_file:
                if won_label:
                    replay_memory_file.write(f"|| {(trick_list)} \n")

        import ast


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

            feature_string, target_string = line.split("||")

            features = [int(v.strip()) for v in feature_string.split(",")]
            targets = [int(v.strip()) for v in target_string.split(",")]

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
    if model_class == 'NN':
        #############################################
        # Neural Network model parameters :
        # learn more about the model or how to use better use it by checking out its documentation
        # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
        # Play around with the model parameters below
        print("Training a Complex (Neural Network) model.")

        # Feel free to experiment with different number of neural layers or different type of neurons per layer
        # Tips: more neurons or more layers of neurons create a more complicated model that takes more time to train and
        # needs a bigger dataset, but if you find the correct combination of neurons and neural layers and provide a big enough training dataset can lead to better performance

        # one layer of 30 neurons
        hidden_layer_sizes = (30)
        # two layers of 30 and 5 neurons respectively
        # hidden_layer_sizes = (30, 5)

        # The learning rate determines how fast we move towards the optimal solution.
        # A low learning rate will converge slowly, but a large one might overshoot.
        learning_rate = 0.0001

        # The regularization term aims to prevent over-fitting, and we can tweak its strength here.
        regularization_strength = 0.0001

        # Train a neural network
        learner = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate,
                                alpha=regularization_strength, verbose=True, early_stopping=True, n_iter_no_change=6,
                                activation='tanh')
    elif model_class == 'LR':
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


def create_state_and_actions_vector_representation(perspective: PlayerPerspective, leader_move: Optional[Move],
                                                   follower_move: Optional[Move]) -> list[int]:
    """
    This function takes as input a PlayerPerspective variable, and the two moves of leader and follower,
    and returns a list of complete feature representation that contains all information
    """
    player_game_state_representation = get_state_feature_vector(perspective)
    leader_move_representation = get_move_feature_vector(leader_move)
    follower_move_representation = get_move_feature_vector(follower_move)

    return player_game_state_representation + leader_move_representation + follower_move_representation


def get_one_hot_encoding_of_card_suit(card_suit: Suit) -> list[int]:
    """
    Translating the suit of a card into one hot vector encoding of size 4.
    """
    card_suit_one_hot: list[int]
    if card_suit == Suit.HEARTS:
        card_suit_one_hot = [0, 0, 0, 1]
    elif card_suit == Suit.CLUBS:
        card_suit_one_hot = [0, 0, 1, 0]
    elif card_suit == Suit.SPADES:
        card_suit_one_hot = [0, 1, 0, 0]
    elif card_suit == Suit.DIAMONDS:
        card_suit_one_hot = [1, 0, 0, 0]
    else:
        raise ValueError("Suit of card was not found!")

    return card_suit_one_hot


def get_one_hot_encoding_of_card_rank(card_rank: Rank) -> list[int]:
    """
    Translating the rank of a card into one hot vector encoding of size 13.
    """
    card_rank_one_hot: list[int]
    if card_rank == Rank.ACE:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    elif card_rank == Rank.TWO:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif card_rank == Rank.THREE:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif card_rank == Rank.FOUR:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif card_rank == Rank.FIVE:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif card_rank == Rank.SIX:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif card_rank == Rank.SEVEN:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.EIGHT:
        card_rank_one_hot = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.NINE:
        card_rank_one_hot = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.TEN:
        card_rank_one_hot = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.JACK:
        card_rank_one_hot = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.QUEEN:
        card_rank_one_hot = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.KING:
        card_rank_one_hot = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        raise AssertionError("Provided card Rank does not exist!")
    return card_rank_one_hot


def get_move_feature_vector(move: Optional[Move]) -> list[int]:
    """
        In case there isn't any move provided move to encode, we still need to create a "padding"-"meaningless" vector of the same size,
        filled with 0s, since the ML models need to receive input of the same dimensionality always.
        Otherwise, we create all the information of the move i) move type, ii) played card rank and iii) played card suit
        translate this information into one-hot vectors respectively, and concatenate these vectors into one move feature representation vector
    """

    if move is None:
        move_type_one_hot_encoding_numpy_array = [0, 0, 0]
        card_rank_one_hot_encoding_numpy_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        card_suit_one_hot_encoding_numpy_array = [0, 0, 0, 0]

    else:
        move_type_one_hot_encoding: list[int]
        # in case the move is a marriage move
        if move.is_marriage():
            move_type_one_hot_encoding = [0, 0, 1]
            card = move.queen_card
        #  in case the move is a trump exchange move
        elif move.is_trump_exchange():
            move_type_one_hot_encoding = [0, 1, 0]
            card = move.jack
        #  in case it is a regular move
        else:
            move_type_one_hot_encoding = [1, 0, 0]
            card = move.card
        move_type_one_hot_encoding_numpy_array = move_type_one_hot_encoding
        card_rank_one_hot_encoding_numpy_array = get_one_hot_encoding_of_card_rank(card.rank)
        card_suit_one_hot_encoding_numpy_array = get_one_hot_encoding_of_card_suit(card.suit)

    return move_type_one_hot_encoding_numpy_array + card_rank_one_hot_encoding_numpy_array + card_suit_one_hot_encoding_numpy_array


def get_state_feature_vector(perspective: PlayerPerspective) -> list[int]:
    """
        This function gathers all subjective information that this bot has access to, that can be used to decide its next move, including:
        - points of this player (int)
        - points of the opponent (int)
        - pending points of this player (int)
        - pending points of opponent (int)
        - the trump suit (1-hot encoding)
        - phase of game (1-hoy encoding)
        - talon size (int)
        - if this player is leader (1-hot encoding)
        - What is the status of each card of the deck (where it is, or if its location is unknown)

        Important: This function should not include the move of this agent.
        It should only include any earlier actions of other agents (so the action of the other agent in case that is the leader)
    """
    # a list of all the features that consist the state feature set, of type np.ndarray
    state_feature_list: list[int] = []

    player_score = perspective.get_my_score()
    # - points of this player (int)
    player_points = player_score.direct_points
    # - pending points of this player (int)
    player_pending_points = player_score.pending_points

    # add the features to the feature set
    state_feature_list += [player_points]
    state_feature_list += [player_pending_points]

    opponents_score = perspective.get_opponent_score()
    # - points of the opponent (int)
    opponents_points = opponents_score.direct_points
    # - pending points of opponent (int)
    opponents_pending_points = opponents_score.pending_points

    # add the features to the feature set
    state_feature_list += [opponents_points]
    state_feature_list += [opponents_pending_points]

    # - the trump suit (1-hot encoding)
    trump_suit = perspective.get_trump_suit()
    trump_suit_one_hot = get_one_hot_encoding_of_card_suit(trump_suit)
    # add this features to the feature set
    state_feature_list += trump_suit_one_hot

    # - phase of game (1-hot encoding)
    game_phase_encoded = [1, 0] if perspective.get_phase() == GamePhase.TWO else [0, 1]
    # add this features to the feature set
    state_feature_list += game_phase_encoded

    # - talon size (int)
    talon_size = perspective.get_talon_size()
    # add this features to the feature set
    state_feature_list += [talon_size]

    # - if this player is leader (1-hot encoding)
    i_am_leader = [0, 1] if perspective.am_i_leader() else [1, 0]
    # add this features to the feature set
    state_feature_list += i_am_leader

    # gather all known deck information
    hand_cards = perspective.get_hand().cards
    trump_card = perspective.get_trump_card()
    won_cards = perspective.get_won_cards().get_cards()
    opponent_won_cards = perspective.get_opponent_won_cards().get_cards()
    opponent_known_cards = perspective.get_known_cards_of_opponent_hand().get_cards()
    # each card can either be i) on player's hand, ii) on player's won cards, iii) on opponent's hand, iv) on opponent's won cards
    # v) be the trump card or vi) in an unknown position -> either on the talon or on the opponent's hand
    # There are all different cases regarding card's knowledge, and we represent these 6 cases using one hot encoding vectors as seen bellow.

    deck_knowledge_in_consecutive_one_hot_encodings: list[int] = []

    for card in SchnapsenDeckGenerator().get_initial_deck():
        card_knowledge_in_one_hot_encoding: list[int]
        # i) on player's hand
        if card in hand_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 0, 0, 1]
        # ii) on player's won cards
        elif card in won_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 0, 1, 0]
        # iii) on opponent's hand
        elif card in opponent_known_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 1, 0, 0]
        # iv) on opponent's won cards
        elif card in opponent_won_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 1, 0, 0, 0]
        # v) be the trump card
        elif card == trump_card:
            card_knowledge_in_one_hot_encoding = [0, 1, 0, 0, 0, 0]
        # vi) in an unknown position as it is invisible to this player. Thus, it is either on the talon or on the opponent's hand
        else:
            card_knowledge_in_one_hot_encoding = [1, 0, 0, 0, 0, 0]
        # This list eventually develops to one long 1-dimensional numpy array of shape (120,)
        deck_knowledge_in_consecutive_one_hot_encodings += card_knowledge_in_one_hot_encoding
    # deck_knowledge_flattened: np.ndarray = np.concatenate(tuple(deck_knowledge_in_one_hot_encoding), axis=0)

    # add this features to the feature set
    state_feature_list += deck_knowledge_in_consecutive_one_hot_encodings

    return state_feature_list
