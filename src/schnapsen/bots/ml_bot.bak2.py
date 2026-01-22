from schnapsen.game import Bot, PlayerPerspective, SchnapsenDeckGenerator, Move, Trick, GamePhase, Hand, Previous
from typing import Optional, cast, Literal
from schnapsen.deck import Suit, Rank, Card
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import joblib
import time
import pathlib
from typing import *
import ast


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

        # Determine how many features the loaded model expects.
        # Some wrappers (e.g., MultiOutputRegressor) store it on the first estimator.
        self.__expected_n_features: Optional[int] = None
        if hasattr(self.__model, "n_features_in_"):
            self.__expected_n_features = int(getattr(self.__model, "n_features_in_"))
        elif hasattr(self.__model, "estimators_") and getattr(self.__model, "estimators_", None):
            first = getattr(self.__model, "estimators_")[0]
            if hasattr(first, "n_features_in_"):
                self.__expected_n_features = int(getattr(first, "n_features_in_"))

    def _score_candidates(self, X: "list[list[int]]") -> "list[float]":
        """Return one scalar score per row in X.

        Supports both:
        - classifiers (prefer `predict_proba`), returning P(class=1)
        - regressors / MultiOutputRegressor, returning a vector; we use the mean as a scalar score.
        """
        import numpy as np

        X_np = np.asarray(X, dtype=np.float32)

        # Classifier path
        if hasattr(self.__model, "predict_proba"):
            proba = self.__model.predict_proba(X_np)
            # Some sklearn wrappers return a list (e.g., multioutput classifiers)
            if isinstance(proba, list):
                # Use the first output head by default; take probability of positive class (index 1)
                proba0 = np.asarray(proba[0])
                if proba0.ndim == 2 and proba0.shape[1] >= 2:
                    return proba0[:, 1].astype(float).tolist()
                return proba0.reshape(-1).astype(float).tolist()

            proba_arr = np.asarray(proba)
            if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                return proba_arr[:, 1].astype(float).tolist()
            return proba_arr.reshape(-1).astype(float).tolist()

        # Regressor path
        y_hat = self.__model.predict(X_np)
        y_arr = np.asarray(y_hat, dtype=np.float32)
        if y_arr.ndim == 1:
            return y_arr.astype(float).tolist()
        # Vector output (e.g., 22 values): reduce to a scalar.
        return y_arr.mean(axis=1).astype(float).tolist()

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        """Choose the move with the best predicted outcome.

        Important: the model is trained on *state* feature vectors produced by `get_state_feature_vector`.
        Those state vectors already include (a) led-card information when following and (b) a legal-moves mask.
        So, at inference time we score each valid move by concatenating:
            [state_features] + [candidate_move_mask_22]
        where the move mask matches the format stored by `MLDataBot` as the target after `||`.
        """
        state_representation = get_state_feature_vector(perspective)

        my_valid_moves = perspective.valid_moves()
        assert len(my_valid_moves) > 0, "No valid moves available"

        # Build 22-dim move masks in the same format as the replay-memory target vector.
        deck = list(SchnapsenDeckGenerator().get_initial_deck())

        def move_to_mask(m: Move) -> list[int]:
            mask = [0 for _ in range(22)]
            if m.is_trump_exchange():
                mask[20] = 1
                return mask
            if m.is_marriage():
                mask[21] = 1
                return mask
            for i, c in enumerate(deck):
                if m.is_regular_move() and m.card == c:
                    mask[i] = 1
                    break
            return mask

        X: list[list[int]] = []



        for mv in my_valid_moves:

            X.append(state_representation + move_to_mask(mv))


        scores = self._score_candidates(X)
        assert len(scores) == len(my_valid_moves), "Model returned unexpected number of scores"

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return my_valid_moves[best_idx]


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

            # Support two possible replay formats:
            # 1) CSV: "f1,f2,... || t1,t2,..." (legacy MLDataBot output)
            # 2) Python list literals: "[f1, f2, ...] || [t1, t2, ...]" (ml_binary_data_bot output)
            def parse_part(part: str) -> list[int]:
                s = part.strip()
                if not s:
                    return []
                # If the part looks like a Python list literal, use ast.literal_eval
                if s[0] == '[':
                    parsed = ast.literal_eval(s)
                    # Ensure booleans become ints and numeric strings are converted
                    return [int(x) for x in parsed]
                # Otherwise assume CSV of numbers
                return [int(v.strip()) for v in s.split(',') if v.strip() != '']

            parts = line.split('||')
            if len(parts) < 2:
                # malformed line; skip
                continue
            feature_string = parts[0]
            # join the rest as target in case there are extra '||' tokens
            target_string = '||'.join(parts[1:])

            features = parse_part(feature_string)
            targets = parse_part(target_string)

            if features and targets:
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


    """
    # a list of all the features that consist the state feature set, of type np.ndarray
    state_feature_list: list[int] = []

    # am_i_leader is a method; store its boolean value
    state_feature_list.append(int(perspective.am_i_leader()))
    state_feature_list.append(int(perspective.get_phase() == GamePhase.TWO))

    ownership_values = (map_cards_to_ownership(perspective).values())

    for value in ownership_values:
        state_feature_list.append(int(value == 0))

    for value in ownership_values:
        state_feature_list.append(int(value == 1))

    for value in ownership_values:
        state_feature_list.append(int(value == 2))

    for value in ownership_values:
        state_feature_list.append(int(value == 3))

    for card in SchnapsenDeckGenerator().get_initial_deck():
        state_feature_list.append(int(card.suit == perspective.get_trump_suit()))
    is_marriage, is_trump_exchange, led_card = False, False, None
    history = perspective.get_game_history()
    # Only try to look at a previous trick if we are following and there is at least one completed trick
    if (not perspective.am_i_leader()) and len(history) >= 2:
        # history[-1] is the current perspective with trick=None
        # history[-2] is the previous perspective and its Trick
        _prev_persp, prev_trick = history[-2]
        if prev_trick is not None:
            if prev_trick.is_trump_exchange():
                is_trump_exchange = True
            else:
                # Regular or marriage trick: inspect the leader's move
                leader_move = prev_trick.as_partial().leader_move
                if leader_move.is_marriage():
                    is_marriage = True
                    # Encode the card which was effectively led because of the marriage.
                    # In this engine, that is the king card via underlying_regular_move(),
                    # but you can also use queen_card if that matches your dataset expectation.
                    led_card = leader_move.underlying_regular_move().card if hasattr(leader_move, "underlying_regular_move") else leader_move.queen_card
                else:
                    # Plain regular move
                    led_card = leader_move.card

    for card in SchnapsenDeckGenerator().get_initial_deck():
        led_active = (not is_marriage and not is_trump_exchange)
        state_feature_list.append(int(led_active and (led_card == card)))
        state_feature_list.append(int(is_marriage))
        state_feature_list.append(int(is_trump_exchange))
        state_feature_list.append(int(led_active and (led_card == card)))
        state_feature_list.append(int(is_marriage))
        state_feature_list.append(int(is_trump_exchange))

    valid_moves = perspective.valid_moves()
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
        state_feature_list.append(int(is_legal))
        state_feature_list.append(int(is_marriage_possible))
        state_feature_list.append(int(is_trump_exchange_possible))
        state_feature_list.append(int(is_legal))
        state_feature_list.append(int(is_marriage_possible))
        state_feature_list.append(int(is_trump_exchange_possible))
