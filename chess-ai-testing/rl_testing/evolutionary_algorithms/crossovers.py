import abc
import itertools
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import chess
import numpy as np

from rl_testing.evolutionary_algorithms.individuals import BoardIndividual, Individual
from rl_testing.util.chess import is_really_valid, has_undefended_attacked_pieces
from rl_testing.util.util import get_random_state


class CrossoverName(Enum):
    CROSSOVER_HALF_BOARD = 0
    CROSSOVER_ONE_QUARTER_BOARD = 1
    CROSSOVER_ONE_EIGHTH_BOARD = 2
    CROSSOVER_EXCHANGE_PIECE_PAIRS = 3


def _ensure_single_kings(
    board: chess.Board,
    random_state: np.random.Generator,
) -> chess.Board:
    """Ensure that the board has only one king per color.

    Args:
        board (chess.Board): The board.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        chess.Board: The board with only one king per color.
    """

    for color in [chess.WHITE, chess.BLACK]:
        # Get the king squares
        squares = [
            square
            for square, piece in board.piece_map().items()
            if piece.piece_type == chess.KING and piece.color == color
        ]

        # If there are more than two kings per color, randomly remove one of them
        if len(squares) > 1:
            board.remove_piece_at(random_state.choice(squares))

        # If there are no kings, randomly place one
        while len(squares) == 0:
            piece_map = board.piece_map()

            # Find all empty squares which are not attacked by the opposing color
            empty_squares = [
                square
                for square in chess.SQUARES
                if board.piece_at(square) is None and not board.is_attacked_by(not color, square)
            ]

            # If there are no empty squares, randomly remove a piece
            if len(empty_squares) == 0:
                board.remove_piece_at(random_state.choice(list(piece_map.keys())))
            else:
                board.set_piece_at(
                    random_state.choice(empty_squares),
                    chess.Piece(chess.KING, color),
                )
                break

    # Assert that there is exactly one king per color
    pieces = board.piece_map().values()
    for color in [chess.WHITE, chess.BLACK]:
        color_name = "white" if color == chess.WHITE else "black"
        num_kings = sum(
            [piece.piece_type == chess.KING and piece.color == color for piece in pieces]
        )
        assert num_kings == 1, f"Board has {num_kings} {color_name} kings."

    return board


def crossover_half_board(
    board1: chess.Board,
    board2: chess.Board,
    axis: Optional[str] = None,
    ensure_single_kings: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> Tuple[chess.Board, chess.Board]:
    """Crossover function that swaps the horizontal halves of the two boards.

    Args:
        board1 (chess.Board): First board.
        board2 (chess.Board): Second board.
        axis (Optional[str], optional): Whether to swap one of the horizontal halves or the vertical halves.
            Must be either "horizontal" or "vertical". Defaults to None in which case it is chosen randomly.
        ensure_single_kings (bool, optional): Whether to ensure that the boards have only one king per color after the crossover.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        Tuple[chess.Board, chess.Board]: The two boards after the crossover.
    """

    random_state = get_random_state(_random_state)

    if axis is None:
        axis = random_state.choice(["horizontal", "vertical"])

    assert axis in [
        "horizontal",
        "vertical",
    ], f"Axis must be either 'horizontal' or 'vertical', got {axis}."

    if axis == "horizontal":
        files = range(8)
        ranks = range(4)
    elif axis == "vertical":
        files = range(4)
        ranks = range(8)

    for rank in ranks:
        for file in files:
            piece1 = board1.piece_at(chess.square(file, rank))
            piece2 = board2.piece_at(chess.square(file, rank))
            board1.set_piece_at(chess.square(file, rank), piece2)
            board2.set_piece_at(chess.square(file, rank), piece1)

    if ensure_single_kings:
        board1 = _ensure_single_kings(board1, random_state)
        board2 = _ensure_single_kings(board2, random_state)

    logging.debug(f"Swapped {axis} halves of the boards.")

    return board1, board2


def crossover_one_quarter_board(
    board1: chess.Board,
    board2: chess.Board,
    ensure_single_kings: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> Tuple[chess.Board, chess.Board]:
    """Crossover function that swaps the quarter boards of the two boards.

    Args:
        board1 (chess.Board): First board.
        board2 (chess.Board): Second board.
        ensure_single_kings (bool, optional): Whether to ensure that the boards have only one king per color after the crossover.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        Tuple[chess.Board, chess.Board]: The two boards after the crossover.
    """

    random_state = get_random_state(_random_state)

    # Randomly select one of the four quarters
    quarter = random_state.choice(["top_left", "top_right", "bottom_left", "bottom_right"])

    if quarter == "top_left":
        files = range(4)
        ranks = range(4, 8)
    elif quarter == "top_right":
        files = range(4, 8)
        ranks = range(4, 8)
    elif quarter == "bottom_left":
        files = range(4)
        ranks = range(4)
    elif quarter == "bottom_right":
        files = range(4, 8)
        ranks = range(4)

    for rank in ranks:
        for file in files:
            piece1 = board1.piece_at(chess.square(file, rank))
            piece2 = board2.piece_at(chess.square(file, rank))
            board1.set_piece_at(chess.square(file, rank), piece2)
            board2.set_piece_at(chess.square(file, rank), piece1)

    if ensure_single_kings:
        board1 = _ensure_single_kings(board1, random_state)
        board2 = _ensure_single_kings(board2, random_state)

    logging.debug(f"Swapped {quarter} quarter of the boards.")

    return board1, board2


def crossover_one_eighth_board(
    board1: chess.Board,
    board2: chess.Board,
    ensure_single_kings: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> Tuple[chess.Board, chess.Board]:
    """Crossover function that swaps the eighth boards of the two boards.

    Args:
        board1 (chess.Board): First board.
        board2 (chess.Board): Second board.
        ensure_single_kings (bool, optional): Whether to ensure that the boards have only one king per color after the crossover.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        Tuple[chess.Board, chess.Board]: The two boards after the crossover.
    """

    random_state = get_random_state(_random_state)

    # Randomly select one of the eight quarters
    start_rank = random_state.choice(range(7))
    start_file = random_state.choice(range(7))

    files = range(start_file, start_file + 2)
    ranks = range(start_rank, start_rank + 2)

    for rank in ranks:
        for file in files:
            piece1 = board1.piece_at(chess.square(file, rank))
            piece2 = board2.piece_at(chess.square(file, rank))
            board1.set_piece_at(chess.square(file, rank), piece2)
            board2.set_piece_at(chess.square(file, rank), piece1)

    if ensure_single_kings:
        board1 = _ensure_single_kings(board1, random_state)
        board2 = _ensure_single_kings(board2, random_state)

    logging.debug(f"Swapped ({start_rank},{start_file}) eighth of the boards.")

    return board1, board2


def _crossover_exchange_piece_pairs_build_candidates(
    board1: chess.Board,
    board2: chess.Board,
    piece_combination1: Tuple[int, int],
    piece_combination2: Tuple[int, int],
) -> Tuple[chess.Board, chess.Board]:
    board1_original = board1.copy()
    board2_original = board2.copy()

    # Add the new pieces
    board1_new_pos1, board1_new_pos2 = piece_combination2
    board1_new_piece1, board1_new_piece2 = board2.piece_at(piece_combination2[0]), board2.piece_at(
        piece_combination2[1]
    )
    board2_new_pos1, board2_new_pos2 = piece_combination1
    board2_new_piece1, board2_new_piece2 = board1.piece_at(piece_combination1[0]), board1.piece_at(
        piece_combination1[1]
    )
    board1.set_piece_at(board1_new_pos1, board1_new_piece1)
    board1.set_piece_at(board1_new_pos2, board1_new_piece2)
    board2.set_piece_at(board2_new_pos1, board2_new_piece1)
    board2.set_piece_at(board2_new_pos2, board2_new_piece2)

    # Remove the old pieces
    if piece_combination1[0] not in piece_combination2:
        board1.remove_piece_at(piece_combination1[0])
    if piece_combination1[1] not in piece_combination2:
        board1.remove_piece_at(piece_combination1[1])
    if piece_combination2[0] not in piece_combination1:
        board2.remove_piece_at(piece_combination2[0])
    if piece_combination2[1] not in piece_combination1:
        board2.remove_piece_at(piece_combination2[1])

    # Ensure that the number of pieces is 8
    if len(board1.piece_map()) != 8:
        print(f"Original board1: {board1_original.fen()}")
        print(f"Original board2: {board2_original.fen()}")
        print(
            f"Square combination1: {[chess.square_name(square) for square in piece_combination1]}"
        )
        print(
            f"Square combination2: {[chess.square_name(square) for square in piece_combination2]}"
        )
        raise ValueError(f"Board1 has {len(board1.piece_map())} pieces.")

    if len(board2.piece_map()) != 8:
        print(f"Original board1: {board1_original.fen()}")
        print(f"Original board2: {board2_original.fen()}")
        print(
            f"Square combination1: {[chess.square_name(square) for square in piece_combination1]}"
        )
        print(
            f"Square combination2: {[chess.square_name(square) for square in piece_combination2]}"
        )
        raise ValueError(f"Board2 has {len(board2.piece_map())} pieces.")

    return board1, board2


def crossover_exchange_piece_pairs(
    board1: chess.Board,
    board2: chess.Board,
    ensure_single_kings: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> Tuple[chess.Board, chess.Board]:
    """Crossover function that swaps one pair of pieces of the same type and opposite color with each other.
    E.G. board1 could have a white and black Knight, and board2 could have a white and black Bishop. This function
    would then move the two Knights on board2 and the two Bishops on board1.

    This function expects that the boards have symmetric positions, where White and Black have the same pieces.

    Args:
        board1 (chess.Board): First board.
        board2 (chess.Board): Second board.
        ensure_single_kings (bool, optional): Whether to ensure that the boards have only one king per color after the crossover.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        Tuple[chess.Board, chess.Board]: The two boards after the crossover.
    """
    random_state = get_random_state(_random_state)

    # Get the two piece maps
    piece_map1 = board1.piece_map()
    piece_map2 = board2.piece_map()

    def create_piece_combinations(
        piece_map: Dict[chess.Square, chess.Piece]
    ) -> List[Tuple[chess.Square, chess.Square]]:
        # Separate white and black pieces
        white_pieces = {
            square: piece for square, piece in piece_map.items() if piece.color == chess.WHITE
        }
        black_pieces = {
            square: piece for square, piece in piece_map.items() if piece.color == chess.BLACK
        }

        # Create a list of piece-combinations of pieces of the same type and opposite color
        piece_combinations = []
        for piece_type in [chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]:
            white_piece_squares = [
                square for square, piece in white_pieces.items() if piece.piece_type == piece_type
            ]
            black_piece_squares = [
                square for square, piece in black_pieces.items() if piece.piece_type == piece_type
            ]

            # Create all possible combinations of pieces of the same type and opposite color
            piece_combinations.extend(
                list(itertools.product(white_piece_squares, black_piece_squares))
            )

        return piece_combinations

    # Create all possible piece combinations for the two boards
    piece_combinations1 = create_piece_combinations(piece_map1)
    piece_combinations2 = create_piece_combinations(piece_map2)

    # Create all possible combinations of piece combinations
    final_piece_combinations = list(itertools.product(piece_combinations1, piece_combinations2))

    # Check for each combination whether this is a valid crossover, i.e. whether there wouldn't be
    # too many pieces of the same type and color on one board
    valid_piece_combinations = []
    for piece_combination1, piece_combination2 in final_piece_combinations:
        # Check that the squares would not remove some other piece from the board
        combination_invalid = False
        for square in piece_combination1:
            if square not in piece_combination2 and board2.piece_at(square) is not None:
                combination_invalid = True
                break

        for square in piece_combination2:
            if square not in piece_combination1 and board1.piece_at(square) is not None:
                combination_invalid = True
                break

        if combination_invalid:
            continue

        # Get the pieces on the boards
        piece1_1 = piece_map1[piece_combination1[0]]
        piece2_1 = piece_map1[piece_combination1[1]]
        piece1_2 = piece_map2[piece_combination2[0]]
        piece2_2 = piece_map2[piece_combination2[1]]

        # Assert that the pieces coming from the same board are of the same type
        assert (
            piece1_1.piece_type == piece2_1.piece_type
        ), f"Piece types of {piece1_1} and {piece2_1} are not the same."
        assert (
            piece1_2.piece_type == piece2_2.piece_type
        ), f"Piece types of {piece1_2} and {piece2_2} are not the same."

        # Extract the piece types of the two boards
        piece_type1 = piece1_1.piece_type
        piece_type2 = piece1_2.piece_type

        # Get the number of pieces of piece_type2 which currently are on board1
        num_pieces1 = sum(
            [
                piece.piece_type == piece_type2 and piece.color == chess.WHITE
                for piece in piece_map1.values()
            ]
        )

        # Get the number of pieces of piece_type1 which currently are on board2
        num_pieces2 = sum(
            [
                piece.piece_type == piece_type1 and piece.color == chess.WHITE
                for piece in piece_map2.values()
            ]
        )
        max_pieces_allowed = {
            chess.ROOK: 2,
            chess.KNIGHT: 2,
            chess.BISHOP: 2,
            chess.QUEEN: 1,
            chess.KING: 1,
        }

        # Filter out the combination if it would lead to too many pieces of the same type on
        # one board
        if (num_pieces1 + 1 > max_pieces_allowed[piece_type1] and piece_type1 != piece_type2) or (
            num_pieces2 + 1 > max_pieces_allowed[piece_type2] and piece_type1 != piece_type2
        ):
            continue

        # Build the new boards and check if they are valid
        new_board1, new_board2 = _crossover_exchange_piece_pairs_build_candidates(
            board1.copy(), board2.copy(), piece_combination1, piece_combination2
        )

        # Check if the new boards are valid
        if not is_really_valid(new_board1) or not is_really_valid(new_board2):
            continue

        # If the combination is valid, add it to the list of valid combinations
        valid_piece_combinations.append((piece_combination1, piece_combination2))

    # If there are no valid piece combinations, return the original boards
    if len(valid_piece_combinations) == 0:
        return board1, board2

    # Select a random piece combination
    piece_combination1, piece_combination2 = random_state.choice(valid_piece_combinations)

    # Reverse the piece combinations with a probability of 0.5
    if random_state.choice([True, False]):
        piece_combination1 = piece_combination1[::-1]

    # Build the new boards
    board1, board2 = _crossover_exchange_piece_pairs_build_candidates(
        board1, board2, piece_combination1, piece_combination2
    )

    return board1, board2


def validity_wrapper(
    function: Callable[[chess.Board, chess.Board, Any], Tuple[chess.Board, chess.Board]],
    retries: int = 0,
) -> Callable[[chess.Board, chess.Board, Any], Tuple[chess.Board, chess.Board]]:
    """Wrapper for the crossover functions that checks if the input boards are valid after the crossover.
    If the boards aren't valid, the original boards are returned.

    Args:
        function (function): The crossover function to wrap.
        retries (int, optional): The number of times to retry the crossover if the board is invalid. Defaults to 0.
        *args: The arguments to pass to the crossover function.
        **kwargs: The keyword arguments to pass to the crossover function.

    Returns:
        inner_function (function): The wrapped crossover function.
    """

    def inner_function(
        board1: chess.Board, board2: chess.Board, *args: Any, **kwargs: Any
    ) -> chess.Board:
        for _ in range(retries + 1):
            # Clone the original board
            board_candidate1 = board1.copy()
            board_candidate2 = board2.copy()

            # Retry the crossover if the board is invalid
            board_candidate1, board_candidate2 = function(
                board_candidate1, board_candidate2, *args, **kwargs
            )

            # Check if the board is valid
            if is_really_valid(board_candidate1) and is_really_valid(board_candidate2):
                return board_candidate1, board_candidate2

        logging.debug(
            f"Board {board_candidate1.fen()} or Board {board_candidate2.fen()}"
            f" is invalid after crossover '{function.__name__}', returning original boards"
        )
        return board1, board2

    return inner_function


def clear_fitness_values_wrapper(
    function: Callable[[Individual, Individual, Any], Tuple[Individual, Individual]]
) -> Callable[[Individual, Individual, Any], Tuple[Individual, Individual]]:
    """Wrapper for crossover functions that clears the fitness values of the mated individuals.

    Args:
        function (Callable[[Individual, Individual, Any], Tuple[Individual, Individual]]): The crossover function to wrap.
    Returns:
        Callable[[Individual, Individual, Any], Tuple[Individual, Individual]]: The wrapped crossover function.
    """

    def inner_function(
        individual1: Individual, individual2: Individual, *args: Any, **kwargs: Any
    ) -> Tuple[Individual, Individual]:
        # Call the crossover function
        crossed_individual1, crossed_individual2 = function(
            individual1, individual2, *args, **kwargs
        )

        # Clear the fitness values
        del crossed_individual1.fitness
        del crossed_individual2.fitness

        return crossed_individual1, crossed_individual2

    return inner_function


def print_side_by_side(board1: chess.Board, board2: chess.Board) -> None:
    """Prints the two boards side by side.

    Args:
        board1 (chess.Board): First board.
        board2 (chess.Board): Second board.
    """
    board1_ranks = str(board1).split("\n")
    board2_ranks = str(board2).split("\n")

    for rank1, rank2 in zip(board1_ranks, board2_ranks):
        print(rank1, "\t", rank2)
    print("\n")


CROSSOVER_NAME_MAP = {
    crossover_half_board: CrossoverName.CROSSOVER_HALF_BOARD,
    crossover_one_quarter_board: CrossoverName.CROSSOVER_ONE_QUARTER_BOARD,
    crossover_one_eighth_board: CrossoverName.CROSSOVER_ONE_EIGHTH_BOARD,
    crossover_exchange_piece_pairs: CrossoverName.CROSSOVER_EXCHANGE_PIECE_PAIRS,
}


class CrossoverFunction:
    def __init__(
        self,
        function: Callable[[chess.Board, chess.Board, Any], Tuple[chess.Board, chess.Board]],
        probability: float = 1.0,
        retries: int = 0,
        check_game_not_over: bool = False,
        check_undefended_attacked_pieces: bool = False,
        clear_fitness_values: bool = False,
        _random_state: Optional[np.random.Generator] = None,
        *args,
        **kwargs,
    ):
        """A convenience class for storing crossover functions together with some settings.
        Args:
            function (Callable[[chess.Board, chess.Board, Any], Tuple[chess.Board, chess.Board]]): The crossover function.
            retries (int, optional): The number of times to retry the crossover if the board is invalid. Defaults to 0.
            clear_fitness_values (bool, optional): Whether to clear the fitness values of the mated individual.
                Defaults to False.
            _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None
            *args: Default arguments to pass to the crossover function.
            **kwargs: Default keyword arguments to pass to the crossover function.
        """
        self.probability = probability
        self.retries = retries
        self.random_state = get_random_state(_random_state)

        self.clear_fitness_values = clear_fitness_values
        self.check_game_not_over = check_game_not_over
        self.check_undefended_attacked_pieces = check_undefended_attacked_pieces
        self.function: Callable[
            [chess.Board, chess.Board, Any], Tuple[chess.Board, chess.Board]
        ] = function

        self.args = args
        self.kwargs = kwargs

    def __id__(self) -> int:
        return f"CrossoverFunction({self.function.__name__})"

    def __hash__(self) -> int:
        return hash(self.__id__())

    def __eq__(self, other: Any) -> bool:
        return self.__id__() == other.__id__()

    def __call__(
        self, board1: BoardIndividual, board2: BoardIndividual, *new_args: Any, **new_kwargs: Any
    ) -> Tuple[BoardIndividual, BoardIndividual]:
        """Call the crossover function.

        Args:
            board1 (BoardIndividual): The first board to mate.
            board2 (BoardIndividual): The second board to mate.
            *new_args: Additional arguments to pass to the crossover function.
            **new_kwargs: Additional keyword arguments to pass to the crossover function.

        Returns:
            Tuple[BoardIndividual, BoardIndividual]: The mated boards.
        """
        for _ in range(self.retries + 1):
            # Clone the original board
            board_candidate1 = board1.copy()
            board_candidate2 = board2.copy()

            # Retry the crossover if the board is invalid
            board_candidate1: BoardIndividual
            board_candidate2: BoardIndividual
            board_candidate1, board_candidate2 = self.function(
                board_candidate1,
                board_candidate2,
                _random_state=self.random_state,
                *self.args,
                *new_args,
                **self.kwargs,
                **new_kwargs,
            )

            # Ensure that the number of pieces is 8
            if len(board_candidate1.piece_map()) != 8:
                print(f"Original board1: {board1.fen()}")
                print(f"Original board2: {board2.fen()}")
                print(f"Board1: {board_candidate1.fen()}")
                print(f"Board2: {board_candidate2.fen()}")
                print(f"Crossover function: {self.function.__name__}")
                raise ValueError(f"Board1 has {len(board_candidate1.piece_map())} pieces.")
            if len(board_candidate2.piece_map()) != 8:
                print(f"Original board1: {board1.fen()}")
                print(f"Original board2: {board2.fen()}")
                print(f"Board1: {board_candidate1.fen()}")
                print(f"Board2: {board_candidate2.fen()}")
                print(f"Crossover function: {self.function.__name__}")
                raise ValueError(f"Board1 has {len(board_candidate2.piece_map())} pieces.")

            # Check if the board is valid
            if is_really_valid(board_candidate1) and is_really_valid(board_candidate2):
                if not self.check_game_not_over or (
                    len(list(board_candidate1.legal_moves)) > 0
                    and len(list(board_candidate2.legal_moves)) > 0
                ):
                    if self.check_undefended_attacked_pieces:
                        if has_undefended_attacked_pieces(
                            board_candidate1
                        ) or has_undefended_attacked_pieces(board_candidate2):
                            continue

                    # Clear the fitness values if requested
                    if self.clear_fitness_values:
                        del board_candidate1.fitness
                        del board_candidate2.fitness

                    # Add the crossover function name to the board's history
                    board_candidate1.history.append(CROSSOVER_NAME_MAP[self.function])
                    board_candidate2.history.append(CROSSOVER_NAME_MAP[self.function])

                    logging.debug(f"Applied crossover '{self.function.__name__}'")

                    return board_candidate1, board_candidate2

        logging.debug(
            f"Board {board_candidate1.fen()} or Board {board_candidate2.fen()}"
            f" is invalid after crossover '{self.function.__name__}', returning original boards"
        )
        return board1, board2


class CrossoverStrategy(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "__call__")
            and hasattr(subclass, "analyze_crossover_effects")
            and hasattr(subclass, "print_crossover_probability_history")
        ) or NotImplemented

    def __init__(
        self,
        crossover_functions: List[CrossoverFunction],
        _random_state: Optional[np.random.Generator] = None,
    ):
        """A base class for crossover strategies.

        Args:
            crossover_functions (List[CrossoverFunction]): A **pointer** to the list of crossover functions.
                This means that the list of crossover functions can be modified externally and the changes
                will be reflected here.
        """
        self.random_state: np.random.Generator = get_random_state(_random_state)
        self.crossover_functions: List[CrossoverFunction] = crossover_functions

    @abc.abstractmethod
    def __call__(
        self,
        individual1: BoardIndividual,
        individual2: BoardIndividual,
        random_seed: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[BoardIndividual, BoardIndividual]:
        raise NotImplementedError

    @abc.abstractmethod
    def analyze_crossover_effects(
        self, population: List[BoardIndividual], print_update: bool = False
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def print_crossover_probability_history(self) -> None:
        raise NotImplementedError


class AllCrossoverFunctionsStrategy(CrossoverStrategy):
    def __call__(
        self,
        individual1: BoardIndividual,
        individual2: BoardIndividual,
        random_seed: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[BoardIndividual, BoardIndividual]:
        """Apply all crossover functions to the two individuals.

        Args:
            individual1 (BoardIndividual): The first individual.
            individual2 (BoardIndividual): The second individual.
            *args: Additional arguments to pass to the crossover functions.
            **kwargs: Additional keyword arguments to pass to the crossover functions.

        Returns:
            Tuple[BoardIndividual, BoardIndividual]: The mated individuals.
        """
        if random_seed:
            self.random_state = np.random.default_rng(random_seed)
        for crossover_function in self.crossover_functions:
            if self.random_state.uniform() < crossover_function.probability:
                individual1, individual2 = crossover_function(
                    individual1, individual2, *args, **kwargs
                )

        return individual1, individual2

    def analyze_crossover_effects(
        self, population: List[BoardIndividual], print_update: bool = False
    ) -> None:
        return

    def print_crossover_probability_history(self) -> None:
        for crossover_function in self.crossover_functions:
            print(f"{crossover_function.function.__name__}: {crossover_function.probability}")


class OneRandomCrossoverFunctionStrategy(CrossoverStrategy):
    def __call__(
        self,
        individual1: BoardIndividual,
        individual2: BoardIndividual,
        random_seed: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[BoardIndividual, BoardIndividual]:
        """Apply one random crossover function to the two individuals. The probability of each crossover function
        is proportional to its probability attribute.

        Args:
            individual1 (BoardIndividual): The first individual.
            individual2 (BoardIndividual): The second individual.
            random_seed (Optional[int]): The random seed to use.
            *args: Additional arguments to pass to the crossover functions.
            **kwargs: Additional keyword arguments to pass to the crossover functions.

        Returns:
            Tuple[BoardIndividual, BoardIndividual]: The mated individuals.
        """
        if random_seed:
            self.random_state = np.random.default_rng(random_seed)
        probabilities = [
            crossover_function.probability for crossover_function in self.crossover_functions
        ]
        crossover_function = self.random_state.choice(self.crossover_functions, p=probabilities)

        return crossover_function(individual1, individual2, *args, **kwargs)

    def analyze_crossover_effects(
        self, population: List[BoardIndividual], print_update: bool = False
    ) -> None:
        return

    def print_crossover_probability_history(self) -> None:
        for crossover_function in self.crossover_functions:
            print(f"{crossover_function.function.__name__}: {crossover_function.probability}")


class NRandomCrossoverFunctionsStrategy(CrossoverStrategy):
    def __init__(
        self,
        crossover_functions: List[CrossoverFunction],
        num_crossover_functions: int,
        _random_state: Optional[np.random.Generator] = None,
    ):
        """Apply "num_crossover_functions" random crossover functions to the two individuals. The probability of each
        crossover function is proportional to its probability attribute.

        Args:
            crossover_functions (List[CrossoverFunction]): A **pointer** to the list of crossover functions.
                This means that the list of crossover functions can be modified externally and the changes
                will be reflected here.
            num_crossover_functions (int): The number of crossover functions to apply.
            _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.
        """
        super().__init__(crossover_functions, _random_state)
        self.num_crossover_functions = num_crossover_functions

    def __call__(
        self,
        individual1: BoardIndividual,
        individual2: BoardIndividual,
        random_seed: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[BoardIndividual, BoardIndividual]:
        """Apply "num_crossover_functions" random crossover functions to the two individuals. The probability of each
        crossover function is proportional to its probability attribute.

        Args:
            individual1 (BoardIndividual): The first individual.
            individual2 (BoardIndividual): The second individual.
            random_seed (Optional[int]): The random seed to use.
            *args: Additional arguments to pass to the crossover functions.
            **kwargs: Additional keyword arguments to pass to the crossover functions.

        Returns:
            Tuple[BoardIndividual, BoardIndividual]: The mated individuals.
        """
        if random_seed:
            self.random_state = np.random.default_rng(random_seed)
        probabilities = [
            crossover_function.probability for crossover_function in self.crossover_functions
        ]
        crossover_functions = self.random_state.choice(
            self.crossover_functions,
            size=self.num_crossover_functions,
            p=probabilities,
            replace=False,
        )

        for crossover_function in crossover_functions:
            individual1, individual2 = crossover_function(individual1, individual2, *args, **kwargs)

        return individual1, individual2

    def analyze_crossover_effects(
        self, population: List[BoardIndividual], print_update: bool = False
    ) -> None:
        return

    def print_crossover_probability_history(self) -> None:
        for crossover_function in self.crossover_functions:
            print(f"{crossover_function.function.__name__}: {crossover_function.probability}")


class DynamicCrossoverFunctionStrategy(CrossoverStrategy):
    def __init__(
        self,
        crossover_functions: List[CrossoverFunction],
        minimum_probability: float,
        _random_state: Optional[np.random.Generator] = None,
    ):
        """Dynamically adapt the probability of each crossover function based on the effect which it has on the
        fitness of the population.

        Args:
            crossover_functions (List[CrossoverFunction]): A **pointer** to the list of crossover functions.
                This means that the list of crossover functions can be modified externally and the changes
                will be reflected here.
            minimum_probability (float): The minimum probability of each crossover function.
            _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.
        """
        super().__init__(crossover_functions, _random_state)
        self.minimum_probability = minimum_probability
        self.crossover_cache: Set[Tuple(BoardIndividual, BoardIndividual)] = set()
        self._history: Dict[CrossoverFunction, List[float]] = {}

    def __call__(
        self,
        individual1: BoardIndividual,
        individual2: BoardIndividual,
        random_seed: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[BoardIndividual, BoardIndividual]:
        """Apply one random crossover function to the two individuals. The probability of each crossover function
        is proportional to its probability attribute.

        Args:
            individual1 (BoardIndividual): The first individual.
            individual2 (BoardIndividual): The second individual.
            random_seed (Optional[int]): The random seed to use.
            *args: Additional arguments to pass to the crossover functions.
            **kwargs: Additional keyword arguments to pass to the crossover functions.

        Returns:
            Tuple[BoardIndividual, BoardIndividual]: The mated individuals.
        """
        if random_seed:
            self.random_state = np.random.default_rng(random_seed)

        probabilities = [
            crossover_function.probability for crossover_function in self.crossover_functions
        ]
        crossover_function = self.random_state.choice(self.crossover_functions, p=probabilities)

        parent_fitness1, parent_fitness2 = individual1.fitness, individual2.fitness

        mated1, mated2 = crossover_function(individual1, individual2, *args, **kwargs)

        # Store the parent's fitness and the crossover function used
        mated1.custom_data["crossover_function"] = crossover_function
        mated1.custom_data["crossover_parent_fitness"] = parent_fitness1
        mated2.custom_data["crossover_function"] = crossover_function
        mated2.custom_data["crossover_parent_fitness"] = parent_fitness2
        self.crossover_cache.add((mated1, mated2))

        return mated1, mated2

    def analyze_crossover_effects(
        self, population: List[BoardIndividual], print_update: bool = False
    ) -> None:
        """Analyzes the effects of the crossover functions on the population and dynamically adjusts the probability
        of each crossover function.

        Args:
            population (List[BoardIndividual]): The population.
        """
        # Initialize the crossover stats
        crossover_stats: Dict[Crossover, List[float]] = {
            crossover_function: [] for crossover_function in self.crossover_functions
        }

        # Gather the crossover stats
        for mated1, mated2 in self.crossover_cache:
            crossover_function = mated1.custom_data["crossover_function"]
            parent_fitness1 = mated1.custom_data["crossover_parent_fitness"]
            parent_fitness2 = mated2.custom_data["crossover_parent_fitness"]
            mated_fitness1 = mated1.fitness
            mated_fitness2 = mated2.fitness

            fitnesses_sorted = sorted(
                [parent_fitness1, parent_fitness2, mated_fitness1, mated_fitness2]
            )

            crossover_stats[crossover_function].append(
                fitnesses_sorted[-1] + fitnesses_sorted[-2] - parent_fitness1 - parent_fitness2
            )

            # Delete the custom data
            del mated1.custom_data["crossover_function"]
            del mated1.custom_data["crossover_parent_fitness"]
            del mated2.custom_data["crossover_function"]
            del mated2.custom_data["crossover_parent_fitness"]

        # Compute the progress values
        progress_values: Dict[CrossoverFunction, float] = {}
        for crossover_function, fitnesses in crossover_stats.items():
            # Some crossover functions may not have been used, so we need to handle that case
            progress_values[crossover_function] = np.mean(fitnesses) if len(fitnesses) > 0 else 0

        # Update the crossover weights. This formula ensures that the sum of all weights is 1 and that
        # all weights are at least "minimum_probability" large
        probabilities: Dict[CrossoverFunction, float] = {}
        progress_value_sum = sum(progress_values.values())
        if progress_value_sum == 0:
            # If all progress values are 0, then we just set all probabilities to equal probability
            for crossover_function in self.crossover_functions:
                probabilities[crossover_function] = 1 / len(self.crossover_functions)
                crossover_function.probability = probabilities[crossover_function]
        else:
            for crossover_function, progress_value in progress_values.items():
                probabilities[crossover_function] = (progress_value / progress_value_sum) * (
                    1 - len(self.crossover_functions) * self.minimum_probability
                ) + self.minimum_probability
                crossover_function.probability = probabilities[crossover_function]

        # Update the history
        for crossover_function, probability in probabilities.items():
            if crossover_function not in self._history:
                self._history[crossover_function] = []
            self._history[crossover_function].append(probability)

        if print_update:
            for crossover_function, progress_value in progress_values.items():
                logging.info(
                    f"{crossover_function.function.__name__}: Progress value: {progress_value:.4f},"
                    f" Probability {probability:.4f}"
                )

        # Clear the crossover cache
        self.crossover_cache.clear()

    def print_crossover_probability_history(self) -> None:
        for crossover_function in self.crossover_functions:
            print(f"{crossover_function.function.__name__}: {self._history[crossover_function]}")


def get_crossover_strategy(
    crossover_strategy: str,
    crossover_functions: List[CrossoverFunction],
    *args: Any,
    **kwargs: Any,
) -> CrossoverStrategy:
    """Get a crossover strategy.

    Args:
        crossover_strategy (str): The name of the crossover strategy to use. Must be one of
            ["all", "one_random", "n_random", "dynamic"] where "all" applies all crossover functions, "one_random" applies one
            random crossover function, and "n_random" applies "num_crossover_functions" random selection functions.
        crossover_functions (List[CrossoverFunction]): A **pointer** to the list of crossover functions.
            This means that the list of crossover functions can be modified externally and the changes
            will be reflected here.
        **kwargs: Additional keyword arguments to pass to the crossover strategy.

    Returns:
        CrossoverStrategy: The crossover strategy.

    Raises:
        ValueError: If "crossover_strategy" is not one of ["all", "one_random", "n_random", "dynamic"].
    """
    if crossover_strategy == "all":
        return AllCrossoverFunctionsStrategy(crossover_functions)
    elif crossover_strategy == "one_random":
        return OneRandomCrossoverFunctionStrategy(crossover_functions)
    elif crossover_strategy == "n_random":
        return NRandomCrossoverFunctionsStrategy(
            crossover_functions, kwargs.get("num_crossover_functions", 1)
        )
    elif crossover_strategy == "dynamic":
        return DynamicCrossoverFunctionStrategy(
            crossover_functions, kwargs.get("minimum_probability", 0.01)
        )
    else:
        raise ValueError(
            f"Invalid crossover strategy '{crossover_strategy}'. Must be one of "
            "['all', 'one_random', 'n_random']"
        )


class Crossover:
    def __init__(
        self,
        crossover_strategy: str = "all",
        num_crossover_functions: Optional[int] = None,
        minimum_probability: float = 0.01,
        _random_state: Optional[np.random.Generator] = None,
    ):
        """Crossover class that can be used to perform crossover on a population.

        Args:
            crossover_strategy (str, optional): The strategy to use for the crossover. Must be one of
                ["all", "one_random", "n_random"] where "all" applies all crossover functions, "one_random" applies one
                random crossover function, and "n_random" applies "num_crossover_functions" random selection functions.
                Defaults to "all".
            num_crossover_functions (Optional[int], optional): The number of crossover functions to use if
                "crossover_strategy" is "n_random". Defaults to None.
            minimum_probability (float, optional): The minimum probability of a crossover function if the "dynamic"
                crossover strategy is used. Defaults to 0.01.
            _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.
        """
        self.num_crossover_functions = num_crossover_functions
        self.minimum_probability = minimum_probability
        if crossover_strategy == "n_random":
            assert self.num_crossover_functions is not None, (
                "Must specify the number of crossover functions to use if crossover strategy is"
                " 'n_random'."
            )

        elif crossover_strategy in ["all", "dynamic"]:
            assert (
                minimum_probability is not None
            ), "Must specify the minimum probability if the crossover strategy is 'dynamic'"

        self.crossover_functions: List[CrossoverFunction] = []
        self.crossover_strategy = get_crossover_strategy(
            crossover_strategy,
            self.crossover_functions,
            num_crossover_functions=num_crossover_functions,
            minimum_probability=minimum_probability,
        )
        self.random_state = get_random_state(_random_state)

    def register_crossover_function(
        self,
        functions: Union[
            Callable[[Individual, Individual, Any], Tuple[Individual, Individual]],
            List[Callable[[Individual, Individual, Any], Tuple[Individual, Individual]]],
        ],
        probability: float = 1.0,
        retries: int = 0,
        check_game_not_over: bool = False,
        clear_fitness_values: bool = True,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> None:
        """Registers a crossover function.

        Args:
            crossover_function (Union[
                Callable[[Individual, Individual, Any], Tuple[Individual, Individual]],
                List[Callable[[Individual, Individual, Any], Tuple[Individual, Individual]]]]):
                The crossover function to register. Can be a single function or a list of functions.
            clear_fitness (bool, optional): Whether to clear the fitness of the individuals after the crossover. Defaults to True.
            args (List[Any], optional): The arguments to pass to the crossover function. Defaults to [].
            kwargs (Dict[str, Any], optional): The keyword arguments to pass to the crossover function. Defaults to {}.
        """
        if not isinstance(functions, list):
            functions = [functions]

        for function in functions:
            self.crossover_functions.append(
                CrossoverFunction(
                    function,
                    probability=probability,
                    retries=retries,
                    check_game_not_over=check_game_not_over,
                    clear_fitness_values=clear_fitness_values,
                    _random_state=self.random_state,
                    *args,
                    **kwargs,
                )
            )

    def multiply_probabilities(self, factor: float) -> None:
        """Sets the probability of all crossover functions.

        Args:
            probability (float): The probability to set.
        """
        for crossover_function in self.crossover_functions:
            if crossover_function.probability * factor >= self.minimum_probability:
                crossover_function.probability *= factor

    def __call__(
        self,
        individual_tuple: Tuple[Individual, Individual],
        random_seed: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Individual, Individual]:
        """Calls the crossover function on the two individuals according to the crossover strategy.

        Args:
            individual_tuple (Tuple[Individual, Individual]): The two individuals to mate.
            random_seed (Optional[int], optional): The random seed to use. Defaults to None.

        Returns:
            Tuple[Individual, Individual]: The two individuals after the crossover.
        """
        if random_seed:
            self.random_state = np.random.default_rng(random_seed)
        individual1, individual2 = individual_tuple
        return self.crossover_strategy(
            individual1, individual2, random_seed=random_seed, *args, **kwargs
        )

    def analyze_crossover_effects(
        self, population: List[BoardIndividual], print_update: bool = False
    ) -> None:
        """Analyzes the effects of the crossover functions on the population.

        Args:
            population (List[BoardIndividual]): The population to analyze.
        """
        self.crossover_strategy.analyze_crossover_effects(population, print_update=print_update)

    def print_crossover_probability_history(self) -> None:
        """Prints the crossover probability history."""
        self.crossover_strategy.print_crossover_probability_history()


if __name__ == "__main__":
    # Configure logging to show debug messages
    logging.basicConfig(level=logging.DEBUG)

    board1 = chess.Board("8/1p6/1p6/pPp1p1n1/P1P1P1k1/1K1P4/8/2B5 w - - 110 118")
    board2 = chess.Board("r3qb1r/pppbk1p1/2np2np/4p2Q/2BPP3/2P5/PP3PPP/RNB2RK1 w - - 4 11")
    print_side_by_side(board1, board2)
    crossover_one_eighth_board(board1, board2, ensure_single_kings=True)
    print_side_by_side(board1, board2)
