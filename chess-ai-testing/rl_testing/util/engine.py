import asyncio
import typing
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
from chess.engine import EngineTerminatedError, InfoDict, Limit

from rl_testing.engine_generators.relaxed_uci_protocol import (
    ExtendedAnalysisResult,
    RelaxedUciProtocol,
)


class MoveStat:
    __slots__ = (
        "move",
        "makes_terminal",
        "visits",
        "policy",
        "win_minus_loss",
        "draw",
        "moves_left_estimated",
        "q_value",
        "u_value",
        "s_value",
        "v_value",
    )

    def __init__(
        self,
        move: chess.Move,
        makes_terminal: bool,
        visits: int,
        policy: float,
        win_minus_loss: Optional[float],
        draw: Optional[float],
        moves_left_estimated: Optional[float],
        q_value: float,
        u_value: float,
        s_value: float,
        v_value: Optional[float],
    ) -> None:
        self.move = move
        self.makes_terminal = makes_terminal
        self.visits = visits
        self.policy = policy
        self.win_minus_loss = win_minus_loss
        self.draw = draw
        self.moves_left_estimated = moves_left_estimated
        self.q_value = q_value
        self.u_value = u_value
        self.s_value = s_value
        self.v_value = v_value

    def __str__(self):
        return (
            f"MoveStat(move={self.move}, "
            f"makes_terminal={self.makes_terminal}, "
            f"visits={self.visits}, "
            f"policy={self.policy}, "
            f"win_minus_loss={self.win_minus_loss}, "
            f"draw={self.draw}, "
            f"moves_left_estimated={self.moves_left_estimated}, "
            f"q_value={self.q_value}, "
            f"u_value={self.u_value}, "
            f"s_value={self.s_value}, "
            f"v_value={self.v_value})"
        )

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_string(string: str) -> "MoveStat":
        """
        Create a MoveStat from a string which has been generated from the __str__ method
        of this class.
        """
        string = string.strip()

        # Remove the class name and the brackets
        string = string[len("MoveStat(") : -1]

        # Split the string into the different attributes
        attributes = string.split(", ")

        # Create a dictionary of the attributes
        attributes_dict = {}

        # Convert the values to the correct type and add them to the dictionary
        for attribute in attributes:
            name, value = attribute.split("=")
            if value == "None":
                value = None
            elif value in ["True", "False"]:
                value = value == "True"
            elif name == "move":
                value = chess.Move.from_uci(value)
            else:
                value = float(value)
            attributes_dict[name] = value

        return MoveStat(**attributes_dict)


class NodeStat:
    __slots__ = (
        "move_stats",
        "is_terminal",
        "visits",
        "policy",
        "win_minus_loss",
        "draw",
        "moves_left_estimated",
        "q_value",
        "v_value",
        "best_move",
    )

    def __init__(
        self,
        move_stats: List[MoveStat],
        is_terminal: bool,
        visits: int,
        policy: float,
        win_minus_loss: float,
        draw: float,
        moves_left_estimated: float,
        q_value: float,
        v_value: float,
    ) -> None:
        # Store the move stats as dictionary indexed by the move
        self.move_stats = {move_stats[i].move: move_stats[i] for i in range(len(move_stats))}
        self.is_terminal = is_terminal
        self.visits = visits
        self.policy = policy
        self.win_minus_loss = win_minus_loss
        self.draw = draw
        self.moves_left_estimated = moves_left_estimated
        self.q_value = q_value
        self.v_value = v_value

        # Compute the best move
        self.best_move = max(self.move_stats, key=lambda move: self.move_stats[move].visits)

    def __str__(self):
        return (
            f"NodeStat(is_terminal={self.is_terminal}, "
            f"visits={self.visits}, "
            f"policy={self.policy}, "
            f"win_minus_loss={self.win_minus_loss}, "
            f"draw={self.draw}, "
            f"moves_left_estimated={self.moves_left_estimated}, "
            f"q_value={self.q_value}, "
            f"v_value={self.v_value})"
        )

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_string(string: str, move_stats: List[MoveStat]) -> "NodeStat":
        """
        Create a NodeStat from a string which has been generated from the __str__ method
        of this class. The move_stats argument is a list of MoveStats which need to be parsed
        before the NodeStat can be created.
        """
        string = string.strip()

        # Remove the "NodeStat(" and ")" from the string
        string = string[len("NodeStat(") : -1]

        # Split the string into a list of key-value pairs
        key_value_pairs = string.split(", ")

        # Split the key-value pairs into keys and values
        keys = [key_value_pair.split("=")[0] for key_value_pair in key_value_pairs]
        values = [key_value_pair.split("=")[1] for key_value_pair in key_value_pairs]

        attribute_dict = {}

        # Convert the values to the correct type and add them to the dictionary
        for index, value in enumerate(values):
            if value == "None":
                values[index] = None
            elif keys[index] in [
                "policy",
                "win_minus_loss",
                "draw",
                "moves_left_estimated",
                "q_value",
                "v_value",
            ]:
                values[index] = float(value)
            elif keys[index] == "is_terminal":
                values[index] = value == "True"
            elif keys[index] == "visits":
                values[index] = int(float(value))
            else:
                raise ValueError(f"Unknown key: {keys[index]}")

            attribute_dict[keys[index]] = values[index]

        # Create the NodeStat
        return NodeStat(move_stats, **attribute_dict)


class InfoParser:
    name_mapping = {
        "move": "move",
        "N": "visits",
        "P": "policy",
        "WL": "win_minus_loss",
        "D": "draw",
        "M": "moves_left_estimated",
        "Q": "q_value",
        "U": "u_value",
        "S": "s_value",
        "V": "v_value",
        "T": "makes_terminal",
    }

    def __init__(self) -> None:
        self.move_stats_temp: List[MoveStat] = []
        self.node_stats: List[NodeStat] = []
        self.finished: bool = False

    def _extract_data(self, line: str) -> Dict[str, Any]:
        """
        First, preprocess the line to make it easier to parse. In particular, make sure that
        all the data is in the form "(key:value)". Then extract the data and return it as a
        dictionary.
        """
        bracket_open = False
        fixed_line = []
        # First make sure that all the data is in the form "(...)"
        for index, char in enumerate(line):
            if not str.isspace(char) and char != "(" and not bracket_open:
                fixed_line.append("(")
                bracket_open = True
            elif not str.isspace(char) and char == "(" and bracket_open:
                fixed_line.append(")")
            if char == ")":
                bracket_open = False
            elif char == "(":
                bracket_open = True

            fixed_line.append(char)

        if bracket_open:
            fixed_line.append(")")

        # Remove all white spaces
        fixed_line = "".join(fixed_line)
        fixed_line = fixed_line.replace(" ", "")

        # Extract the data into a list
        data_tokens = fixed_line.split(")(")
        data_tokens[0] = data_tokens[0][1:]  # Remove the first "("
        data_tokens[-1] = data_tokens[-1][:-1]  # Remove the last ")"

        # Add a name to the first token
        data_tokens[0] = "move:" + data_tokens[0]

        # If the last token is a terminal token, add a value to it
        if data_tokens[-1] == "T":
            data_tokens[-1] = "T:True"
        else:
            data_tokens.append("T:False")

        # Drop all remaining tokens which don't contain a name or a value
        data_tokens = [token for token in data_tokens if ":" in token]
        result = {}
        for key_value in data_tokens:
            key, value = key_value.split(":")
            if key == "move":
                if value == "node":
                    continue
                result[key] = chess.Move.from_uci(value)
            elif key == "T":
                result[key] = value == "True"
            elif value.endswith("%"):
                result[key] = float(value[:-1]) / 100
            elif "-.-" in value:
                result[key] = None
            else:
                result[key] = float(value)
        return result

    def parse_line(self, line: str) -> None:
        is_node_stat = "node" in line

        print(f"Line: {line}")

        # Extract the data from the line
        data = self._extract_data(line)

        # Replace the keys with the names we want
        data = {self.name_mapping[key]: value for key, value in data.items()}

        if is_node_stat:
            data["is_terminal"] = data.pop("makes_terminal")
            self.node_stats.append(NodeStat(move_stats=self.move_stats_temp, **data))
            self.move_stats_temp = []
        else:
            self.move_stats_temp.append(MoveStat(**data))


@typing.overload
async def engine_analyse(
    engine: RelaxedUciProtocol,
    board: chess.Board,
    limit: Limit,
    intermediate_info: bool,
    **kwargs: Any,
) -> Union[InfoDict, Tuple[InfoDict, List[NodeStat]]]:
    ...


@typing.overload
async def engine_analyse(
    engine: RelaxedUciProtocol,
    board: chess.Board,
    limit: Limit,
    intermediate_info: bool,
    multipv: int,
    **kwargs: Any,
) -> Union[List[InfoDict], Tuple[List[InfoDict], List[NodeStat]]]:
    ...


class AsyncTimedIterable:
    """
    This code is taken from the following brilliant Medium article by Dmitry Pankrashov:
    https://medium.com/@dmitry8912/implementing-timeouts-in-pythons-asynchronous-generators-f7cbaa6dc1e9
    """

    def __init__(self, iterable, timeout=0):
        class AsyncTimedIterator:
            def __init__(self):
                self._iterator = iterable.__aiter__()

            async def __anext__(self):
                try:
                    result = await asyncio.wait_for(self._iterator.__anext__(), timeout)
                    # if you want to stop the iteration just raise StopAsyncIteration using some conditions (when the last chunk arrives, for example)
                    if not result:
                        raise StopAsyncIteration
                    return result
                except asyncio.TimeoutError as e:
                    raise e

        self._factory = AsyncTimedIterator

    def __aiter__(self):
        return self._factory()


async def engine_analyse(
    engine: RelaxedUciProtocol,
    board: chess.Board,
    limit: Limit,
    intermediate_info: bool = False,
    multipv: Optional[int] = None,
    **kwargs: Any,
) -> Union[
    InfoDict,
    Tuple[InfoDict, List[NodeStat]],
    List[InfoDict],
    Tuple[List[InfoDict], List[NodeStat]],
]:
    # Gather all intermediate info from the analysis
    if intermediate_info:
        info_parser = InfoParser()
        try:
            with await engine.analysis(board, limit, multipv=multipv, **kwargs) as analysis:
                timed_analysis = AsyncTimedIterable(analysis, timeout=5)
                async for info in timed_analysis:
                    if "string" in info:
                        info_parser.parse_line(info["string"])
        except asyncio.TimeoutError:
            raise EngineTerminatedError

        node_stats = info_parser.node_stats

        # The last node_stat might be present twice. Remove it if it is.
        if len(node_stats) > 1 and node_stats[-1].visits == node_stats[-2].visits:
            node_stats.pop()

    # Gather only the final info from the analysis
    else:
        analysis = await engine.analysis(board, limit, multipv=multipv, **kwargs)

    # Wait for the analysis to finish and return the info object.
    with analysis:
        try:
            await asyncio.wait_for(analysis.wait(), 30)
        except asyncio.TimeoutError:
            raise EngineTerminatedError

    analysis_result = analysis.info if multipv is None else analysis.multipv

    if intermediate_info:
        return analysis_result, node_stats

    return analysis_result
