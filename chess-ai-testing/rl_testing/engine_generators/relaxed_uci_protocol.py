import asyncio
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import chess
from chess import InvalidMoveError, Move
from chess.engine import (
    INFO_ALL,
    INFO_CURRLINE,
    INFO_PV,
    INFO_REFUTATION,
    INFO_SCORE,
    LOGGER,
    UCI_REGEX,
    AnalysisResult,
    BaseCommand,
    BestMove,
    ConfigMapping,
    Cp,
    EngineError,
    Info,
    InfoDict,
    Limit,
    Mate,
    PovScore,
    PovWdl,
    UciProtocol,
    Wdl,
)

from rl_testing.mcts.tree_parser import NodeInfo, OneNodeParser, TreeInfo, TreeParser


def parse_uci_relaxed(self, uci: str) -> Move:
    """
    Parses the given move in UCI notation.

    Supports both Chess960 and standard UCI notation.

    The returned move is guaranteed to be either legal or a null move.

    :raises: :exc:`ValueError` if the move is invalid or illegal in the
        current position (but not a null move).
    """
    try:
        move = Move.from_uci(uci)
    except InvalidMoveError:
        move = Move.from_uci("0000")

    if not move:
        return move

    move = self._to_chess960(move)
    move = self._from_chess960(
        self.chess960, move.from_square, move.to_square, move.promotion, move.drop
    )

    return move


def _parse_uci_info_relaxed(
    arg: str, root_board: chess.Board, selector: Info = INFO_ALL
) -> InfoDict:
    info: InfoDict = {}
    if not selector:
        return info

    tokens = arg.split(" ")
    while tokens:
        parameter = tokens.pop(0)

        if parameter == "string":
            info["string"] = " ".join(tokens)
            break
        elif parameter in [
            "depth",
            "seldepth",
            "nodes",
            "multipv",
            "currmovenumber",
            "hashfull",
            "nps",
            "tbhits",
            "cpuload",
        ]:
            try:
                info[parameter] = int(tokens.pop(0))  # type: ignore
            except (ValueError, IndexError):
                LOGGER.error("Exception parsing %s from info: %r", parameter, arg)
        elif parameter == "time":
            try:
                info["time"] = int(tokens.pop(0)) / 1000.0
            except (ValueError, IndexError):
                LOGGER.error("Exception parsing %s from info: %r", parameter, arg)
        elif parameter == "ebf":
            try:
                info["ebf"] = float(tokens.pop(0))
            except (ValueError, IndexError):
                LOGGER.error("Exception parsing %s from info: %r", parameter, arg)
        elif parameter == "score" and selector & INFO_SCORE:
            try:
                kind = tokens.pop(0)
                value = tokens.pop(0)
                if tokens and tokens[0] in ["lowerbound", "upperbound"]:
                    info[tokens.pop(0)] = True  # type: ignore
                if kind == "cp":
                    info["score"] = PovScore(Cp(int(value)), root_board.turn)
                elif kind == "mate":
                    info["score"] = PovScore(Mate(int(value)), root_board.turn)
                else:
                    LOGGER.error(
                        "Unknown score kind %r in info (expected cp or mate): %r",
                        kind,
                        arg,
                    )
            except (ValueError, IndexError):
                LOGGER.error("Exception parsing score from info: %r", arg)
        elif parameter == "currmove":
            try:
                info["currmove"] = chess.Move.from_uci(tokens.pop(0))
            except (ValueError, IndexError):
                LOGGER.error("Exception parsing currmove from info: %r", arg)
        elif parameter == "currline" and selector & INFO_CURRLINE:
            try:
                if "currline" not in info:
                    info["currline"] = {}

                cpunr = int(tokens.pop(0))
                currline: List[chess.Move] = []
                info["currline"][cpunr] = currline

                board = root_board.copy(stack=False)
                while tokens and UCI_REGEX.match(tokens[0]):
                    currline.append(board.push_uci(tokens.pop(0)))
            except (ValueError, IndexError):
                LOGGER.error(
                    "Exception parsing currline from info: %r, position at root: %s",
                    arg,
                    root_board.fen(),
                )
        elif parameter == "refutation" and selector & INFO_REFUTATION:
            try:
                if "refutation" not in info:
                    info["refutation"] = {}

                board = root_board.copy(stack=False)
                refuted = board.push_uci(tokens.pop(0))

                refuted_by: List[chess.Move] = []
                info["refutation"][refuted] = refuted_by

                while tokens and UCI_REGEX.match(tokens[0]):
                    refuted_by.append(board.push_uci(tokens.pop(0)))
            except (ValueError, IndexError):
                LOGGER.error(
                    "Exception parsing refutation from info: %r, position at root: %s",
                    arg,
                    root_board.fen(),
                )
        elif parameter == "pv" and selector & INFO_PV:
            try:
                pv: List[chess.Move] = []
                info["pv"] = pv
                board = root_board.copy(stack=False)
                while tokens and UCI_REGEX.match(tokens[0]):
                    pv.append(board.push_uci(tokens.pop(0)))
            except (ValueError, IndexError):
                LOGGER.error(
                    "Exception parsing pv from info: %r, position at root: %s",
                    arg,
                    root_board.fen(),
                )
        elif parameter == "wdl":
            try:
                info["wdl"] = PovWdl(
                    Wdl(int(tokens.pop(0)), int(tokens.pop(0)), int(tokens.pop(0))),
                    root_board.turn,
                )
            except (ValueError, IndexError):
                LOGGER.error("Exception parsing wdl from info: %r", arg)

    return info


def _parse_uci_bestmove_relaxed(board: chess.Board, args: str) -> BestMove:
    tokens = args.split()

    move = None
    ponder = None

    if tokens and tokens[0] not in ["(none)", "NULL"]:
        failed = False
        try:
            # AnMon 5.75 uses uppercase letters to denote promotion types.
            board_copy = board.copy()
            # Testing it on the copy first to not change the state of the original
            board_copy.push_uci(tokens[0].lower())
            move = board.push_uci(tokens[0].lower())
        except (ValueError, InvalidMoveError) as err:
            failed = True
            print("Illegal move detected!")
            print(err)
            move = parse_uci_relaxed(board, tokens[0].lower())
        try:
            # Houdini 1.5 sends NULL instead of skipping the token.
            if len(tokens) >= 3 and tokens[1] == "ponder" and tokens[2] not in ["(none)", "NULL"]:
                ponder = board.parse_uci(tokens[2].lower())
        except (ValueError, chess.InvalidMoveError):
            LOGGER.exception("Engine sent invalid ponder move")
        finally:
            if not failed:
                board.pop()

    return BestMove(move, ponder), not failed


class ExtendedAnalysisResult(AnalysisResult):
    def __init__(self, stop: Optional[Callable[[], None]] = None):
        super().__init__(stop)
        self.mcts_tree: Optional[TreeInfo] = None
        self.root_and_child_scores: Optional[NodeInfo] = None


class RelaxedUciProtocol(UciProtocol):
    """
    A relaxed implementation of the
    `Universal Chess Interface <https://www.chessprogramming.org/UCI>`_
    protocol.
    """

    async def analysis(
        self,
        board: chess.Board,
        limit: Optional[Limit] = None,
        *,
        multipv: Optional[int] = None,
        game: object = None,
        info: Info = INFO_ALL,
        root_moves: Optional[Iterable[chess.Move]] = None,
        options: ConfigMapping = {},
    ) -> ExtendedAnalysisResult:
        self.invalid_best_move = False

        class UciAnalysisCommand(BaseCommand[UciProtocol, AnalysisResult]):
            def start(self, engine: UciProtocol) -> None:
                self.analysis = ExtendedAnalysisResult(stop=lambda: self.cancel(engine))
                self.tree_parser = TreeParser(self.analysis)
                self.last_node_parser = OneNodeParser(self.analysis)
                self.sent_isready = False

                if "Ponder" in engine.options:
                    engine._setoption("Ponder", False)
                if (
                    "UCI_AnalyseMode" in engine.options
                    and "UCI_AnalyseMode" not in engine.target_config
                    and all(name.lower() != "uci_analysemode" for name in options)
                ):
                    engine._setoption("UCI_AnalyseMode", True)
                if "MultiPV" in engine.options or (multipv and multipv > 1):
                    engine._setoption("MultiPV", 1 if multipv is None else multipv)

                engine._configure(options)

                if engine.first_game or engine.game != game:
                    engine.game = game
                    engine._ucinewgame()
                    self.sent_isready = True
                    engine._isready()
                else:
                    self._readyok(engine)

            def line_received(self, engine: UciProtocol, line: str) -> None:
                # print(line)
                if self.tree_parser.is_line_parsable(line):
                    self.tree_parser.parse_line(line)
                elif line.startswith("info "):
                    self._info(engine, line.split(" ", 1)[1])
                    if self.last_node_parser.is_line_parsable(line):
                        self.last_node_parser.parse_line(line)
                elif line.startswith("bestmove "):
                    self._bestmove(engine, line.split(" ", 1)[1])
                elif line == "readyok" and self.sent_isready:
                    self._readyok(engine)
                else:
                    LOGGER.warning("%s: Unexpected engine output: %r", engine, line)

                # except EngineError as err:
                #    print("ERROR: engine error!")
                #    print(err)
                #    if "illegal uci" in str(err):
                #        self.set_finished()
                #        self.analysis.set_finished()

                #    # self.engine_terminated(self, err)

            def _readyok(self, engine: UciProtocol) -> None:
                self.sent_isready = False
                engine._position(board)

                if limit:
                    engine._go(limit, root_moves=root_moves)
                else:
                    engine._go(Limit(), root_moves=root_moves, infinite=True)

                self.result.set_result(self.analysis)

            def _info(self, engine: UciProtocol, arg: str) -> None:
                self.analysis.post(_parse_uci_info_relaxed(arg, engine.board, info))

            def _bestmove(self2, engine: UciProtocol, arg: str) -> None:
                if not self2.result.done():
                    raise EngineError("was not searching, but engine sent bestmove")

                best, valid = _parse_uci_bestmove_relaxed(engine.board, arg)

                if not valid:
                    self.invalid_best_move = True

                self2.set_finished()
                self2.analysis.set_finished(best)

            def cancel(self, engine: UciProtocol) -> None:
                try:
                    engine.send_line("stop")
                except BrokenPipeError:
                    print("Broken pipe error")

            def engine_terminated(self, engine: UciProtocol, exc: Exception) -> None:
                LOGGER.debug(
                    "%s: Closing analysis because engine has been terminated (error: %s)",
                    engine,
                    exc,
                )
                self.analysis.set_exception(exc)

        return await self.communicate(UciAnalysisCommand)


async def popen_uci_relaxed(
    command: Union[str, List[str]], *, setpgrp: bool = False, **popen_args: Any
) -> Tuple[asyncio.SubprocessTransport, RelaxedUciProtocol]:
    """
    Spawns and initializes a UCI engine.

    :param command: Path of the engine executable, or a list including the
        path and arguments.
    :param setpgrp: Open the engine process in a new process group. This will
        stop signals (such as keyboard interrupts) from propagating from the
        parent process. Defaults to ``False``.
    :param popen_args: Additional arguments for
        `popen <https://docs.python.org/3/library/subprocess.html#popen-constructor>`_.
        Do not set ``stdin``, ``stdout``, ``bufsize`` or
        ``universal_newlines``.

    Returns a subprocess transport and engine protocol pair.
    """
    transport, protocol = await RelaxedUciProtocol.popen(command, setpgrp=setpgrp, **popen_args)
    try:
        await protocol.initialize()
    except:  # noqa: E722
        transport.close()
        raise
    return transport, protocol
