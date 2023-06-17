import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import chess
import chess.svg
import numpy as np
import yaml


def log_time(start_time: float, message: str = ""):
    """Log the time since the start of the program.

    Args:
        start_time (float): The time the program started.
    """
    end_time = time.time()
    time_elapsed = end_time - start_time
    logging.info(f"Time {message}: {time_elapsed:.2f} seconds.")


def get_task_result_handler(
    logger: logging.Logger,
    message: str,
    message_args: Tuple[Any, ...] = (),
) -> Any:
    def handle_task_result(
        task: asyncio.Task,
        *,
        logger: logging.Logger,
        message: str,
        message_args: Tuple[Any, ...] = (),
    ) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass  # Task cancellation should not be logged as an error.
        # Ad the pylint ignore: we want to handle all exceptions here so that the result of the task
        # is properly logged. There is no point re-raising the exception in this callback.
        except Exception:  # pylint: disable=broad-except
            logger.exception(message, *message_args)

    return lambda task: handle_task_result(
        task, logger=logger, message=message, message_args=message_args
    )


def get_random_state(
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> np.random.Generator:
    """Get a random state. Use the provided random state if it is not None, otherwise use the default random state.

    Args:
        random_state (Optional[Union[int, np.random.Generator]], optional): An optional random state or a seed. Defaults to None.

    Returns:
        np.random.Generator: The random state.
    """
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, int):
        return np.random.default_rng(random_state)
    return random_state


class FakeAsyncResult:
    def __init__(self, result) -> None:
        self._result = result

    def get(self, timeout=None):
        return self._result

    def wait(self, timeout=None):
        return

    def ready(self):
        return True

    def successful(self):
        return True


class FakePool:
    def __init__(self) -> None:
        self._doing_work = False
        self._processes = 8  # This is the default value for multiprocessing.Pool.

    def apply(self, func: Callable[[Any], Any], *args: Any, **kwds: Any) -> Any:
        self._doing_work = True
        result = func(*args, **kwds)
        self._doing_work = False
        return result

    def apply_async(
        self,
        func: Callable[[Any], Any],
        *args: Any,
        callback: Optional[Callable[[Any], Any]] = None,
        error_callback: Optional[Callable[[Any], Any]] = None,
        **kwds: Any,
    ) -> FakeAsyncResult:
        self._doing_work = True
        try:
            result = func(*args, **kwds)
            if callback is not None:
                callback(result)
        except Exception as e:
            result = None
            if error_callback is not None:
                error_callback(e)
        self._doing_work = False
        return FakeAsyncResult(result)

    def map(
        self, func: Callable[[Any], Any], iterable: Iterable, chunksize: Optional[int] = None
    ) -> List[Any]:
        self._doing_work = True
        result = [func(item) for item in iterable]
        self._doing_work = False
        return result

    def map_async(
        self,
        func,
        iterable: Iterable,
        chunksize: Optional[int] = None,
        callback: Optional[Callable[[Any], Any]] = None,
        error_callback: Optional[Callable[[Any], Any]] = None,
    ) -> FakeAsyncResult:
        self._doing_work = True
        try:
            result = [func(item) for item in iterable]
            if callback is not None:
                callback(result)
        except Exception as e:
            result = None
            if error_callback is not None:
                error_callback(e)
        self._doing_work = False
        return FakeAsyncResult(result)

    def imap(
        self, func: Callable[[Any], Any], iterable: Iterable, chunksize: Optional[int] = None
    ) -> Iterable[Any]:
        return self.map(func, iterable, chunksize)

    def imap_unordered(
        self, func: Callable[[Any], Any], iterable: Iterable, chunksize: Optional[int] = None
    ) -> Iterable[Any]:
        return self.map(func, iterable, chunksize)

    def starmap(
        self,
        func: Callable[[Any], Any],
        iterable: Iterable[Iterable],
        chunksize: Optional[int] = None,
    ) -> List[Any]:
        self._doing_work = True
        result = [func(*item) for item in iterable]
        self._doing_work = False
        return result

    def starmap_async(
        self,
        func: Callable[[Any], Any],
        iterable: Iterable[Iterable],
        chunksize: Optional[int] = None,
        callback: Optional[Callable[[Any], Any]] = None,
        error_callback: Optional[Callable[[Any], Any]] = None,
    ) -> FakeAsyncResult:
        self._doing_work = True
        try:
            result = [func(*item) for item in iterable]
            if callback is not None:
                callback(result)
        except Exception as e:
            result = None
            if error_callback is not None:
                error_callback(e)
        self._doing_work = False
        return FakeAsyncResult(result)

    def close(self):
        return

    def terminate(self):
        return

    def join(self):
        while self._doing_work:
            time.sleep(0.1)


class LoaderWithInclude(yaml.SafeLoader):
    def __init__(self, stream):

        self._root: Path = Path(stream.name).parent

        super(LoaderWithInclude, self).__init__(stream)

    def include(self, node):
        include_path = Path(self.construct_scalar(node))
        if include_path.exists():
            filename = include_path
        else:
            filename = self._root / include_path

        assert filename.exists(), f"File {filename} does not exist!"

        with open(filename, "r") as f:
            return yaml.load(f, LoaderWithInclude)


LoaderWithInclude.add_constructor("!include", LoaderWithInclude.include)


if __name__ == "__main__":
    fen = "rnbqkbnr/pppp1ppp/8/3Pp3/5B2/6N1/PPP1PPPP/RNBQK2R w Kq e6 0 1"
    fen = "8/1p3r2/P1pK4/P3Pnp1/PpnR3N/Q4bqB/p2Bp3/Nk6 w - - 70 50"
    board = chess.Board(fen)
