from typing import Optional

import chess


class BoardGenerator:
    def __iter__(self):
        return self

    def __next__(self) -> chess.Board:
        return self.next()

    def next(self) -> chess.Board:
        raise NotImplementedError()
