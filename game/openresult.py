from enum import Enum, auto


class OpenResult(int, Enum):
    FAIL = auto()
    WIN = auto()
    NEIGHBOR = auto()
    ISOLATED = auto()
