from typing import Generator, Optional
from collections import deque
import random

from .openresult import OpenResult


class GameBoard:
    # Cells with mines (for hidden board)
    MINE = -1
    # Cells that are not open yet (for visible board)
    CLOSED = -2

    @staticmethod
    @property
    def lower_bound() -> int:
        """Lower bound of values in the game board."""
        return min(GameBoard.MINE, GameBoard.CLOSED)

    upper_bound: int = 9

    def __init__(self, n_rows: int, n_cols: int, n_mines: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_mines = n_mines

    def reset_board(self, seed: Optional[int] = None):
        """Initializes the game board for a new game."""

        random.seed(seed)

        # The hidden game board that contains all information about the mines.
        self._hidden_board = self._get_new_board()
        # The game board that is visible to the player.
        self._visible_board = [
            [GameBoard.CLOSED] * self.n_cols for _ in range(self.n_rows)
        ]
        # The number of cells that is not opened yet.
        self.n_closed = self.n_rows * self.n_cols

    def get_visible_board(self) -> list[list[int]]:
        return self._visible_board

    def render(self) -> str:
        return GameBoard._board_to_string(self._visible_board)

    def print(self, print_hidden: bool = False):
        print("=" * self.n_cols)
        print(
            GameBoard._board_to_string(self._visible_board, colored=[GameBoard.CLOSED])
        )

        print("n_closed =", self.n_closed)

        if print_hidden:
            print("[hidden]")
            print(
                GameBoard._board_to_string(self._hidden_board, colored=[GameBoard.MINE])
            )

    def open(self, x: int, y: int) -> OpenResult:
        """Opens the givin position. Returns the result as enum variable."""

        if self._visible_board[x][y] != GameBoard.CLOSED:
            raise RuntimeError(f"Given position ({x}, {y}) is already opened")

        if self._hidden_board[x][y] == GameBoard.MINE:
            return OpenResult.FAIL

        queue = deque([(x, y)])  # queue for DFS

        while queue:
            qx, qy = queue.popleft()
            if self._visible_board[qx][qy] != GameBoard.CLOSED:
                continue  # already opened

            hidden = self._hidden_board[qx][qy]
            self._visible_board[qx][qy] = hidden  # open
            self.n_closed -= 1
            if hidden != 0:
                continue

            # Open neighboring positions if 0 is found
            for ax, ay in self._around(qx, qy):
                queue.append((ax, ay))

        if self.n_closed <= self.n_mines:
            return OpenResult.WIN
        elif self._is_neighboring(x, y):
            return OpenResult.NEIGHBOR
        else:
            return OpenResult.ISOLATED

    def _get_new_board(self) -> list[list[int]]:
        """Fills the game board with mines and numbers to start a new game."""

        board = [[0] * self.n_cols for _ in range(self.n_rows)]

        # Randomly select the positions of mines
        positions = set()
        while len(positions) < self.n_mines:
            x = random.randrange(0, self.n_rows)
            y = random.randrange(0, self.n_cols)
            positions.add((x, y))

        # Fill mines and numbers
        for mx, my in positions:
            board[mx][my] = GameBoard.MINE

            for ax, ay in self._around(mx, my):
                if board[ax][ay] != GameBoard.MINE:
                    board[ax][ay] += 1

        return board

    def _around(self, x: int, y: int) -> Generator[tuple[int, int], None, None]:
        """Generates coordinates around the given position."""

        DELTAS = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
        for dx, dy in DELTAS:
            nx = x + dx
            ny = y + dy
            if (0 <= nx < self.n_rows) and (0 <= ny < self.n_cols):
                yield nx, ny

    def _is_neighboring(self, x: int, y: int) -> bool:
        """Check if the given position is neighboring to a opened position."""

        for ax, ay in self._around(x, y):
            if self._visible_board[ax][ay] != GameBoard.CLOSED:
                return True
        return False

    @staticmethod
    def _board_to_string(
        board: list[list[int]],
        colored: Optional[list[int]] = None,
    ) -> str:
        output = ""

        for row in board:
            for elem in row:
                if colored is not None and elem in colored:
                    output += f"\033[31m{elem: >3}\033[37m"
                else:
                    output += f"{elem: >3}"
            output += "\n"

        return output
