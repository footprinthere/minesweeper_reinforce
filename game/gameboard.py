from typing import Generator, Optional
from collections import deque
import random

from .openresult import OpenResult


class GameBoard:
    # Cells with mines (for hidden board)
    MINE = -1
    # Cells that are not open yet (for visible board)
    CLOSED = -5

    def __init__(self, n_rows: int, n_cols: int, n_mines: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_mines = n_mines

        self.reset_board()

    def reset_board(self):
        """Initializes the game board for a new game."""

        # The hidden game board that contains all information about the mines.
        self._hidden_board = self._get_new_board()
        # The game board that is visible to the player.
        self._visible_board = [
            [GameBoard.CLOSED] * self.n_cols for _ in range(self.n_rows)
        ]
        # The number of cells that is not opened yet.
        self._n_closed = self.n_rows * self.n_cols

    def get_visible_board(self) -> list[list[int]]:
        return self._visible_board

    def print(self, print_hidden: bool = False):
        print("=" * self.n_cols)
        GameBoard._print_board(self._visible_board, colored=[GameBoard.CLOSED])

        print("n_closed =", self._n_closed)

        if print_hidden:
            print("[hidden]")
            GameBoard._print_board(self._hidden_board, colored=[GameBoard.MINE])

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
                continue    # already opened

            hidden = self._hidden_board[qx][qy]
            self._visible_board[qx][qy] = hidden  # open
            self._n_closed -= 1
            if hidden != 0:
                continue

            # Open neighboring positions if 0 is found
            for ax, ay in self._around(qx, qy):
                queue.append((ax, ay))

        if self._n_closed <= self.n_mines:
            return OpenResult.WIN
        else:
            return OpenResult.OK

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

    @staticmethod
    def _print_board(board: list[list[int]], colored: Optional[list[int]] = None):
        for row in board:
            for elem in row:
                if colored is not None and elem in colored:
                    print(f"\033[31m{elem: >3}\033[37m", end="")
                else:
                    print(f"{elem: >3}", end="")
            print()
