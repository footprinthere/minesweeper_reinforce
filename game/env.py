from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces

from .gameboard import GameBoard
from .openresult import OpenResult


class MineSweeperEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        board_size: tuple[int, int],
        n_mines: int,
        flat_action: bool = False,
    ):
        self.gameboard = GameBoard(*board_size, n_mines)

        # State as an integer matrix of the same shape as the game board
        self.observation_space = spaces.Box(
            low=GameBoard.lower_bound,
            high=GameBoard.upper_bound,
            shape=board_size,
            dtpye=int,
        )

        self.flat_action = flat_action
        if flat_action:
            # Action as an intger that specifies a certain position
            self.action_space = spaces.Discrete(board_size[0] * board_size[1])
        else:
            # Action as a coordinate on the game board
            self.action_space = spaces.MultiDiscrete(list(board_size))

        self.reward_map = {
            OpenResult.FAIL: -10,
            OpenResult.WIN: 10,
            OpenResult.NEIGHBOR: 1,
            OpenResult.ISOLATED: -1,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        """Reset game environment. Guaranteed to be called before `step()`."""

        super().reset(seed=seed)
        self.gameboard.reset_board(seed=seed)

        observation = self.gameboard.get_visible_board()
        info = self._get_info()
        return observation, info

    def step(self, action: tuple[int, int]):
        result = self.gameboard.open(*action)

        observation = self.gameboard.get_visible_board()
        reward = self.reward_map[result]
        terminated = reward in (OpenResult.FAIL, OpenResult.WIN)
        info = self._get_info()
        return observation, reward, terminated, False, info

    def sample_action(self) -> tuple[int, int]:
        action = self.action_space.sample()
        if self.flat_action:
            return self.convert_action(action)
        else:
            return tuple(action)

    def render(self) -> str:
        return self.gameboard.render()

    def convert_action(self, action: int) -> tuple[int, int]:
        return (action // self.gameboard.n_rows, action % self.gameboard.n_cols)

    def _get_info(self) -> dict[str, Any]:
        return {
            "n_closed": self.gameboard.n_closed,
        }
