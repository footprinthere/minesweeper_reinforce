from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
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
            dtype=int,
        )

        self.flat_action = flat_action
        if flat_action:
            # Action as an intger that specifies a certain position: shape ()
            self.action_space = spaces.Discrete(board_size[0] * board_size[1])
        else:
            # Action as a coordinate on the game board: shape (2,)
            self.action_space = spaces.MultiDiscrete(list(board_size))

        self.reward_map = {
            OpenResult.FAIL: -10,
            OpenResult.WIN: 10,
            OpenResult.NEIGHBOR: 1,
            OpenResult.ISOLATED: -3,
            OpenResult.DUPLICATED: -3,
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
        return observation, {}

    def step(self, action: tuple[int, int]):
        result = self.gameboard.open(*action)

        observation = self.gameboard.get_visible_board()
        reward = self.reward_map[result]
        terminated = result in (OpenResult.FAIL, OpenResult.WIN)
        info = {
            "n_closed": self.gameboard.n_closed,
            "result": result,
        }
        return observation, reward, terminated, False, info

    def sample_action(self) -> Tensor:
        action = self.action_space.sample()
        if self.flat_action:
            return torch.tensor([[action]], dtype=torch.long)
        else:
            return torch.tensor(np.array([action]), dtype=torch.long)

    def tensor_to_pos(self, action: Tensor) -> tuple[int, int]:
        if action.shape == (1, 1):
            action = action.item()
            return action // self.gameboard.n_rows, action % self.gameboard.n_cols
        elif action.shape == (1, 2):
            return tuple(action[0].numpy())

    def render(self) -> str:
        return self.gameboard.render()
