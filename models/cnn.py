from torch import nn, Tensor
import torch.nn.functional as F


class MineSweeperCNN(nn.Module):

    def __init__(
        self,
        board_size: tuple[int, int],
        n_channels: int,
        depth: int,
    ):
        super().__init__()

        self.input = nn.Conv2d(1, n_channels, (3, 3), padding="same")
        self.cnns = nn.ModuleList(
            [
                nn.Conv2d(n_channels, n_channels, (3, 3), padding="same")
                for _ in range(depth)
            ]
        )

        n_cells = board_size[0] * board_size[1]
        self.ff = nn.Linear(in_features=n_channels * n_cells, out_features=n_cells)

    def forward(self, state: Tensor) -> Tensor:
        # state: (N, H, W)
        state = state.unsqueeze(1)  # (N, 1, H, W)
        v = F.relu(self.input(state))  # (N, C, H, W)
        for layer in self.cnns:
            v = F.relu(layer(v))

        v = v.reshape(v.shape[0], -1)  # (N, C*H*W)
        v = F.relu(self.ff(v))

        return v
