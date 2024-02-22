import math
import random
from itertools import count

import torch
from torch import nn, optim, Tensor

from .memory import ReplayMemory, Transition
from game import MineSweeperEnv


class MineSweeperTrainer:

    def __init__(
        self,
        batch_size: int = 128,
        gamma: float = 0.99,
        eps_range: tuple[float, float] = (0.9, 0.05),
        eps_decay: int = 1000,
        tau: float = 0.005,
        lr: float = 1e-4,
        memory_size: int = 1000,
    ):
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start, self.eps_end = eps_range
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.memory = ReplayMemory(capacity=memory_size)

        self.policy_net: nn.Module = None
        self.target_net: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.env: MineSweeperEnv = None

        self.steps_done = 0

    def register_models(
        self,
        policy_net: nn.Module,
        target_net: nn.Module,
        env: MineSweeperEnv,
    ):
        self.policy_net = policy_net
        self.target_net = target_net
        self.env = env

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True
        )

    def train(self, num_episodes: int):
        episode_durations = []

        for i in range(num_episodes):
            # Reset environment
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1, ...]

            for t in count():
                next_state, terminated = self.step(state)
                if terminated:
                    episode_durations.append(t + 1)
                    break

                state = next_state

    def step(self, state: Tensor) -> tuple[Tensor | None, bool]:
        """
        Conducts one training step.

        RETURN
        - `next_state`
        - `terminated`: whether the current episode is terminated
        """

        action = self._select_action(state)
        observation, reward, terminated, _, _ = self.env.step(action)
        reward = torch.tensor([reward])

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        # Store transition in memory
        self.memory.push(state, action, next_state, reward)

        # Perform one step of optimization on the policy network
        self.optimize()

        # Soft update the target network's weights
        #   Alternatively, we can update the target network's weights every C steps
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

        self.steps_done += 1

        return next_state, terminated

    def optimize(self):

        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(batch_size=self.batch_size)
        batch = Transition(*zip(*transitions))
        # Each element of Transtion is now a list of BATCH_SIZE items

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)  # (batch_size, 2)
        reward_batch = torch.cat(batch.reward)  # (batch_size,)

        # Compute Q(s, a)
        #   The policy network returns Q(s),
        #   and then we choose the values corresponding to the given actions
        state_action_values = torch.gather(
            self.policy_net(state_batch), dim=1, index=action_batch
        )

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            dtype=torch.bool,
        )
        non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])

        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = torch.max(
                self.target_net(non_final_next_state), dim=1
            ).values
        expected_state_action_values = (next_state_values) * self.gamma + reward_batch
        # expected (target): R + gamma * max_a Q(s_{t+1}, a)

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(0))

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def _select_action(self, state: Tensor) -> tuple[int, int]:
        """
        Uses policy network to select an action given the current state.
        Chooses a random action (exploration) with certain probability.

        PARAM
        - `state`: tensor of shape (1, ...)

        RETURN
        - `action`: coordinate tuple (x, y)
        """

        p = random.random()
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )

        if p < eps:
            # exploration
            return self.env.sample_action()

        # exploitation
        with torch.no_grad():
            output = self.policy_net(state)
        if len(output.shape) == 2:
            # flat action space
            action = torch.max(output, dim=1).indices.item()
            return self.env.convert_action(action)
        elif len(output.shape) == 3:
            # grid action space
            x = torch.max(output, dim=1).indices.item()
            y = torch.max(output, dim=2).indices.item()
            return (x, y)
        else:
            raise RuntimeError(f"Invalid model output of shape {output.shape}")
