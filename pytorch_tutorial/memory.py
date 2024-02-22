from collections import deque, namedtuple
import random


Transition = namedtuple(
    "Transition",
    ("state", "action", "next_state", "reward"),
)


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size: int):
        return random.sample(self.memory, k=batch_size)
    
    def __len__(self):
        return len(self.memory)
