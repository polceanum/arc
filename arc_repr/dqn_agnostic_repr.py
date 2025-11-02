from typing import Optional
import random
import torch
from torch import nn, optim
from .base import RepresentationExtractor, Tensor, FitCallback


def _to_unit(x: Tensor) -> Tensor:
    # 0..9 -> 0..1
    return x.float().div(9.0)


class GridPointerEnv:
    """
    Same simple env as before:
      - Iterate cells in raster order.
      - Action = choose color in {0..9} for the current cell.
      - Reward = 1 if matches target at that cell else 0.
      - Episode length = H*W.
    """
    def __init__(self, input_grid: Tensor, target_grid: Tensor):
        self.inp = _to_unit(input_grid)           # (H,W) float [0,1]
        self.tgt = target_grid.to(torch.int64)    # (H,W)
        self.H, self.W = self.inp.shape
        self.N = self.H * self.W
        self.reset()

    def reset(self):
        self.grid = (self.inp * 9.0).clone()  # work in 0..9
        self.ptr = 0
        return self._obs()

    def _obs(self) -> Tensor:
        g = (self.grid / 9.0).unsqueeze(0)        # (1,H,W)
        pmap = torch.zeros(1, self.H, self.W)     # (1,H,W) one-hot current pointer
        r = self.ptr // self.W
        c = self.ptr % self.W
        pmap[0, r, c] = 1.0
        return torch.cat([g, pmap], dim=0)        # (2,H,W)

    def step(self, action_color: int):
        r = self.ptr // self.W
        c = self.ptr % self.W
        self.grid[r, c] = float(action_color)
        reward = 1.0 if action_color == int(self.tgt[r, c].item()) else 0.0
        self.ptr += 1
        done = self.ptr >= self.N
        return (self._obs() if not done else None), reward, done


class SmallDQNAgnostic(nn.Module):
    """
    Size-agnostic CNN:
      - Convs over (2,H,W)
      - Global Average Pooling to (B, C, 1, 1)
      - MLP head to 10 Q-values
    Works for any HxW >= 1x1.
    """
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # -> (B, 32, 1, 1)
        self.head = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, obs: Tensor) -> Tensor:
        # obs: (B,2,H,W)
        x = self.body(obs)                   # (B,32,H,W)
        x = self.gap(x).squeeze(-1).squeeze(-1)  # (B,32)
        return self.head(x)                  # (B,10)


class DQNAgnosticRepresentation(RepresentationExtractor):
    """
    DQN variant whose parameter count is independent of input grid size.
    Uses global average pooling so embeddings are comparable across HxW.
    """
    def __init__(
        self,
        episodes: int = 8,
        gamma: float = 0.0,
        lr: float = 1e-3,
        eps_start: float = 0.5,
        eps_end: float = 0.05,
        device: str = "cpu",
        verbose: bool = False,
        print_every: int = 100,
        traj_every: int = 1,
    ):
        self.episodes = episodes
        self.gamma = gamma
        self.lr = lr
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.device = device
        self.verbose = verbose
        self.print_every = print_every
        self.traj_every = traj_every

        self.model: Optional[SmallDQNAgnostic] = None
        self._shape = None  # (H, W)
        self._last_acc: float = 0.0

    def _eps_schedule(self, ep: int) -> float:
        if self.episodes <= 1:
            return self.eps_end
        a = ep / (self.episodes - 1)
        return self.eps_start * (1 - a) + self.eps_end * a

    def fit(self, input_grid: Tensor, output_grid: Tensor, callback: Optional[FitCallback] = None) -> "DQNAgnosticRepresentation":
        if input_grid.shape != output_grid.shape:
            raise ValueError("DQNAgnosticRepresentation expects equal HxW (same-shape filtering).")

        self._shape = tuple(output_grid.shape)
        env = GridPointerEnv(input_grid, output_grid)

        self.model = SmallDQNAgnostic().to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        steps_total = env.N
        self.model.train()
        last_acc = 0.0

        for ep in range(1, self.episodes + 1):
            obs = env.reset().unsqueeze(0).to(self.device)  # (1,2,H,W)
            eps = self._eps_schedule(ep - 1)
            correct = 0

            for _ in range(steps_total):
                # ε-greedy over 10 colors
                if random.random() < eps:
                    action = random.randint(0, 9)
                else:
                    with torch.no_grad():
                        action = int(self.model(obs).argmax(dim=1).item())

                next_obs, reward, done = env.step(action)
                correct += int(reward)

                q_pred = self.model(obs)              # (1,10)
                target = q_pred.detach().clone()
                # gamma=0 simple target; keep identical to original baseline
                target[0, action] = reward

                loss = loss_fn(q_pred, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                if done:
                    break
                obs = next_obs.unsqueeze(0).to(self.device)

            last_acc = correct / steps_total
            if self.verbose and (ep % self.print_every == 0 or ep == 1 or ep == self.episodes):
                H, W = self._shape
                print(f"[DQN-AGN] ep {ep:3d}/{self.episodes}  steps={steps_total}  acc={last_acc:.3f}  eps={eps:.3f}  shape={H}x{W}")

            if callback is not None and (ep % self.traj_every == 0 or ep == 1 or ep == self.episodes):
                callback(ep, self._flatten_params().cpu())

        self._last_acc = last_acc
        return self

    def embed(self) -> Tensor:
        return self._flatten_params().cpu()

    def predict(self, input_grid: Tensor) -> Tensor:
        assert self.model is not None, "Call fit() first."
        assert self._shape is not None
        H, W = self._shape
        env = GridPointerEnv(input_grid, torch.zeros(H, W, dtype=torch.int64))
        obs = env.reset().unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            for _ in range(H * W):
                action = int(self.model(obs).argmax(dim=1).item())
                next_obs, _, done = env.step(action)
                if done:
                    break
                obs = next_obs.unsqueeze(0)
        return env.grid.round().clamp(0, 9).to(torch.int64)
