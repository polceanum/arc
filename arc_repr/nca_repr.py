from typing import Optional, Tuple
import torch
from torch import nn, optim
from .base import RepresentationExtractor, Tensor

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def to_class_grid(x: Tensor) -> Tensor:
    """H×W int grid (already 0..9)."""
    return x.to(torch.long)

def one_hot_10(x: Tensor) -> Tensor:
    """(H,W) int -> (10,H,W) float one-hot."""
    oh = torch.zeros(10, x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    oh.scatter_(0, x.unsqueeze(0), 1.0)
    return oh

# ------------------------------------------------------------
# NCA
# ------------------------------------------------------------
class SoftmaxNCA(nn.Module):
    """
    NCA with 10 visible *logit* channels + L latent channels.
    - Perception: 3x3 conv
    - Update: 1x1 conv
    - Stochastic 'fire' mask per step (p = fire_rate)
    """
    def __init__(self, latent_ch: int = 8, hidden: int = 64, fire_rate: float = 0.5):
        super().__init__()
        self.latent_ch = latent_ch
        self.total_ch = 10 + latent_ch
        self.fire_rate = fire_rate

        self.percep = nn.Conv2d(self.total_ch, hidden, 3, padding=1, bias=False)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden, self.total_ch, 1, bias=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.1)

    def step(self, state: Tensor) -> Tensor:
        # state: (B, 10+L, H, W)
        y  = self.percep(state)
        dy = self.update(y)

        if self.fire_rate is not None and self.fire_rate < 1.0:
            # stochastic update mask per cell (broadcast over channels)
            if self.training:
                mask = (torch.rand_like(state[:, :1, :, :]) < self.fire_rate).float()
            else:
                # deterministic expectation during eval
                mask = torch.full_like(state[:, :1, :, :], self.fire_rate)
            dy = dy * mask

        # residual update (no in-place ops)
        state = state + dy
        return state

    def forward(self, state: Tensor, steps: int) -> Tensor:
        for _ in range(steps):
            state = self.step(state)
        return state


class NCARepresentation(RepresentationExtractor):
    """
    Trains a SoftmaxNCA to map input->output for a SINGLE pair.
    Embedding = flattened state_dict() weights.
    """
    def __init__(
        self,
        step_min: int = 16,
        step_max: int = 32,
        iters: int = 400,
        lr: float = 1e-3,
        latent_ch: int = 8,
        hidden: int = 64,
        fire_rate: float = 0.5,
        grad_clip: float = 1.0,
        device: str = "cpu",
        verbose: bool = False,
        print_every: int = 100,
    ):
        self.step_min = step_min
        self.step_max = step_max
        self.iters = iters
        self.lr = lr
        self.latent_ch = latent_ch
        self.hidden = hidden
        self.fire_rate = fire_rate
        self.grad_clip = grad_clip
        self.device = device
        self.verbose = verbose
        self.print_every = print_every

        self.model = SoftmaxNCA(latent_ch=latent_ch, hidden=hidden, fire_rate=fire_rate).to(device)
        self._shape: Optional[Tuple[int, int]] = None
        self._last_loss: Optional[float] = None

    def fit(self, input_grid: Tensor, output_grid: Tensor) -> "NCARepresentation":
        if input_grid.shape != output_grid.shape:
            raise ValueError("NCARepresentation expects equal HxW (you enabled same-shape filtering).")

        H, W = output_grid.shape
        self._shape = (H, W)

        # Build initial state: visible logits initialized from input one-hot (as logits)
        # Trick: start with logits = log(onehot + eps) ~ sharp one-hot around input colors
        x_cls = to_class_grid(input_grid)
        vis0 = one_hot_10(x_cls)  # (10,H,W)
        eps = 1e-3
        vis_logits0 = torch.log(vis0 + eps) - torch.log1p(-vis0 + eps)  # logit(p)
        latent0 = torch.zeros(self.latent_ch, H, W)
        state0 = torch.cat([vis_logits0, latent0], dim=0).unsqueeze(0).to(self.device)  # (1,10+L,H,W)

        # Targets as class indices (CrossEntropy expects class indices)
        y_cls = to_class_grid(output_grid).to(self.device)  # (H,W)

        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()  # over 10 classes per pixel

        self.model.train()
        for it in range(1, self.iters + 1):
            opt.zero_grad(set_to_none=True)

            # random number of steps per iteration (curriculum / robustness)
            steps = torch.randint(self.step_min, self.step_max + 1, (1,)).item()
            pred_state = self.model(state0, steps=steps)   # (1, 10+L, H, W)
            logits = pred_state[:, :10]                    # (1, 10, H, W)
            loss = loss_fn(logits, y_cls.unsqueeze(0))     # CrossEntropy: input (N,C,H,W), target (N,H,W)


            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            opt.step()

            self._last_loss = float(loss.item())
            if self.verbose and (it % self.print_every == 0 or it == 1 or it == self.iters):
                print(f"[NCA] it {it:4d}/{self.iters}  steps={steps:2d}  CE={self._last_loss:.4f}")

        return self

    def embed(self) -> Tensor:
        return self._flatten_params().cpu()

    def predict(self, input_grid: Tensor) -> Tensor:
        assert self._shape is not None, "Call fit() first."
        H, W = self._shape
        x_cls = to_class_grid(input_grid)
        vis0 = one_hot_10(x_cls)
        eps = 1e-3
        vis_logits0 = torch.log(vis0 + eps) - torch.log1p(-vis0 + eps)
        latent0 = torch.zeros(self.latent_ch, H, W)
        state0 = torch.cat([vis_logits0, latent0], dim=0).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            # Use a fixed step count near upper range for crisp outputs
            steps = self.step_max
            pred_state = self.model(state0, steps=self.step_max)  # (1, 10+L, H, W)
            logits = pred_state[:, :10]                           # (1, 10, H, W)
            pred = logits.argmax(dim=1)[0]                        # (H, W)
            return pred.to(torch.int64)

