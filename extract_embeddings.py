import os, argparse, json, hashlib
import torch
from arc_data import iter_arc_train_pairs_same_shape, to_tensor_grid
from arc_repr.nca_repr import NCARepresentation
from arc_repr.dqn_repr import DQNRepresentation
from arc_repr.dqn_movepaint_repr import DQNMovePaintRepresentation
from arc_repr.dqn_agnostic_repr import DQNAgnosticRepresentation

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x  # fallback if tqdm isn't available


def _lazy_import_matplotlib():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    return matplotlib, plt, np


def visualize_pair(task_id, pair_id, model_name, x, y, pred, mode, out_dir="viz"):
    if mode == "off":
        return
    matplotlib, plt, np = _lazy_import_matplotlib()
    x_np, y_np, p_np = x.cpu().numpy(), y.cpu().numpy(), pred.cpu().numpy()
    assert x_np.shape == y_np.shape == p_np.shape
    H, W = y_np.shape
    diff = (p_np != y_np).astype(float)

    fig, axs = plt.subplots(1, 4, figsize=(10, 2.6))
    for ax in axs: ax.set_xticks([]); ax.set_yticks([])
    axs[0].imshow(x_np, cmap="tab10", vmin=0, vmax=9); axs[0].set_title("Input")
    axs[1].imshow(y_np, cmap="tab10", vmin=0, vmax=9); axs[1].set_title("Target")
    axs[2].imshow(p_np, cmap="tab10", vmin=0, vmax=9); axs[2].set_title(f"Pred ({model_name})")
    axs[3].imshow(diff, cmap="gray", vmin=0, vmax=1); axs[3].set_title("Diff")
    fig.suptitle(f"{task_id} | pair {pair_id} | {H}x{W}")
    fig.tight_layout()
    if mode == "show":
        plt.show()
    else:
        os.makedirs(f"{out_dir}/{task_id}", exist_ok=True)
        path = f"{out_dir}/{task_id}/{pair_id}_{model_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[VIZ] saved: {path}")
        plt.close(fig)


def save_embedding(path: str, emb: torch.Tensor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(emb, path)


class TrajectoryRecorder:
    def __init__(self, out_root: str, model_name: str, task_id: str, pair_id: str):
        self.out_root, self.model_name, self.task_id, self.pair_id = out_root, model_name, task_id, pair_id
        self._buf = []

    def __call__(self, step_idx: int, emb: torch.Tensor):
        self._buf.append(emb.clone())

    def flush(self):
        if not self._buf: return
        arr = torch.stack(self._buf, dim=0)  # (T, D)
        out_dir = os.path.join(self.out_root, self.model_name, self.task_id)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.pair_id}.pt")
        torch.save(arr, path)
        print(f"[TRAJ] saved: {path}  (T={arr.shape[0]}, D={arr.shape[1]})")


# ---------- stable ID helpers ----------
def stable_pair_id(pair) -> str:
    """
    Deterministic short ID from the input grid contents.
    Uses JSON with compact separators and Blake2b digest (8 hex chars).
    """
    s = json.dumps(pair["input"], separators=(",", ":"), ensure_ascii=False)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=4).hexdigest()

def sort_pairs_for_determinism(pairs):
    """Sort by task_id then by stable_pair_id so iteration is reproducible."""
    keyed = []
    for (task_id, pair) in pairs:
        pid = stable_pair_id(pair)
        keyed.append((task_id, pid, pair))
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [(task_id, pair, pid) for (task_id, pid, pair) in keyed]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="arc-prize-2024")
    parser.add_argument("--visualize", choices=["off", "show", "save"], default="off")
    parser.add_argument("--viz-limit", type=int, default=None)
    parser.add_argument("--save-traj", action="store_true")
    parser.add_argument("--traj-dir", default="embeddings_traj")
    parser.add_argument("--traj-every", type=int, default=10)
    parser.add_argument("--nca-iters", type=int, default=500)
    parser.add_argument("--nca-lr", type=float, default=1e-3)
    parser.add_argument("--nca-step-min", type=int, default=16)
    parser.add_argument("--nca-step-max", type=int, default=32)
    parser.add_argument("--dqn-episodes", type=int, default=500)
    parser.add_argument("--dqn-lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true", help="Ignore existing files and retrain.")
    args = parser.parse_args()

    train_path = f"{args.data_dir}/arc-agi_training_challenges.json"

    models = {
        # "nca": lambda: NCARepresentation(
        #     step_min=args.nca_step_min,
        #     step_max=args.nca_step_max,
        #     iters=args.nca_iters,
        #     lr=args.nca_lr,
        #     device=args.device,
        #     verbose=True,
        #     print_every=100,
        #     traj_every=max(1, args.traj_every),
        # ),
        # "dqn": lambda: DQNRepresentation(
        #     episodes=args.dqn_episodes,
        #     lr=args.dqn_lr,
        #     device=args.device,
        #     verbose=True,
        #     print_every=100,
        #     traj_every=max(1, args.traj_every),
        # ),
        # "dqn_mp": lambda: DQNMovePaintRepresentation(
        #     episodes=1000, gamma=0.95, lr=1e-3,
        #     eps_start=0.6, eps_end=0.05, step_mult=4.0,
        #     device=args.device, verbose=True, print_every=100, traj_every=max(1, args.traj_every),
        # ),
        "dqn_agn": lambda: DQNAgnosticRepresentation(
            lr=args.dqn_lr,
            device=args.device,
            verbose=True,
            print_every=10,
            traj_every=max(1, args.traj_every),
        ),
    }

    out_dir = "embeddings"

    # Deterministic list of pairs + stable ids
    raw_pairs = list(iter_arc_train_pairs_same_shape(train_path))
    pairs = sort_pairs_for_determinism(raw_pairs)
    print(f"Found {len(pairs)} same-shape training pairs.")

    viz_budget = args.viz_limit if args.viz_limit is not None else len(pairs)
    seen_for_viz = 0

    for (task_id, pair, pid) in tqdm(pairs, desc="Pairs"):
        x, y = to_tensor_grid(pair)
        print(f"\n=== Task {task_id} | pair_id {pid} | shape {tuple(x.shape)} ===")

        for name, ctor in models.items():
            # paths for this (task, pair, model)
            save_path = f"{out_dir}/{task_id}/{pid}_{name}.pt"
            traj_path = os.path.join(args.traj_dir, name, task_id, f"{pid}.pt")

            emb_exists = os.path.isfile(save_path)
            traj_exists = os.path.isfile(traj_path)
            should_skip = (not args.force) and (
                (args.save_traj and emb_exists and traj_exists) or
                (not args.save_traj and emb_exists)
            )

            if should_skip:
                print(f"[{name.upper()}] skip: already have "
                      f"{'embedding+trajectory' if args.save_traj else 'embedding'} "
                      f"({save_path}{' & ' + traj_path if args.save_traj else ''})")
                continue

            print(f"[{name.upper()}] training...")
            model = ctor()

            # optional trajectory recording
            recorder = None
            cb = None
            if args.save_traj:
                recorder = TrajectoryRecorder(args.traj_dir, name, task_id, pid)
                cb = recorder.__call__

            model.fit(x, y, callback=cb)
            emb = model.embed()

            # save final embedding
            save_embedding(save_path, emb)
            print(f"[{name.upper()}] saved: {save_path}  (emb_dim={emb.numel()})")

            # flush trajectory
            if recorder is not None:
                recorder.flush()

            # visualize only newly trained items
            if args.visualize != "off" and seen_for_viz < viz_budget:
                try:
                    pred = model.predict(x)
                    visualize_pair(task_id, pid, name, x, y, pred, mode=args.visualize, out_dir="viz")
                except Exception as e:
                    print(f"[VIZ] skipped ({name}) due to error: {e}")

        if args.visualize != "off" and seen_for_viz < viz_budget:
            seen_for_viz += 1

    print("\n✅ Embeddings extraction finished.")


if __name__ == "__main__":
    main()
