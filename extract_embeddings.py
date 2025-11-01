import os
import argparse
import torch
from arc_data import iter_arc_train_pairs_same_shape, to_tensor_grid
from arc_repr.nca_repr import NCARepresentation
from arc_repr.dqn_repr import DQNRepresentation

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x  # fallback if tqdm isn't available


# -----------------------------
# Visualization helpers
# -----------------------------
def _lazy_import_matplotlib():
    import matplotlib
    # Do not force non-interactive backend; user may want windows. For CI, set MPLBACKEND externally.
    import matplotlib.pyplot as plt
    import numpy as np
    return matplotlib, plt, np


def visualize_pair(task_id: str,
                   pair_id: int,
                   model_name: str,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   pred: torch.Tensor,
                   mode: str,                 # "off" | "show" | "save"
                   out_dir: str = "viz"):
    if mode == "off":
        return

    matplotlib, plt, np = _lazy_import_matplotlib()

    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    p_np = pred.cpu().numpy()

    assert x_np.shape == y_np.shape == p_np.shape, "visualize expects matching HxW"
    H, W = y_np.shape

    # diff mask: mismatched cells -> 1, else 0
    diff = (p_np != y_np).astype(float)

    fig, axs = plt.subplots(1, 4, figsize=(10, 2.6))
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])

    axs[0].imshow(x_np, cmap="tab10", vmin=0, vmax=9); axs[0].set_title("Input")
    axs[1].imshow(y_np, cmap="tab10", vmin=0, vmax=9); axs[1].set_title("Target")
    axs[2].imshow(p_np, cmap="tab10", vmin=0, vmax=9); axs[2].set_title(f"Pred ({model_name})")
    axs[3].imshow(diff, cmap="gray", vmin=0, vmax=1); axs[3].set_title("Diff")

    fig.suptitle(f"{task_id} | pair {pair_id} | {H}x{W}")
    fig.tight_layout()

    if mode == "show":
        plt.show()
    elif mode == "save":
        os.makedirs(f"{out_dir}/{task_id}", exist_ok=True)
        path = f"{out_dir}/{task_id}/{pair_id}_{model_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[VIZ] saved: {path}")
        plt.close(fig)


def save_embedding(path: str, emb: torch.Tensor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(emb, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="arc-prize-2024")
    parser.add_argument("--visualize", choices=["off", "show", "save"], default="off",
                        help="Visualize each pair’s results per model.")
    parser.add_argument("--viz-limit", type=int, default=None,
                        help="Visualize at most N pairs (still trains on all).")
    parser.add_argument("--nca-iters", type=int, default=500)
    parser.add_argument("--nca-lr", type=float, default=1e-3)
    parser.add_argument("--nca-step-min", type=int, default=16)
    parser.add_argument("--nca-step-max", type=int, default=32)
    parser.add_argument("--dqn-episodes", type=int, default=500)
    parser.add_argument("--dqn-lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_path = f"{args.data_dir}/arc-agi_training_challenges.json"

    models = {
        "nca": lambda: NCARepresentation(
            step_min=args.nca_step_min,
            step_max=args.nca_step_max,
            iters=args.nca_iters,
            lr=args.nca_lr,
            device=args.device,
            verbose=True,
            print_every=100,
        ),
        "dqn": lambda: DQNRepresentation(
            episodes=args.dqn_episodes,
            lr=args.dqn_lr,
            device=args.device,
            verbose=True,
            print_every=100,
        ),
    }

    out_dir = "embeddings"

    pairs = list(iter_arc_train_pairs_same_shape(train_path))
    print(f"Found {len(pairs)} same-shape training pairs.")

    # For a deterministic small sample visualization, cap the iterator view if requested
    viz_budget = args.viz_limit if args.viz_limit is not None else len(pairs)
    seen_for_viz = 0

    for (task_id, pair) in tqdm(pairs, desc="Pairs"):
        x, y = to_tensor_grid(pair)
        idx = hash(str(pair["input"])) & 0xFFFF
        print(f"\n=== Task {task_id} | pair_id {idx} | shape {tuple(x.shape)} ===")

        for name, ctor in models.items():
            print(f"[{name.upper()}] training...")
            model = ctor().fit(x, y)
            emb = model.embed()
            save_path = f"{out_dir}/{task_id}/{idx}_{name}.pt"
            save_embedding(save_path, emb)
            print(f"[{name.upper()}] saved: {save_path}  (emb_dim={emb.numel()})")

            # Optional visualization
            if args.visualize != "off" and seen_for_viz < viz_budget:
                try:
                    pred = model.predict(x)
                    visualize_pair(task_id, idx, name, x, y, pred, mode=args.visualize, out_dir="viz")
                except Exception as e:
                    print(f"[VIZ] skipped ({name}) due to error: {e}")

        if args.visualize != "off" and seen_for_viz < viz_budget:
            seen_for_viz += 1

    print("\n✅ Embeddings extraction finished.")


if __name__ == "__main__":
    main()
