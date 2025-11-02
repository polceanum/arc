# analyze_embeddings.py
import os
import argparse
import glob
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.exceptions import ConvergenceWarning

# Silence benign SVD/PCA warnings in rank-1 cases
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.decomposition._truncated_svd")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_trajectories(traj_root: str, model: str, max_pairs: int = None):
    """
    Load (T, D) tensors from embeddings_traj/<model>/<task_id>/<pair_id>.pt
    Returns:
      trajs: list of np.ndarray (T, D)
      meta:  list of dicts: {task_id, pair_id, model, T, D, path}
    """
    pattern = os.path.join(traj_root, model, "*", "*.pt")
    files = sorted(glob.glob(pattern))
    if max_pairs is not None:
        files = files[:max_pairs]

    trajs, meta = [], []
    for path in files:
        arr = torch.load(path, map_location="cpu")
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        if arr.ndim != 2:
            continue  # expect (T, D)
        T, D = arr.shape
        task_id = os.path.basename(os.path.dirname(path))
        pair_id = os.path.splitext(os.path.basename(path))[0]
        trajs.append(arr)
        meta.append({"task_id": task_id, "pair_id": pair_id, "model": model,
                     "T": T, "D": D, "path": path})
    return trajs, meta


def reduce_points(
    X: np.ndarray,
    method: str = "pca",
    out_dim: int = 2,
    random_state: int = 0,
    tsne_perp: float = 30.0,
):
    """
    Rank-aware reducer that never requests more components than data can support.
    Pads with zeros when necessary to return out_dim columns.
    For t-SNE, clamps perplexity to a valid range for the number of samples.
    """
    n_samples, n_features = X.shape

    if method == "pca":
        # PCA needs n_components <= min(n_samples-1, n_features)
        n_comp_max = 0
        if n_samples >= 2 and n_features >= 1:
            n_comp_max = min(out_dim, n_samples - 1, n_features)

        if n_comp_max <= 0:
            return np.zeros((n_samples, out_dim), dtype=X.dtype), None

        reducer = PCA(n_components=n_comp_max, random_state=random_state)
        Yp = reducer.fit_transform(X)  # (n_samples, n_comp_max)

        if n_comp_max < out_dim:
            Y = np.zeros((n_samples, out_dim), dtype=Yp.dtype)
            Y[:, :n_comp_max] = Yp
            return Y, reducer
        return Yp, reducer

    elif method == "tsne":
        # Need at least 2 samples to do anything
        if n_samples < 2:
            return np.zeros((n_samples, out_dim), dtype=X.dtype), None

        # Clamp perplexity into a safe range: (2, n_samples-1), with a heuristic for tiny n
        max_valid = max(2.0, float(n_samples) - 1.0)
        perp = min(tsne_perp, max_valid)
        if n_samples < 30:
            perp = min(perp, max(2.0, n_samples / 3.0))

        reducer = TSNE(
            n_components=out_dim,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
            perplexity=perp,
        )
        Y = reducer.fit_transform(X)
        return Y, reducer

    else:
        raise ValueError("method must be 'pca' or 'tsne'")


def get_final_points(trajs, meta):
    """Return (points, meta) keeping only the last embedding from each trajectory."""
    finals = [tr[-1:] for tr in trajs if tr.shape[0] > 0]  # each is (1, D)
    return finals, meta


def plot_trajectories(
    trajs,
    meta,
    Y,
    title: str,
    color_by: str = "task_id",
    save_path: str | None = None,
    max_legend: int = 30,
    final_only: bool = False,
    legend_outside: bool = False,
):
    """
    Plot trajectories (lines + endpoints) or just final points.
    Clean legend: exactly one handle per group, colors consistent.
    """
    # Re-slice Y back into per-trajectory segments
    idx = 0
    groups = {}
    for tr, m in zip(trajs, meta):
        T = tr.shape[0]
        Y_tr = Y[idx: idx + T]
        idx += T
        key = m[color_by]
        groups.setdefault(key, []).append(Y_tr)

    # Colormap (new Matplotlib API)
    cmap = mpl.colormaps.get("tab20")
    colors_list = list(cmap.colors)
    keys = sorted(groups.keys())
    color_of = {k: colors_list[i % len(colors_list)] for i, k in enumerate(keys)}

    plt.figure(figsize=(8, 7))
    ax = plt.gca()

    for k in keys:
        color = color_of[k]
        first = True
        for Y_tr in groups[k]:
            if final_only:
                pt = Y_tr[-1, :]
                ax.scatter(pt[0], pt[1], s=24, color=color, zorder=3, label=k if first else "_nolegend_")
                first = False
            else:
                if Y_tr.shape[0] < 2:
                    ax.scatter(Y_tr[-1, 0], Y_tr[-1, 1], s=18, color=color, label=k if first else "_nolegend_")
                    first = False
                    continue
                ax.plot(Y_tr[:, 0], Y_tr[:, 1], "-", alpha=0.85, color=color,
                        linewidth=1.5, label=k if first else "_nolegend_")
                first = False
                ax.scatter(Y_tr[-1, 0], Y_tr[-1, 1], s=22, color=color, zorder=3, label="_nolegend_")

    ax.set_title(title)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")

    # Build clean legend
    handles, labels = ax.get_legend_handles_labels()
    filt = [(h, l) for h, l in zip(handles, labels) if l != "_nolegend_"]
    seen, handles_u, labels_u = set(), [], []
    for h, l in filt:
        if l in seen:
            continue
        seen.add(l)
        handles_u.append(h)
        labels_u.append(l)

    # Legend policy:
    # - max_legend < 0 → show all
    # - else: trim to max_legend (but still show something)
    if len(labels_u) > 0:
        show_all = (max_legend is None) or (max_legend < 0)
        if not show_all and len(labels_u) > max_legend:
            handles_u, labels_u = handles_u[:max_legend], labels_u[:max_legend]

        many = len(labels_u) > 15
        ncol = 1 if not many else min(4, max(2, len(labels_u) // 15))

        if legend_outside or many:
            ax.legend(handles_u, labels_u,
                      loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      borderaxespad=0.0, fontsize=8, title=color_by, ncol=ncol)
            plt.tight_layout(rect=[0, 0, 0.82, 1])  # leave space on right
        else:
            ax.legend(handles_u, labels_u, loc="best", fontsize=8, title=color_by, ncol=ncol)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] saved: {save_path}")
        plt.close()
    else:
        plt.show()


def concat_points(trajs):
    """Concatenate list of (T, D) into (sum_T, D); requires same D."""
    D = trajs[0].shape[1]
    assert all(t.shape[1] == D for t in trajs), "Different D in concat_points"
    return np.vstack(trajs), D


def group_by_dimension(trajs, meta):
    """Return dict D -> (trajs_list, meta_list)."""
    buckets = {}
    for tr, m in zip(trajs, meta):
        D = tr.shape[1]
        if D not in buckets:
            buckets[D] = ([], [])
        buckets[D][0].append(tr)
        buckets[D][1].append(m)
    return buckets


def svd_align(trajs, svd_dim: int, random_state: int = 0):
    """
    Project each (T, D_i) trajectory independently to (T, svd_dim) using TruncatedSVD.
    Returns concatenated points X (sum_T, svd_dim) and per-trajectory reduced arrays.
    """
    reduced = []
    for tr in trajs:
        # Ensure >=1 component and <= "rank-ish" bounds
        k = min(svd_dim, tr.shape[1], max(1, min(tr.shape[0], tr.shape[1])))
        svd = TruncatedSVD(n_components=k, random_state=random_state)
        Z = svd.fit_transform(tr)  # (T, k)
        if k < svd_dim:
            Zp = np.zeros((Z.shape[0], svd_dim), dtype=Z.dtype)
            Zp[:, :k] = Z
            Z = Zp
        reduced.append(Z)
    X = np.vstack(reduced)  # (sum_T, svd_dim)
    return X, reduced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default="embeddings_traj")
    ap.add_argument("--model", choices=["nca", "dqn", "dqn_agn"], default="nca")
    ap.add_argument("--method", choices=["pca", "tsne"], default="pca")
    ap.add_argument("--tsne-perp", type=float, default=30.0,
                    help="t-SNE perplexity; auto-clamped to a valid range given n_samples.")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--color-by", choices=["task_id", "model"], default="task_id")
    ap.add_argument("--save-dir", default=None, help="If set, save figures here instead of showing.")
    ap.add_argument("--per-task", nargs="*", default=None, help="Optional list of task_ids to focus on.")
    ap.add_argument("--align", choices=["group", "svd"], default="group",
                    help="'group': one plot per D; 'svd': project all to a common dim with TruncatedSVD.")
    ap.add_argument("--svd-dim", type=int, default=256, help="Common dim for --align svd.")
    ap.add_argument("--only-dim", type=int, default=None,
                    help="When using --align group, restrict to a specific embedding dimension D.")
    ap.add_argument("--final-only", action="store_true",
                    help="Plot only final embeddings (no full trajectories).")
    ap.add_argument("--legend-max", type=int, default=30,
                    help="Maximum number of legend entries; use -1 for unlimited.")
    ap.add_argument("--legend-outside", action="store_true",
                    help="Place the legend outside on the right.")
    args = ap.parse_args()

    trajs, meta = load_trajectories(args.traj_dir, args.model, args.max_pairs)
    if not trajs:
        print("No trajectories found. Did you run extract_embeddings.py with --save-traj ?")
        return

    # Optional task filter
    if args.per_task:
        keep = [m["task_id"] in args.per_task for m in meta]
        trajs = [t for t, k in zip(trajs, keep) if k]
        meta  = [m for m, k in zip(meta, keep) if k]
        if not trajs:
            print("No trajectories left after filtering by --per-task.")
            return

    # Optional dimension filter (group mode)
    if args.only_dim is not None:
        keep = [m["D"] == args.only_dim for m in meta]
        trajs = [t for t, k in zip(trajs, keep) if k]
        meta  = [m for m, k in zip(meta, keep) if k]
        if not trajs:
            print(f"No trajectories with D={args.only_dim}.")
            return

    # Summary of dimensions
    dims = {}
    for m in meta:
        dims[m["D"]] = dims.get(m["D"], 0) + 1
    print(f"Found {len(trajs)} trajectories across {len(dims)} embedding dimensions:")
    for d, cnt in sorted(dims.items()):
        print(f"  D={d}: {cnt} trajs")

    # If final-only and single D with SVD align requested, take a fast direct path
    unique_D = sorted({m["D"] for m in meta})
    if args.final_only and args.align == "svd" and len(unique_D) == 1:
        X, _ = concat_points(trajs)  # (N, D)
        print(f"[FAST] Final-only with single D={unique_D[0]} → bypass SVD; points={X.shape[0]}")
        Y, _ = reduce_points(X, method=args.method, out_dim=2, random_state=0, tsne_perp=args.tsne_perp)
        title = f"{args.model.upper()} finals via {args.method.upper()} | color={args.color_by}"
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"{args.model}_{args.method}_finals_direct.png")
        plot_trajectories(trajs, meta, Y, title, color_by=args.color_by,
                          save_path=save_path, max_legend=args.legend_max,
                          final_only=True, legend_outside=args.legend_outside)
        return

    # If final-only, reduce each trajectory to endpoint
    if args.final_only:
        print("⚙️  Using only final embeddings per trajectory.")
        trajs, meta = get_final_points(trajs, meta)

    if args.align == "group":
        buckets = group_by_dimension(trajs, meta)
        for D, (tr_list, meta_list) in sorted(buckets.items(), key=lambda kv: kv[0]):
            X, _ = concat_points(tr_list)  # (sum_T, D)
            print(f"[GROUP] D={D}  total_points={X.shape[0]}")
            Y, _ = reduce_points(X, method=args.method, out_dim=2, random_state=0, tsne_perp=args.tsne_perp)
            title = f"{args.model.upper()} {'finals' if args.final_only else 'trajectories'} via {args.method.upper()} | D={D} | color={args.color_by}"
            save_path = None
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                tag = "finals" if args.final_only else "traj"
                save_path = os.path.join(args.save_dir, f"{args.model}_{args.method}_{tag}_D{D}.png")
            plot_trajectories(tr_list, meta_list, Y, title, color_by=args.color_by,
                              save_path=save_path, max_legend=args.legend_max,
                              final_only=args.final_only, legend_outside=args.legend_outside)
    else:
        # Align all to a common dim via SVD, then single plot
        X, _ = svd_align(trajs, svd_dim=args.svd_dim, random_state=0)  # (sum_T, K)
        print(f"[SVD] Aligned to common dim={args.svd_dim}, total_points={X.shape[0]}")
        Y, _ = reduce_points(X, method=args.method, out_dim=2, random_state=0, tsne_perp=args.tsne_perp)
        title = f"{args.model.upper()} {'finals' if args.final_only else 'trajectories'} via {args.method.upper()} | align=SVD(K={args.svd_dim}) | color={args.color_by}"
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            tag = "finals" if args.final_only else "traj"
            save_path = os.path.join(args.save_dir, f"{args.model}_{args.method}_{tag}_svdK{args.svd_dim}.png")
        plot_trajectories(trajs, meta, Y, title, color_by=args.color_by,
                          save_path=save_path, max_legend=args.legend_max,
                          final_only=args.final_only, legend_outside=args.legend_outside)


if __name__ == "__main__":
    main()
