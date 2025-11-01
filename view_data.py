import json, random, numpy as np
import matplotlib.pyplot as plt

def show_arc_task(json_path, task_id=None):
    """
    Visualize all train/test pairs for a given ARC task.
    Works with both training and evaluation JSONs.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # pick a random task if none provided
    if task_id is None:
        task_id = random.choice(list(data.keys()))
    task = data[task_id]

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    n_train, n_test = len(train_pairs), len(test_pairs)
    total_rows = max(n_train, n_test)
    fig, axs = plt.subplots(total_rows, 4, figsize=(8, 2 * total_rows))

    if total_rows == 1:
        axs = np.array([axs])  # keep 2D indexing

    for i in range(total_rows):
        # TRAIN pair
        if i < n_train:
            inp = np.array(train_pairs[i]["input"])
            out = np.array(train_pairs[i]["output"])
            axs[i, 0].imshow(inp, cmap="tab10", vmin=0, vmax=9)
            axs[i, 0].set_title(f"train input {i}")
            axs[i, 1].imshow(out, cmap="tab10", vmin=0, vmax=9)
            axs[i, 1].set_title(f"train output {i}")
        else:
            axs[i, 0].axis("off")
            axs[i, 1].axis("off")

        # TEST pair (output may be missing in eval/test data)
        if i < n_test:
            inp = np.array(test_pairs[i]["input"])
            axs[i, 2].imshow(inp, cmap="tab10", vmin=0, vmax=9)
            axs[i, 2].set_title(f"test input {i}")
            if "output" in test_pairs[i]:
                out = np.array(test_pairs[i]["output"])
                axs[i, 3].imshow(out, cmap="tab10", vmin=0, vmax=9)
                axs[i, 3].set_title(f"test output {i}")
            else:
                axs[i, 3].imshow(np.zeros_like(inp), cmap="Greys")
                axs[i, 3].set_title(f"test output ?")
        else:
            axs[i, 2].axis("off")
            axs[i, 3].axis("off")

        for j in range(4):
            axs[i, j].axis("off")

    plt.suptitle(f"{json_path.split('/')[-1]} | Task: {task_id}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_dir = "arc-prize-2024"
    # Try both training and evaluation sets
    print("Showing one training task:")
    show_arc_task(f"{data_dir}/arc-agi_training_challenges.json")
    print("Showing one evaluation task:")
    show_arc_task(f"{data_dir}/arc-agi_evaluation_challenges.json")
