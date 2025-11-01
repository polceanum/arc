# arc_data.py
import json, torch
from typing import Iterator, Tuple, Dict

def iter_arc_train_pairs(json_path: str) -> Iterator[Tuple[str, Dict]]:
    data = json.load(open(json_path, "r"))
    for task_id, task in data.items():
        for pair in task.get("train", []):
            yield task_id, pair

def to_tensor_grid(pair) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(pair["input"], dtype=torch.int64)
    y = torch.tensor(pair["output"], dtype=torch.int64)
    return x, y

def iter_arc_train_pairs_same_shape(json_path: str) -> Iterator[Tuple[str, Dict]]:
    """Yield only pairs with input/output having identical HxW."""
    data = json.load(open(json_path, "r"))
    for task_id, task in data.items():
        for pair in task.get("train", []):
            hin, win = len(pair["input"]), len(pair["input"][0])
            hout, wout = len(pair["output"]), len(pair["output"][0])
            if (hin, win) == (hout, wout):
                yield task_id, pair
