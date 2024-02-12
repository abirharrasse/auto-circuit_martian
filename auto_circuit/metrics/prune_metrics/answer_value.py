from typing import Any, Literal

import torch as t

from auto_circuit.tasks import Task
from auto_circuit.types import CircuitOutputs, Measurements
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.tensor_ops import batch_avg_answer_val


def identity(*args: Any, **kwargs: Any) -> Any:
    return args[0]


def measure_answer_val(
    task: Task,
    pruned_outs: CircuitOutputs,
    prob_func: Literal["log_softmax", "softmax", "logits"] = "logits",
) -> Measurements:
    measurements = []
    if prob_func == "softmax":
        apply_prob_func = t.nn.functional.softmax
    elif prob_func == "log_softmax":
        apply_prob_func = t.nn.functional.log_softmax
    else:
        assert prob_func == "logits"
        apply_prob_func = identity

    for edge_count, pruned_out in (pruned_out_pbar := tqdm(pruned_outs.items())):
        pruned_out_pbar.set_description_str(f"Answer Value for {edge_count} edges")
        avg_ans_probs = []
        for batch in task.test_loader:
            batch_probs = apply_prob_func(pruned_out[batch.key], dim=-1)
            avg_ans_probs.append(batch_avg_answer_val(batch_probs, batch))
        measurements.append((edge_count, t.stack(avg_ans_probs).mean().item()))
    return measurements