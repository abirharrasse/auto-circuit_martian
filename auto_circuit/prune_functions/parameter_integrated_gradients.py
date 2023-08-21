from enum import Enum
from itertools import chain
from typing import Dict

import torch as t
from ordered_set import OrderedSet
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import Edge
from auto_circuit.utils import graph_edges


class BaselineWeights(Enum):
    """The baseline weights to use for integrated gradients."""

    ZERO = 0
    MEAN = 1
    PERMUTED = 2


def parameter_integrated_grads_prune_scores(
    model: t.nn.Module,
    train_data: DataLoader[PromptPairBatch],
    baseline_weights: BaselineWeights,
    samples: int = 50,
) -> Dict[Edge, float]:
    """Gradients of weights wrt to network output, integrated between some baseline
    weights and the actual weights."""

    edges: OrderedSet[Edge] = graph_edges(model)
    weights = OrderedSet(chain(*[[e.src.weight, e.dest.weight] for e in edges]))
    print("weights", weights)
    normal_state_dict = {}
    for name, param in model.state_dict().items():
        if name in weights:
            assert isinstance(param, t.Tensor)
            normal_state_dict[name] = param.clone()

    baseline_state_dict = {}
    if baseline_weights == BaselineWeights.ZERO:
        baseline_state_dict = dict(
            [(name, t.zeros_like(param)) for name, param in normal_state_dict.items()]
        )
    else:
        raise NotImplementedError

    integrated_grads = dict(
        [(name, t.zeros_like(param)) for name, param in normal_state_dict.items()]
    )
    for idx in range(samples):
        interpolated_state_dict = {}
        for name in weights:
            interpolated_state_dict[name] = baseline_state_dict[name] + (
                normal_state_dict[name] - baseline_state_dict[name]
            ) * idx / (samples - 1)
            model.load_state_dict(interpolated_state_dict, strict=False)

        model.zero_grad()
        for batch in train_data:
            out = model(batch.clean)
            loss = out.mean()
            loss.backward()

        for name, param in model.named_parameters():
            if name in weights:
                assert isinstance(param.grad, t.Tensor)
                integrated_grads[name] += param.grad / samples

    weight_diffs = dict(
        [
            (name, normal_param - baseline_state_dict[name])
            for name, normal_param in normal_state_dict.items()
        ]
    )
    for name, ig in integrated_grads.items():
        integrated_grads[name] = ig * weight_diffs[name]

    prune_scores = {}
    for edge in edges:
        try:
            src_ig = integrated_grads[edge.src.weight].abs()
            src_ig = src_ig[edge.src.weight_t_idx]
            dest_ig = integrated_grads[edge.dest.weight].abs()
            dest_ig = dest_ig[edge.dest.weight_t_idx]
        except Exception as E:
            print(f"Error on edge {edge}")
            print(
                "edge.src.weight",
                edge.src.weight,
                model.state_dict()[edge.src.weight].shape,
                "edge.src.weight_t_idx",
                edge.src.weight_t_idx,
            )
            print(
                "edge.dest.weight",
                edge.dest.weight,
                model.state_dict()[edge.dest.weight].shape,
                "edge.dest.weight_t_idx",
                edge.dest.weight_t_idx,
            )
            raise E
        prune_scores[edge] = (src_ig.sum().item() + dest_ig.sum().item()) / 2

    return prune_scores
