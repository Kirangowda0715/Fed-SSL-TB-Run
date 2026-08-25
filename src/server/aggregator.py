"""
src/server/aggregator.py
------------------------
Federated aggregation strategies: FedAvg and FedProx.

Design Notes:
  - FedAvg  : Weighted average of local encoder state_dicts
  - FedProx : Same aggregation as FedAvg on the SERVER side.
              The proximal regularization term (μ/2 * ||w - w_global||²)
              is added in the CLIENT's loss inside ssl_train.py.
"""

import copy
from typing import List, Dict, Any


def fedavg(
    encoder_weights_list: List[Dict[str, Any]],
    sample_counts: List[int],
) -> Dict[str, Any]:
    """
    Federated Averaging aggregation.

    Computes a weighted average of local encoder state_dicts,
    where the weight of each hospital is proportional to its sample count.

    Args:
        encoder_weights_list : List of encoder state_dicts from each hospital
        sample_counts        : List of sample counts (one per hospital)

    Returns:
        Aggregated encoder state_dict
    """
    assert len(encoder_weights_list) == len(sample_counts), (
        "Number of weight dicts must match number of sample counts."
    )
    assert len(encoder_weights_list) > 0, "No weights to aggregate."

    total_samples = sum(sample_counts)
    if total_samples == 0:
        raise ValueError("Total sample count is 0 — cannot compute weighted average.")

    # Compute per-hospital weights
    weights = [n / total_samples for n in sample_counts]

    # Recurse through encoder, decoder, and proto_head component state dicts.
    if "encoder" in encoder_weights_list[0]:
        return {
            component: fedavg(
                [weights[component] for weights in encoder_weights_list], sample_counts
            )
            for component in encoder_weights_list[0]
        }

    # Initialize aggregated state_dict with zeros (same structure as first hospital)
    agg_weights = copy.deepcopy(encoder_weights_list[0])
    for key in agg_weights:
        agg_weights[key] = agg_weights[key].float() * weights[0]

    # Accumulate weighted contributions from remaining hospitals
    for i in range(1, len(encoder_weights_list)):
        w = weights[i]
        for key in agg_weights:
            if encoder_weights_list[i][key].dtype.is_floating_point:
                agg_weights[key] += encoder_weights_list[i][key].float() * w
            else:
                agg_weights[key] = encoder_weights_list[0][key].clone()

    return agg_weights


def fedprox(
    global_weights: Dict[str, Any],
    local_weights_list: List[Dict[str, Any]],
    sample_counts: List[int],
    mu: float = 0.01,
) -> Dict[str, Any]:
    """
    FedProx aggregation (server side).

    On the SERVER side, FedProx uses the same weighted aggregation as FedAvg.
    The proximal regularization term has already been applied in each hospital's
    local training loss (in src/client/ssl_train.py):
        loss += (mu / 2) * ||w_local - w_global||²

    Args:
        global_weights    : Current global encoder state_dict (reference weights)
        local_weights_list: List of local encoder state_dicts
        sample_counts     : Sample counts per hospital
        mu                : Proximal term coefficient (informational only here)

    Returns:
        Aggregated encoder state_dict (FedAvg-style)
    """
    # Server-side aggregation is identical to FedAvg
    # The proximal correction is baked into local training, not aggregation
    aggregated = fedavg(local_weights_list, sample_counts)
    return aggregated
