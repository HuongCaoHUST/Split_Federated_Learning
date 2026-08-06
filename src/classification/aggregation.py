import torch
import torch.nn as nn


def _load_checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type in {path}: {type(checkpoint)}")
    for key in ("model_state_dict", "state_dict", "model"):
        if key not in checkpoint:
            continue
        state = checkpoint[key]
        return state.state_dict() if isinstance(state, nn.Module) else state
    return checkpoint


def _canonical_state_items(state, layer_offset=0):
    for key, value in state.items():
        clean_key = key.removeprefix("module.").removeprefix("layers.")
        parts = clean_key.split(".")
        if not parts or not parts[0].isdigit():
            continue
        global_layer = int(parts[0]) + layer_offset
        target_key = ".".join(["layers", str(global_layer)] + parts[1:])
        yield global_layer, target_key, value


def _weighted_average(entries, target):
    total_weight = sum(entry[1] for entry in entries)
    if total_weight <= 0:
        raise ValueError("Cannot average a state with zero batch exposure.")
    if target.is_floating_point() or target.is_complex():
        dtype = torch.complex128 if target.is_complex() else torch.float64
        result = torch.zeros_like(target, dtype=dtype, device="cpu")
        for value, weight in entries:
            result.add_(value.to(dtype=dtype, device="cpu"), alpha=weight)
        return (result / total_weight).to(dtype=target.dtype)

    result = torch.zeros_like(target, dtype=torch.float64, device="cpu")
    for value, weight in entries:
        result.add_(value.to(dtype=torch.float64, device="cpu"), alpha=weight)
    result /= total_weight
    if target.dtype == torch.bool:
        return (result >= 0.5).to(dtype=target.dtype)
    return result.round().to(dtype=target.dtype)


def merge_classification_models(
    full_model,
    edge_models,
    server_models,
    minimum_cut_layer,
):
    """Merge dynamic-cut partitions using batches that traversed each copy."""
    if not edge_models or not server_models:
        raise ValueError("Both edge and server model updates are required.")
    total_batches = sum(int(model["num_batches"]) for model in edge_models)
    if total_batches <= 0:
        raise ValueError("Total edge batch count must be positive.")

    fallback_routes = {}
    for model in edge_models:
        cut = int(model["cut_layer"])
        count = int(model["num_batches"])
        fallback_routes[cut] = fallback_routes.get(cut, 0) + count

    full_state = full_model.state_dict()
    candidates = {}
    server_routes = []

    def collect(state, offset, exposure_for_layer):
        for global_layer, target_key, value in _canonical_state_items(state, offset):
            if target_key not in full_state:
                continue
            if value.shape != full_state[target_key].shape:
                raise ValueError(f"Shape mismatch for aggregated state {target_key}.")
            exposure = int(exposure_for_layer(global_layer))
            if exposure <= 0:
                continue
            candidates.setdefault(target_key, []).append((value, exposure))

    for model in edge_models:
        state = _load_checkpoint_state(model["path"])
        cut = int(model["cut_layer"])
        batches = int(model["num_batches"])
        collect(state, 0, lambda layer, c=cut, n=batches: n if layer <= c else 0)

    for model in server_models:
        routes = {
            int(cut): int(count)
            for cut, count in model.get("route_batch_counts", {}).items()
        }
        if not routes:
            if len(server_models) != 1:
                raise ValueError(
                    "route_batch_counts is required with multiple server workers."
                )
            routes = fallback_routes
        server_routes.append(routes)
        state = _load_checkpoint_state(model["path"])
        collect(
            state,
            minimum_cut_layer + 1,
            lambda layer, counts=routes: sum(
                count for cut, count in counts.items() if cut < layer
            ),
        )

    global_layers = {
        int(key.split(".")[1])
        for key in full_state
        if key.startswith("layers.")
    }
    exposure_by_layer = {
        layer: sum(
            int(model["num_batches"])
            for model in edge_models
            if layer <= int(model["cut_layer"])
        )
        + sum(
            count
            for routes in server_routes
            for cut, count in routes.items()
            if cut < layer
        )
        for layer in global_layers
    }
    invalid_exposure = {
        layer: exposure for layer, exposure in exposure_by_layer.items()
        if exposure != total_batches
    }
    if invalid_exposure:
        raise ValueError(
            f"Layer exposure must equal {total_batches}: {invalid_exposure}."
        )

    missing = [key for key in full_state if key not in candidates]
    if missing:
        raise ValueError(f"Aggregation has no source for: {missing[:5]}.")

    merged_state = {
        key: _weighted_average(entries, full_state[key])
        for key, entries in candidates.items()
    }
    full_model.load_state_dict(merged_state, strict=True)
    return full_model
