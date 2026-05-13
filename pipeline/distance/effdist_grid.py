"""Gridded effective distance on heterogeneous-spatial populations.

Extends Brockmann & Helbing (2013) effective-distance from the airport-pair
network to a 2-D grid where each populated cell is a node, edges connect
spatially-adjacent cells, and edge weights derive from a gravity model on
population × distance × optional per-cell capacity weights (SES, healthcare).

The effective distance from source s to target t is
    d_eff(s, t) = min over paths P  of  Σ -log P_step
where P_step is the probability of moving from one cell to its neighbour
under the gravity model. With the negative-log transform, this becomes a
shortest-path problem on a non-negative-weighted graph solvable by Dijkstra
in O((E + V log V)).

Per-cell SES weights enter multiplicatively on the inflow side, so a cell
with low detection capacity raises its incoming -log probability (it's
"further" in the effective-distance sense — harder to confirm a case there).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def build_grid_graph(
    pop: np.ndarray,
    *,
    radius_cells: int = 3,
    beta: float = 1.5,
    ses_weight: np.ndarray | None = None,
    min_pop: float = 1.0,
) -> tuple[csr_matrix, np.ndarray]:
    """Build a sparse gravity graph over populated grid cells.

    Args:
        pop: (H, W) population grid.
        radius_cells: Neighbourhood radius for edges, in cells. Larger means
                      longer edges (more connectivity, more compute).
        beta: Distance-decay exponent in the gravity model.
              flow_ij ∝ pop_i * pop_j / dist_ij^beta
        ses_weight: Optional (H, W) array of SES detection multipliers in
                    (0, 1]. Lower values raise effective distance into that
                    cell (lower detection probability ↔ longer effective
                    distance, since a case arriving there is less likely to
                    be confirmed and contribute to the empirical surface).
                    If None, all cells weight 1.
        min_pop: Cells below this population are excluded from the graph.

    Returns:
        (csgraph, node_indices) where
            csgraph: (N, N) sparse matrix of edge weights w_ij = -log P(i→j).
            node_indices: ndarray of length N mapping graph-row to flat grid index.
    """
    H, W = pop.shape
    flat_pop = pop.ravel()
    mask = flat_pop >= min_pop
    node_idx = np.flatnonzero(mask)
    N = len(node_idx)
    if N == 0:
        raise ValueError("No populated cells above min_pop threshold")

    # Lookup: flat-grid-index -> graph-node-index
    rev = -np.ones(H * W, dtype=np.int64)
    rev[node_idx] = np.arange(N)

    # Cell (row, col) for each node
    rows = node_idx // W
    cols = node_idx % W

    # Effective per-cell receiver weight: SES on inflow side
    if ses_weight is None:
        recv_w = np.ones(N, dtype=np.float32)
    else:
        flat_ses = ses_weight.ravel()
        recv_w = np.clip(flat_ses[node_idx].astype(np.float32), 1e-3, 1.0)

    # Build edges: for each node i, look in a (2r+1)^2 window of neighbours
    src_list, dst_list, w_list = [], [], []
    pop_i_arr = flat_pop[node_idx].astype(np.float32)

    for di in range(-radius_cells, radius_cells + 1):
        for dj in range(-radius_cells, radius_cells + 1):
            if di == 0 and dj == 0:
                continue
            # Euclidean cell distance (in cell units)
            dist = float(np.sqrt(di * di + dj * dj))
            # Candidate destination indices
            r2 = rows + di
            c2 = cols + dj
            ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
            dst_flat = r2[ok] * W + c2[ok]
            dst_node = rev[dst_flat]
            valid = dst_node >= 0
            if not valid.any():
                continue
            src_node = np.flatnonzero(ok)[valid]
            dst_node = dst_node[valid]

            # Gravity: pop_j / dist^beta, normalised to a probability later
            pop_j = flat_pop[node_idx[dst_node]]
            raw = pop_j / (dist ** beta) * recv_w[dst_node]
            src_list.append(src_node)
            dst_list.append(dst_node)
            w_list.append(raw)

    src_arr = np.concatenate(src_list)
    dst_arr = np.concatenate(dst_list)
    raw_arr = np.concatenate(w_list)

    # Normalise outflows to per-source probabilities, then convert to -log
    # Use a single pass groupby on src_arr
    order = np.argsort(src_arr, kind="stable")
    src_sorted = src_arr[order]
    dst_sorted = dst_arr[order]
    raw_sorted = raw_arr[order]

    out_sum = np.zeros(N, dtype=np.float32)
    np.add.at(out_sum, src_sorted, raw_sorted)
    # Avoid div-by-zero (isolated cells)
    out_sum_for_div = np.where(out_sum > 0, out_sum, 1.0)
    prob = raw_sorted / out_sum_for_div[src_sorted]
    prob = np.clip(prob, 1e-12, 1.0)
    weights = -np.log(prob)

    csgraph = csr_matrix(
        (weights, (src_sorted, dst_sorted)), shape=(N, N), dtype=np.float32
    )
    return csgraph, node_idx


def effective_distance_from(
    csgraph: csr_matrix,
    source_node: int,
) -> np.ndarray:
    """Single-source Dijkstra returning d_eff to every reachable node.

    Returns a length-N array; np.inf where unreachable.
    """
    return dijkstra(csgraph, directed=True, indices=source_node, return_predecessors=False)


def grid_index_for_lonlat(
    lon: float, lat: float, transform, shape: tuple[int, int]
) -> int:
    """Convert a (lon, lat) point to a flat grid index using a rasterio Affine."""
    import rasterio
    col, row = ~transform * (lon, lat)
    col = int(col)
    row = int(row)
    H, W = shape
    if not (0 <= row < H and 0 <= col < W):
        raise ValueError(f"Point ({lon}, {lat}) is outside grid bounds")
    return row * W + col


def graph_to_grid(values: np.ndarray, node_idx: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Scatter per-node values back to a full (H, W) grid; NaN where unmasked."""
    out = np.full(shape[0] * shape[1], np.nan, dtype=np.float32)
    out[node_idx] = values
    return out.reshape(shape)
