"""Typed calls into the exact level solver."""

import ctypes
from pathlib import Path

import numpy as np

from runner.compiled_kernels import C_FLAGS, load
from runner.lane import LANE

HERE = Path(__file__).parent
THREADS = HERE / f"threads_{LANE}"
_TYPED = {}


def library():
    lib = load(
        "level-solver",
        [HERE / "level_solver.c", HERE / "sheet_flattening.c"],
        (*C_FLAGS, "-pthread", f"-I{THREADS}"),
        headers=[THREADS / "parallel_blocks.h"],
    )
    if id(lib) in _TYPED:
        return lib
    i32, i64, u64, f32, f64, ptr = (
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_uint64,
        ctypes.c_float,
        ctypes.c_double,
        ctypes.c_void_p,
    )
    signatures = {
        "hx_level_minimize": ([i64, i64, ptr, ptr, ptr, ptr, ptr, ptr, i32, ptr], i64),
        "hx_level_two_labels": ([i64, i64, ptr, ptr, ptr, ptr, ptr], i64),
        "hx_level_terms": ([i64, ptr, ptr, ptr, i64, ptr, ptr, i64, ptr, i32, ptr, ptr], i64),
        "hx_level_terms_take": ([u64, ptr, ptr, ptr], None),
        "hx_level_weights": ([i64, ptr, i64, ptr, ptr, i64, f32, ptr, i32, ptr, ptr], None),
        "hx_level_votes": (
            [i64, i64, ptr, ptr, ptr, ptr, i64, ptr, ptr, i64, f32, ptr, ptr, i32, i32, i32, i32, i32, u64, ptr, ptr],
            i64,
        ),
        "hx_level_flatten": ([ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, f64, f64, i32], None),
        "hx_level_blocks": ([i64, ptr, ptr, ptr, i64, f32, ptr], i64),
        "hx_level_block_pairs": ([i64, ptr, ptr, ptr, ptr, i64, i32, ptr], i64),
        "hx_level_domains": ([i64, ptr, ptr, ptr, i64, ptr, ptr, ptr], i64),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result
    _TYPED[id(lib)] = True
    return lib


def minimize(count, p, q, k, up, dn, x, first_step=1):
    """x at a global minimum of sum up*max(r,0) + dn*max(-r,0), r = x[q] - x[p] - k."""
    x = np.array(x, dtype=np.int32)
    log = np.zeros(3, dtype=np.int64)
    if (
        library().hx_level_minimize(
            count,
            len(p),
            p.ctypes.data,
            q.ctypes.data,
            k.ctypes.data,
            up.ctypes.data,
            dn.ctypes.data,
            x.ctypes.data,
            first_step,
            log.ctypes.data,
        )
        < 0
    ):
        raise RuntimeError("flow proof failed")
    return x, log


def two_labels(count, a, b, weight, unary):
    """Labels of one minimum cut."""
    label = np.zeros(count, dtype=np.uint8)
    library().hx_level_two_labels(
        count, len(a), a.ctypes.data, b.ctypes.data, weight.ctypes.data, unary.ctypes.data, label.ctypes.data
    )
    return label.astype(bool)


def level_terms(patch, inner, count, edges, rays, ray_weight, theta, chirality):
    """(p, q, k, each element's term), elements being edges then rays."""
    lib = library()
    term = np.empty(len(edges) + len(rays), dtype=np.int32)
    handle = np.zeros(1, dtype=np.uint64)
    terms = lib.hx_level_terms(
        count,
        patch.ctypes.data,
        inner.ctypes.data,
        edges.ctypes.data,
        len(edges),
        rays.ctypes.data,
        ray_weight.ctypes.data,
        len(rays),
        theta.ctypes.data,
        int(chirality),
        term.ctypes.data,
        handle.ctypes.data,
    )
    p, q, k = (np.empty(terms, dtype=np.int32) for _ in range(3))
    lib.hx_level_terms_take(int(handle[0]), p.ctypes.data, q.ctypes.data, k.ctypes.data)
    return p, q, k, term


def term_weights(terms, term, edges, ray_weight, hidden, scale, ray_up=None):
    """Integer costs each way of every term; a ray's cost of skipping sheets is ray_up when given, else hidden times
    its weight."""
    up, dn = np.empty(terms, dtype=np.int32), np.empty(terms, dtype=np.int32)
    skip = None if ray_up is None else np.ascontiguousarray(ray_up, dtype=np.float32)
    library().hx_level_weights(
        terms,
        edges.ctypes.data,
        len(edges),
        ray_weight.ctypes.data,
        None if skip is None else skip.ctypes.data,
        len(ray_weight),
        hidden,
        term.ctypes.data,
        scale,
        up.ctypes.data,
        dn.ctypes.data,
    )
    return up, dn


def votes(
    count,
    p,
    q,
    k,
    term,
    edges,
    ray_weight,
    hidden,
    x0,
    draws,
    threads,
    scale,
    seed,
    boot=True,
    join_shape=0,
    ray_up=None,
):
    """How many reweighted solves keep each term."""
    kept = np.zeros(len(p), dtype=np.int32)
    log = np.zeros(1, dtype=np.int64)
    up = None if ray_up is None else np.ascontiguousarray(ray_up, dtype=np.float32)
    failed = library().hx_level_votes(
        count,
        len(p),
        p.ctypes.data,
        q.ctypes.data,
        k.ctypes.data,
        edges.ctypes.data,
        len(edges),
        ray_weight.ctypes.data,
        None if up is None else up.ctypes.data,
        len(ray_weight),
        hidden,
        term.ctypes.data,
        x0.ctypes.data,
        draws,
        threads,
        scale,
        int(boot),
        join_shape,
        seed,
        kept.ctypes.data,
        log.ctypes.data,
    )
    if failed:
        raise RuntimeError("flow proof failed")
    return kept


def _pointers(runs):
    return np.array([run.ctypes.data for run in runs], dtype=np.int64), np.array(
        [len(run) for run in runs], dtype=np.int64
    )


def blocks(nodes, runs, side):
    """Each node's block, nodes joined by agreeing edges inside one cube of the given side."""
    starts, lengths = _pointers(runs)
    block = np.empty(len(nodes), dtype=np.int32)
    count = library().hx_level_blocks(
        len(nodes), nodes.ctypes.data, starts.ctypes.data, lengths.ctypes.data, len(runs), side, block.ctypes.data
    )
    return block, count


def block_pairs(block, count, nodes, runs, units):
    """Pairs of blocks joined by agreeing edges, and each pair's summed integer weight."""
    lib = library()
    starts, lengths = _pointers(runs)
    handle = np.zeros(1, dtype=np.uint64)
    pairs = lib.hx_level_block_pairs(
        count,
        block.ctypes.data,
        nodes.ctypes.data,
        starts.ctypes.data,
        lengths.ctypes.data,
        len(runs),
        units,
        handle.ctypes.data,
    )
    a, b, weight = (np.empty(pairs, dtype=np.int32) for _ in range(3))
    lib.hx_level_terms_take(int(handle[0]), a.ctypes.data, b.ctypes.data, weight.ctypes.data)
    return a, b, weight


def domains(nodes, runs, flip):
    """Flipped nodes worth keeping."""
    starts, lengths = _pointers(runs)
    keep = np.zeros(len(nodes), dtype=np.uint8)
    log = np.zeros(1, dtype=np.int64)
    kept = library().hx_level_domains(
        len(nodes),
        nodes.ctypes.data,
        starts.ctypes.data,
        lengths.ctypes.data,
        len(runs),
        flip.ctypes.data,
        keep.ctypes.data,
        log.ctypes.data,
    )
    return keep.astype(bool), kept, int(log[0])


def flatten(nodes, order, piece_first, ea, eb, edge_first, steps, tolerance, held, threads):
    """(sign, u, v) of every node of the pieces, each laid flat in frames carried along its sheet."""
    local, sign = np.zeros(len(nodes), dtype=np.int32), np.zeros(len(nodes), dtype=np.int8)
    u, v = np.zeros(len(nodes), dtype=np.float32), np.zeros(len(nodes), dtype=np.float32)
    library().hx_level_flatten(
        nodes.ctypes.data,
        order.ctypes.data,
        piece_first.ctypes.data,
        len(piece_first) - 1,
        ea.ctypes.data,
        eb.ctypes.data,
        edge_first.ctypes.data,
        local.ctypes.data,
        sign.ctypes.data,
        u.ctypes.data,
        v.ctypes.data,
        steps,
        tolerance,
        held,
        threads,
    )
    return sign, u, v
