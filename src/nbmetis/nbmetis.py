"""Low level bindings."""

import random
from typing import Annotated

from . import _nbmetis as metis

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix

if metis.SIZEOF_IDX_T == 64:
    idx_t = np.int64
    idx_t_dtype_str = "int64"
else:
    idx_t = np.int32
    idx_t_dtype_str = "int32"

if metis.SIZEOF_REAL_T == 64:
    real_t = np.float32
    real_t_dtype_str = "float32"
else:
    real_t = np.float64
    real_t_dtype_str = "float64"

IdxA = Annotated[ndarray, dict(dtype=idx_t_dtype_str, shape=(None), device="cpu")]
RealA = Annotated[ndarray, dict(dtype=real_t_dtype_str, shape=(None), device="cpu")]

IdxA_NULL: IdxA = np.zeros(shape=(0,), dtype=idx_t)
RealA_NULL: RealA = np.zeros(shape=(0,), dtype=real_t)


def part_graph(
    num_nodes: int,
    num_edges: int,
    num_parts: int,
    weighted_adjacency: csr_matrix,
    vertex_weights: ndarray | None,
    vertex_size: ndarray | None,
    target_parition_weights: ndarray | None,
    allowed_load_imbalance: ndarray | None,
    options: ndarray | None,
    part_kway: bool,
) -> tuple[int, ndarray]:
    assert weighted_adjacency.shape == (num_nodes, num_nodes)
    assert weighted_adjacency.indptr.shape == (num_nodes + 1,)
    assert weighted_adjacency.indices.shape == (2 * num_edges,)

    if vertex_weights is not None:
        assert len(vertex_weights.shape) == 2
        assert vertex_weights.shape[0] == num_nodes
    if vertex_size is not None:
        assert vertex_size.shape == (num_nodes,)

    xadj = weighted_adjacency.indptr.astype(idx_t, copy=False)
    adjncy = weighted_adjacency.indices.astype(idx_t, copy=False)
    if vertex_weights is None:
        vwgt = IdxA_NULL
    else:
        vwgt = vertex_weights.astype(idx_t, copy=False)
    if vertex_size is None:
        vsize = IdxA_NULL
    else:
        vsize = vertex_size.astype(idx_t, copy=False)

    adjwgt = weighted_adjacency.data.astype(idx_t, copy=False)

    if target_parition_weights is None:
        tpwgts = RealA_NULL
    else:
        tpwgts = target_parition_weights.astype(real_t, copy=False)
    if allowed_load_imbalance is None:
        ubvec = RealA_NULL
    else:
        ubvec = allowed_load_imbalance.astype(real_t, copy=False)

    if options is None:
        options = IdxA_NULL
    else:
        options = options.astype(idx_t, copy=False)

    part = np.zeros(shape=(num_nodes,), dtype=idx_t)

    ret, edgecut = metis.PartGraph(
        nparts=num_parts,
        xadj=xadj,
        adjncy=adjncy,
        vwgt=vwgt,
        vsize=vsize,
        adjwgt=adjwgt,
        tpwgts=tpwgts,
        ubvec=ubvec,
        options=options,
        part=part,
        part_kway=int(part_kway),
    )

    assert ret == metis.OK

    return edgecut, part


def node_nd(
    num_nodes: int,
    num_edges: int,
    weighted_adjacency: csr_matrix,
    vertex_weights: ndarray | None,
    options: ndarray | None,
) -> tuple[ndarray, ndarray]:
    assert weighted_adjacency.shape == (num_nodes, num_nodes)
    assert weighted_adjacency.indptr.shape == (num_nodes + 1,)
    assert weighted_adjacency.indices.shape == (2 * num_edges,)

    if vertex_weights is not None:
        assert vertex_weights.shape == (num_nodes,)

    xadj = weighted_adjacency.indptr.astype(idx_t, copy=False)
    adjncy = weighted_adjacency.indices.astype(idx_t, copy=False)

    if vertex_weights is None:
        vwgt = IdxA_NULL
    else:
        vwgt = vertex_weights.astype(idx_t, copy=False)

    if options is None:
        options = IdxA_NULL
    else:
        options = options.astype(idx_t, copy=False)

    perm = np.zeros(shape=(num_nodes,), dtype=idx_t)
    iperm = np.zeros(shape=(num_nodes,), dtype=idx_t)

    ret = metis.NodeND(
        nvtxs=num_nodes,
        xadj=xadj,
        adjncy=adjncy,
        vwgt=vwgt,
        options=options,
        perm=perm,
        iperm=iperm,
    )
    assert ret == metis.OK

    return perm, iperm


def default_options_node_nd(verbose: bool = False) -> ndarray:
    options = np.zeros(shape=(metis.NOPTIONS), dtype=idx_t)
    ret = metis.SetDefaultOptions(options)
    assert ret == metis.OK

    options[metis.OPTION_OBJTYPE] = metis.OBJTYPE_NODE
    options[metis.OPTION_PTYPE] = metis.PTYPE_KWAY
    options[metis.OPTION_CTYPE] = metis.CTYPE_SHEM
    options[metis.OPTION_IPTYPE] = metis.IPTYPE_EDGE
    options[metis.OPTION_RTYPE] = metis.RTYPE_SEP1SIDED
    options[metis.OPTION_UFACTOR] = 30
    options[metis.OPTION_PFACTOR] = 0
    options[metis.OPTION_NO2HOP] = 0
    options[metis.OPTION_COMPRESS] = 1
    options[metis.OPTION_CCORDER] = 1
    options[metis.OPTION_NITER] = 10
    options[metis.OPTION_NSEPS] = 1
    options[metis.OPTION_NUMBERING] = 0
    options[metis.OPTION_SEED] = random.randint(0, 2**31 - 1)
    if verbose:
        options[metis.OPTION_DBGLVL] = metis.DBG_INFO
    else:
        options[metis.OPTION_DBGLVL] = 0


    return options
