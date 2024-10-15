"""Test low level nested disection."""

from nbmetis.nbmetis import node_nd, default_options_node_nd

import numpy as np
import networkx as nx
from hypothesis import given
from hypothesis.strategies import booleans, integers, floats


@given(
    n=integers(min_value=1, max_value=100),
    p=floats(min_value=0, max_value=1.0),
    custom_options=booleans(),
)
def test_erdos_renyi(n: int, p: float, custom_options: bool):
    g = nx.erdos_renyi_graph(n=n, p=p)

    num_nodes = g.number_of_nodes()
    num_edges = g.number_of_edges()

    weighted_adjacency = nx.adjacency_matrix(g).tocsr()

    if custom_options:
        options = default_options_node_nd()
    else:
        options = None

    perm, iperm = node_nd(
        num_nodes=num_nodes,
        num_edges=num_edges,
        weighted_adjacency=weighted_adjacency,
        vertex_weights=None,
        options=options,
    )

    xs = np.arange(num_nodes, dtype=int)

    assert np.all(perm[iperm] == xs)

    perm.sort()
    assert np.all(perm == xs)

    iperm.sort()
    assert np.all(iperm == xs)
