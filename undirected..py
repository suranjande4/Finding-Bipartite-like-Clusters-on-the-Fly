import networkx as nx
import matplotlib.pyplot as plt
import scipy
import localgraphclustering as lgc
import stag.graph
import stag.random
import stag.graphio
import stag.cluster
import numpy as np
import time
import numpy.random as npr
import seaborn as sns
from plots.plot_utils import *
from plots.plot_style import *


def edge_sampling_algorithm(G, C, lambda_k_plus_1):
    """
    Sample edges from graph G according to the specified probability function.

    Parameters
    ----------
    G : networkx.Graph
        Weighted input graph with edge weights as 'weight' attribute.
    C : float
        Positive constant used in probability calculation.
    lambda_k_plus_1 : float
        (k+1)-th smallest eigenvalue of normalized adjacency matrix of G.

    Returns
    -------
    H : networkx.Graph
        Weighted graph containing sampled edges with adjusted weights.
    """
    H = nx.Graph()
    n = G.number_of_nodes()
    log_n = np.log(n)

    for u, v, data in G.edges(data=True):
        w_uv = data.get('weight', 1)
        degree_u = G.degree(u, weight='weight')
        degree_v = G.degree(v, weight='weight')

        p_u_v = min(C * (log_n**3 / lambda_k_plus_1) * (w_uv / degree_u), 1)
        p_v_u = min(C * (log_n**3 / lambda_k_plus_1) * (w_uv / degree_v), 1)

        p_e = p_u_v + p_v_u - (p_u_v * p_v_u)

        if np.random.rand() <= p_e:
            adjusted_weight = w_uv / p_e
            H.add_edge(u, v, weight=adjusted_weight)

    return H


def simplify(num_g_vertices: int, sparse_vector):
    """
    Simplify a sparse vector from approximate pagerank on the double cover.

    Parameters
    ----------
    num_g_vertices : int
        Number of vertices in the original graph.
    sparse_vector : scipy.sparse matrix
        Sparse vector from approximate pagerank on the double cover graph.

    Returns
    -------
    scipy.sparse.csc_matrix
        Simplified approximate pagerank vector.
    """
    new_vector = scipy.sparse.lil_matrix((2 * num_g_vertices, 1))

    limit = min(num_g_vertices, sparse_vector.shape[0] - num_g_vertices)
    for i in range(limit):
        val1 = sparse_vector[i, 0]
        val2 = sparse_vector[i + num_g_vertices, 0]

        if val1 > val2:
            new_vector[i, 0] = val1 - val2
        elif val2 > val1:
            new_vector[i + num_g_vertices, 0] = val2 - val1

    return new_vector.tocsc()


def local_bipart_dc(g: stag.graph.Graph, start_vertex: int, alpha: float, eps: float):
    """
    Run the local bipartite clustering algorithm on the double cover graph.

    Parameters
    ----------
    g : stag.graph.Graph
        Input graph.
    start_vertex : int
        Starting vertex for approximate pagerank.
    alpha : float
        Teleportation parameter for pagerank.
    eps : float
        Approximation accuracy.

    Returns
    -------
    tuple
        (this_cluster, that_cluster, bipartiteness ratio)
    """
    adj_mat = g.adjacency().to_scipy()
    identity = scipy.sparse.csc_matrix((g.number_of_vertices(), g.number_of_vertices()))
    double_cover_adj = scipy.sparse.bmat([[identity, adj_mat], [adj_mat, identity]])
    h = stag.graph.Graph(double_cover_adj)

    seed_vector = scipy.sparse.lil_matrix((h.number_of_vertices(), 1))
    seed_vector[start_vertex, 0] = 1

    p, r = stag.cluster.approximate_pagerank(h, seed_vector.tocsc(), alpha, eps)
    p_simplified = simplify(g.number_of_vertices(), p.to_scipy())

    sweep_set = stag.cluster.sweep_set_conductance(h, p_simplified)
    bipartiteness = stag.cluster.conductance(h, sweep_set)

    this_cluster = [i for i in sweep_set if i < g.number_of_vertices()]
    that_cluster = [i - g.number_of_vertices() for i in sweep_set if i >= g.number_of_vertices()]
    return this_cluster, that_cluster, bipartiteness


edge_pr = 0.3
pt_time = []
our_time = []
pt_flr = []
our_flr = []
n_nodes = [5000]

for ind, n in enumerate(n_nodes):
    G = stag.random.sbm(n, 2, edge_pr / 10, edge_pr)
    nx_graph = G.to_networkx()

    nx_graph = edge_sampling_algorithm(nx_graph, 1, 1)

    starting_vertex = 100
    start_time = time.time()
    L, R, bipartiteness = local_bipart_dc(G, starting_vertex, 0.5, 4e-7)
    end_time = time.time()
    print(f" Iteration: {ind + 1} ")
    print("--------------------------------------------------------------------------------------------------")
    print(f"Bipartiteness Ratio: {bipartiteness:.3f}")
    pt_flr.append(bipartiteness)
    print(f"Time taken by MS algorithm : {end_time - start_time : .4f} secs")
    pt_time.append(end_time - start_time)
    print(" ")

    H = stag.graph.from_networkx(nx_graph)
    start_time = time.time()
    L, R, bipartiteness = local_bipart_dc(H, starting_vertex, 0.5, 4e-7)
    print(f"bipartiteness Ratio: {bipartiteness:.3f}")
    our_flr.append(bipartiteness)
    end_time = time.time()
    print(f"Time taken MS+ Our algorithm: {end_time - start_time : .4f} secs")
    our_time.append(end_time - start_time)

    print("---------------------------------------------------------------------------------------------------------------")
    print(" ")

plot_runtime_and_bipartiteness(
    n_nodes, pt_time, our_time, pt_flr, our_flr,
    save_path="Unidirected plots"
)
