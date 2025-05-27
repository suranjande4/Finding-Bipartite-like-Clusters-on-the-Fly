import networkx as nx
import matplotlib.pyplot as plt
import scipy
import localgraphclustering as lgc
import stag.graph
import numpy as np
import time
import numpy.random as npr
import seaborn as sns
import math
import pandas as pd







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
    Given a sparse vector (presumably from an approximate pagerank calculation on the double cover),
    and the number of vertices in the original graph, compute the 'simplified' approximate pagerank vector.
    """
    # Initialise the new sparse vector
    new_vector = scipy.sparse.lil_matrix((2 * num_g_vertices, 1))

    # Iterate through the entries in the matrix
    for i in range(min(num_g_vertices, sparse_vector.shape[0] - num_g_vertices)):
        if sparse_vector[i, 0] > sparse_vector[i + num_g_vertices, 0]:
            new_vector[i, 0] = sparse_vector[i, 0] - sparse_vector[i + num_g_vertices, 0]
        elif sparse_vector[i + num_g_vertices, 0] > sparse_vector[i, 0]:
            new_vector[i + num_g_vertices, 0] = sparse_vector[i + num_g_vertices, 0] - sparse_vector[i, 0]

    return new_vector.tocsc()


def local_bipart_dc(g: stag.graph.Graph, start_vertex: int, alpha: float, eps: float):
    """
    An implementation of the local_bipart_dc algorithm using the STAG library.
    """
    # Now, we construct the double cover graph of g
    adj_mat = g.adjacency().to_scipy()
    identity = scipy.sparse.csc_matrix((g.number_of_vertices(), g.number_of_vertices()))
    double_cover_adj = scipy.sparse.bmat([[identity, adj_mat], [adj_mat, identity]])
    h = stag.graph.Graph(double_cover_adj)

    # Run the approximate pagerank on the double cover graph
    seed_vector = scipy.sparse.lil_matrix((h.number_of_vertices(), 1))
    seed_vector[start_vertex, 0] = 1
    p, r = stag.cluster.approximate_pagerank(h, seed_vector.tocsc(), alpha, eps)

    # Compute the simplified pagerank vector
    p_simplified = simplify(g.number_of_vertices(), p.to_scipy())

    # Compute the sweep set in the double cover
    sweep_set = stag.cluster.sweep_set_conductance(h, p_simplified)
    bipartiteness = stag.cluster.conductance(h, sweep_set)

    # Split the returned vertices into those in the same cluster as the seed, and others.
    this_cluster = [i for i in sweep_set if i < g.number_of_vertices()]
    that_cluster = [i - g.number_of_vertices() for i in sweep_set if i >= g.number_of_vertices()]
    return this_cluster, that_cluster, bipartiteness

edgelist_file = "Realworld_graph_bulider/Mid_data_graph_construction/example_graph_from_1900_1950/dyadic_mid_1900_1950.edgelist"

# Create an undirected graph from the edgelist
G = nx.read_edgelist(edgelist_file, nodetype=int, data=(('weight', float),), create_using=nx.Graph())
H = edge_sampling_algorithm(G, 1, 1)




starting_vertex = 0



start_time=time.time()
L, R, bipartiteness = local_bipart_dc(G, starting_vertex, 0.5, 4e-7)
end_time=time.time()
#print(f" Iteration: {ind + 1} ")
print("--------------------------------------------------------------------------------------------------")
#print(f"LocBipartDC with {n} nodes")
print(f"bipartiteness Ratio: {bipartiteness:.3f}")
#pt_flr.append(bipartiteness)
end_time=time.time()
print(f"Time taken : {end_time - start_time : .4f} secs")
#pt_time.append(end_time - start_time )
print(" ")
#print(f"Cluster One: {sorted(L)}")
#print(f"Cluster Two: {sorted(R)}")
#print(f"Bipartiteness: {bipartiteness:.3f}")


H = stag.graph.from_networkx(H)


start_time=time.time()
L, R, bipartiteness = local_bipart_dc(H, starting_vertex, 0.5, 4e-7)
#print(f"Cluster One: {sorted(L)}")
#print(f"Cluster Two: {sorted(R)}")
#print(f"Bipartiteness: {bipartiteness:.3f}")
print(f"bipartiteness Ratio: {bipartiteness:.3f}")
#our_flr.append(bipartiteness)
end_time=time.time()
print(f"Time taken : {end_time - start_time : .4f} secs")
#our_time.append(end_time - start_time )

print("---------------------------------------------------------------------------------------------------------------")
print(" ")
