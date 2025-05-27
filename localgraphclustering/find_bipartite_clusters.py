"""
This file gives several methods for finding an almost-bipartite set.
Some of the methods are local in the sense that they will find a set close to some seed vertex in a graph and run in
time proportional to the size of the output set and independent of the size of the graph.
"""
from .cpp import *
from .sweep_cut import sweep_cut_dc, sweep_cut_dc_from_signed_vec
from .algorithms import eig_nL
import scipy as sp
import scipy.sparse
import math
import random
import numpy.random as npr
import networkx as nx




def edge_sampling_algorithm(G, C, lambda_k_plus_1):
    """
    Sample edges from the input graph G based on the specified probability function and return a new graph H.

    Parameters:
    - G: NetworkX graph (weighted) where the edge weights are stored as `weight` attributes.
    - C: Positive constant used in the probability function.
    - lambda_k_plus_1: The (k+1)-th smallest eigenvalue of the normalized adjacency matrix of G.

    Returns:
    - H: A new weighted NetworkX graph containing the sampled edges.
    """
    H = nx.Graph()  # Create an empty graph to store the sampled edges
    
    n = G.number_of_nodes()
    log_n = np.log(n)  # Compute log(n)

    # Compute the probability for each edge and sample
    for u, v, data in G.edges(data=True):
        w_uv = data.get('weight', 1)  # Default weight is 1 if not specified
        degree_u = G.degree(u, weight='weight')  # Weighted degree of u
        degree_v = G.degree(v, weight='weight')  # Weighted degree of v

        # Calculate p_u(v) and p_v(u) based on the formula given
        p_u_v = min(C * (log_n**3 / ( lambda_k_plus_1)) * (w_uv / degree_u), 1)
        p_v_u = min(C * (log_n**3  / (lambda_k_plus_1)) * (w_uv / degree_v), 1)

        # Compute the probability of retaining the edge
        p_e = p_u_v + p_v_u - (p_u_v * p_v_u)

        # Sample the edge with probability p_e
        if np.random.rand() <= p_e:
            # Add the edge to H with the adjusted weight
            adjusted_weight = w_uv / p_e
            H.add_edge(u, v, weight=adjusted_weight)

    return H

def lp_almost_bipartite(G, starting_vertex, T=100, xi_0=0.01):
    """
    Find an almost-bipartite set close to starting_vertex, using the truncated power method algorithm given by
    Li and Peng.

    :param G: a GraphLocal object on which to perform the algorithm
    :param starting_vertex: the index of the vertex at which to start the algorithm
    :param T: the number of iterations of the power method to perform.
    :param xi_0: the starting threshold for truncating the vectors.
    :return: A tuple containing:
      - L - the vertices in the left set
      - R - the vertices in the right set
      - bipart - the bipartiteness of the resulting set.
    """
    # Get the pseudo-laplacian of the graph for use in the power method
    n = G.adjacency_matrix.shape[0]
    M = G.rw_laplacian

    # Construct the starting vector for the power method
    r = sp.sparse.csc_matrix((n, 1))
    r[starting_vertex] = 1

    best_bipartiteness = 1
    best_L = [starting_vertex]
    best_R = []
    xi_t = xi_0
    for t in range(T):
        # Perform the matrix product
        q_t = M.dot(r).asformat('csc')

        # Truncate the new vector
        indices = []
        data = []
        new_data_length = 0
        for i, v in enumerate(q_t.data):
            if abs(v) > xi_t:
                data.append(v)
                indices.append(q_t.indices[i])
                new_data_length += 1
        indptr = [0, new_data_length]
        r = sp.sparse.csc_matrix((data, indices, indptr), (n, 1))

        # Run the bipartiteness sweep set
        L, R, bipart = sweep_cut_dc_from_signed_vec(G, indices, data, normalise_by_degree=True)
        if bipart < best_bipartiteness:
            best_bipartiteness = bipart
            best_R = R
            best_L = L

        # Increment the value of the truncation parameter
        xi_t = 2 * xi_t

    return best_L, best_R, best_bipartiteness


def local_bipartite_dc(G, starting_vertex, alpha=0.1, epsilon=1e-5, max_iterations=1000000):
    """
    Find an almost-bipartite set close to starting_vertex, using the double cover pagerank algorithm.

    :param G: a GraphLocal object on which to perform the algorithm.
    :param starting_vertex: the vertex id at which to start the algorithm.
    :param alpha: the alpha parameter for the approximate pagerank computation
    :param epsilon: the epsilon parameter for the approximate pagerank computation
    :param max_iterations: the maximum number of iterations for the pagerank computation
    :return: A tuple containing:
      - L - the vertices in the left set
      - R - the vertices in the right set
      - bipart - the bipartiteness of the resulting set.
    """
    # If alpha is equal to 0, then this algorithm is not defined
    if alpha == 0:
        raise AssertionError("Parameter alpha cannot be 0 for double cover pagerank algorithm.")

    # First, compute the approximate pagerank on the double cover of the graph
    # The result is simplified before being returned.
    n = G.adjacency_matrix.shape[0]

    if G.weighted:
        x_ind_1, x_ind_2, values_1, values_2 = dcpagerank_weighted_cpp(n, G.ai, G.aj, G.adjacency_matrix.data,
                                                                       alpha, epsilon, [starting_vertex],
                                                                       max_iterations, xlength=n)
    else:
        x_ind_1, x_ind_2, values_1, values_2 = dcpagerank_cpp(n, G.ai, G.aj, alpha, epsilon, [starting_vertex],
                                                              max_iterations, xlength=n)

    # Perform the sweep set procedure on the pagerank vector on the double cover of the graph.
    sweepset_dc, conductance_dc = sweep_cut_dc(G, x_ind_1, x_ind_2, values_1, values_2, normalise_by_degree=True)

    # Split the sweep set into L and R
    L = []
    R = []
    for index in sweepset_dc:
        if index < n:
            L.append(index)
        else:
            R.append(index - n)

    return L, R, conductance_dc


def evo_cut_directed(G, starting_vertices, target_phi, T=None, debug=False):
    """
    An implementation of the EvoCutDirected algorithm. The graph is assumed to be unweighted.

    :param G: the semi-double cover of the directed graph on which to operate
    :param starting_vertices: a list of starting vertices
    :param target_phi: the flow ratio of the target sets
    :param T: Optionally specify the internal parameter to use instead of the one computed from phi
    :return: the returned clusters L and R, as vertex indices on the original graph, along with the flow ratio phi
    """
    # Compute the value of T to use
    if T is None:
        T = math.floor(1 / (100 * (target_phi ** (2/3))))

    if debug:
        print(f"T: {T}")

    # Get the adjacency matrix of the graph
    A = G.adjacency_matrix
    # Compute the evolving set process for T steps.
    # S will be the current evolving set.
    # X will be the position of the random walk particle
    S = set(starting_vertices)
    probabilities = [G.d[v] / G.volume(list(S)) for v in S]
    X = npr.choice(list(S), p=probabilities)

    # Define the probability function from a vertex to a set
    def p(vert, evolv_set):
        """Get the probability of moving from vert to a vertex inside evolv_set"""
        d_vert = G.d[vert]
        w_vert_S = 0
        for v in G.neighbors(vert):
            if v in evolv_set:
                w_vert_S += A[vert, v]
        if vert in evolv_set:
            return 0.5 + (0.5 * w_vert_S / d_vert)
        else:
            return 0.5 * w_vert_S / d_vert

    for t in range(T):
        if debug:
            print(f"Time {t}; S = {S}")
        # Choose the next location for the random walk particle
        # This is GenerateSample step 1(a)
        X_neighbours = G.neighbors(X)
        X_degree = G.d[X]
        probabilities = [A[X, v] / X_degree for v in X_neighbours]
        X = npr.choice(X_neighbours, p=probabilities)
        if debug:
            print(f"X_(t+1) = {X}")

        # Choose the value of Z
        # This is GenerateSample step 1(b)
        Z = random.uniform(0, p(X, S))
        if debug:
            print(f"Z = {Z}")

        # Update the evolving set S
        # this is GenerateSample step 1(c)
        S_new = set()
        checked = set()  # keep track of which vertices have been checked already
        for v in S:
            if debug:
                print(f"Examining vertex {v}")
            # Check whether v is still inside S
            checked.add(v)
            if p(v, S) >= Z:
                if debug:
                    print(f"Adding {v} to S.")
                S_new.add(v)

            # Check each neighbour of S
            # Note that this is not optimally efficient
            for u in G.neighbors(v):
                if u not in checked:
                    checked.add(u)
                    if p(u, S) >= Z:
                        if debug:
                            print(f"Adding neighbour {u} to S.")
                        S_new.add(u)
        S = S_new

    # Return the left and right sets
    n = int(G.adjacency_matrix.shape[0] / 2)
    if debug:
        print(f"n = {n}")
    L = []
    R = []
    L_other = []
    R_other = []
    for v in S:
        if debug:
            print(f"Processing {v}")
        if v < n:
            if (v + n) not in S:
                L.append(v)
                L_other.append(v + n)
        else:
            if (v - n) not in S:
                R.append(v)
                R_other.append(v - n)

    # If either cluster is empty, return
    if len(L) == 0 or len(R) == 0:
        return L, R_other, 0

    # Compute the cut imbalance
    w_L_R = G.compute_weight(L, R)
    w_R_L = G.compute_weight(R_other, L_other)
    CI = (1/2) * abs((w_L_R - w_R_L)/(w_L_R + w_R_L))
    '''# Counting the numarator of bipartiteness 
    for v in L:
        for u in G.neighbors(v):
            if u in R_other:
                edge_from_L_to_R += 1''' 

    return L, R_other, CI


def bipart_cheeger_cut(G):
    """Find the almost-bipartite set given by a sweep-set operation on the top eigenvector of the normalised graph
    laplacian matrix. See [Trevisan 2012] for details.

    :return: A tuple containing:
        - L - the vertices in the left set
        - R - the vertices in the right set
        - bipart - the bipartiteness of the resulting set.
    """
    # Get the number of vertices in the graph
    n = G.adjacency_matrix.shape[0]

    # Find the top eigenvector of the graph laplacian matrix.
    top_eigvec, _ = eig_nL(G, find_top_eigs=True)

    # Perform the sweep cut and return
    return sweep_cut_dc_from_signed_vec(G, range(n), top_eigvec, normalise_by_degree=True)









def new_evo_cut_directed(G, starting_vertices, target_phi, T=None, debug=False):
    """
    An implementation of the EvoCutDirected algorithm. The graph is assumed to be unweighted.

    :param G: the semi-double cover of the directed graph on which to operate
    :param starting_vertices: a list of starting vertices
    :param target_phi: the flow ratio of the target sets
    :param T: Optionally specify the internal parameter to use instead of the one computed from phi
    :return: the returned clusters L and R, as vertex indices on the original graph, along with the flow ratio phi
    """
    # Compute the value of T to use
    if T is None:
        T = math.floor(1 / (100 * (target_phi ** (2/3))))

    if debug:
        print(f"T: {T}")

    # Get the adjacency matrix of the graph
    A = G.adjacency_matrix
    Pr_Matrix=np.zeros(A.shape)
    # Compute the evolving set process for T steps.
    # S will be the current evolving set.
    # X will be the position of the random walk particle
    S = set(starting_vertices)
    probabilities = [G.d[v] / G.volume(list(S)) for v in S]
    X = npr.choice(list(S), p=probabilities)

    # Define the probability function from a vertex to a set
    def p(vert, evolv_set):
        """Get the probability of moving from vert to a vertex inside evolv_set"""
        d_vert = G.d[vert]
        w_vert_S = 0
        for v in G.neighbors(vert):
            if v in evolv_set:
                w_vert_S += A[vert, v]
        if vert in evolv_set:
            return 0.5 + (0.5 * w_vert_S / d_vert)
        else:
            return 0.5 * w_vert_S / d_vert 
    
    # Define the probability function for sparsification
    def pr_for_sparsification(v):
        pr=[]
        for u in G.neighbors(v):
            if Pr_Matrix[u,v]==0:
                temp_pr= (1/G.d[u] + 1/G.d[v] - 1/(G.d[u]*G.d[v]))
                Pr_Matrix[u,v]=temp_pr
                Pr_Matrix[v,u]=temp_pr
                pr.append(temp_pr)
            else:
                pr.append(Pr_Matrix[u,v])
        
        return npr.choice(list(pr))


    for t in range(T):
        if debug:
            print(f"Time {t}; S = {S}")
        # Choose the next location for the random walk particle
        # This is GenerateSample step 1(a)
        X_neighbours = G.neighbors(X)
        X_degree = G.d[X]
        probabilities = [A[X, v] / X_degree for v in X_neighbours]
        X = npr.choice(X_neighbours, p=probabilities)
        if debug:
            print(f"X_(t+1) = {X}")

        # Choose the value of Z
        # This is GenerateSample step 1(b)
        Z = random.uniform(0, p(X, S))
        if debug:
            print(f"Z = {Z}")

        # Update the evolving set S
        # this is GenerateSample step 1(c)
        S_new = set()
        checked = set()  # keep track of which vertices have been checked already
        for v in S:
            if debug:
                print(f"Examining vertex {v}")
            # Check whether v is still inside S
            checked.add(v)
            if p(v, S) >= Z:
                if debug:
                    print(f"Adding {v} to S.")
                S_new.add(v)

            # Check each neighbour of S
            # Note that this is not optimally efficient
            '''for u in G.neighbors(v):
                if u not in checked:
                    checked.add(u)
                    if p(u, S) >= Z:
                        if debug:
                            print(f"Adding neighbour {u} to S.")
                        S_new.add(u)'''
            V=pr_for_sparsification(v)
            for u in G.neighbors(v):
                if u not in checked:
                    checked.add(u)
                    if  Pr_Matrix[u,v]>= V and p(u, S) >= Z:
                        S_new.add(u)
        S = S_new

    # Return the left and right sets
    n = int(G.adjacency_matrix.shape[0] / 2)
    if debug:
        print(f"n = {n}")
    L = []
    R = []
    L_other = []
    R_other = []
    for v in S:
        if debug:
            print(f"Processing {v}")
        if v < n:
            if (v + n) not in S:
                L.append(v)
                L_other.append(v + n)
        else:
            if (v - n) not in S:
                R.append(v)
                R_other.append(v - n)

    # If either cluster is empty, return
    if len(L) == 0 or len(R) == 0:
        return L, R_other, 0

    # Compute the cut imbalance
    w_L_R = G.compute_weight(L, R)
    w_R_L = G.compute_weight(R_other, L_other)
    CI = (1/2) * abs((w_L_R - w_R_L)/(w_L_R + w_R_L))
    '''
    edge_from_L_to_R=0
    
    # Counting the numarator of bipartiteness 
    for v in L:
        for u in G.neighbors(v):
            if u in R_other:
                edge_from_L_to_R += 1 
'''
    return L, R_other, CI