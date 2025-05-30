import networkx as nx
import numpy as np
import random
import stag.graph
import stag.random

def offline_edge_sampler(G_orig, C, lambda_k_plus_1):
    H_offline = nx.Graph()
    n = G_orig.number_of_nodes()
    log_n = np.log(n)

    for u, v, data in G_orig.edges(data=True):
        w_uv = data.get('weight', 1.0)
        degree_u = G_orig.degree(u, weight='weight')
        degree_v = G_orig.degree(v, weight='weight')

        if degree_u == 0 or degree_v == 0:
            continue

        p_u_v = min(C * (log_n / lambda_k_plus_1) * (w_uv / degree_u), 1.0)
        p_v_u = min(C * (log_n / lambda_k_plus_1) * (w_uv / degree_v), 1.0)
        p_e = p_u_v + p_v_u - (p_u_v * p_v_u)

        if p_e > 0 and np.random.rand() <= p_e:
            adjusted_weight = w_uv / p_e
            H_offline.add_edge(u, v, weight=adjusted_weight)

    return H_offline

class OnlineStaticOracleSampler:
    def __init__(self, nodes_list, static_degree_oracle, C, lambda_k_plus_1):
        self.H_online = nx.Graph()
        self.nodes_list = nodes_list
        self.H_online.add_nodes_from(self.nodes_list)
        self.static_degree_oracle = static_degree_oracle
        self.C = C
        self.lambda_k_plus_1 = lambda_k_plus_1
        self.n = len(self.nodes_list)
        self.log_n = np.log(self.n)

    def process_edge_online(self, u, v, weight):
        degree_u = self.static_degree_oracle.get(u, 0)
        degree_v = self.static_degree_oracle.get(v, 0)

        if degree_u == 0 or degree_v == 0:
            return False

        p_u_v = min(self.C * (self.log_n / self.lambda_k_plus_1) * (weight / degree_u), 1.0)
        p_v_u = min(self.C * (self.log_n / self.lambda_k_plus_1) * (weight / degree_v), 1.0)
        p_e = p_u_v + p_v_u - (p_u_v * p_v_u)

        if p_e > 0 and np.random.rand() <= p_e:
            adjusted_weight = weight / p_e
            if not self.H_online.has_node(u):
                self.H_online.add_node(u)
            if not self.H_online.has_node(v):
                self.H_online.add_node(v)
            self.H_online.add_edge(u, v, weight=adjusted_weight)
            return True

        return False

    def get_sampled_graph(self):
        return self.H_online

if __name__ == "__main__":
    num_nodes = 10000
    num_clusters = 2
    p_in_cluster = 0.01
    p_between_clusters = 0.1
    C_param = 1.0
    lambda_k_plus_1_param = 1

    print(f"Generating SBM graph (G_sbm) with {num_nodes} nodes.")
    stag_sbm = stag.random.sbm(num_nodes, num_clusters, p_in_cluster, p_between_clusters)
    G_sbm = stag_sbm.to_networkx()

    for u, v in G_sbm.edges():
        G_sbm[u][v]['weight'] = 1.0

    print(f"Initial graph G_sbm created with {G_sbm.number_of_nodes()} nodes and {G_sbm.number_of_edges()} edges.")

    print(f"\nRunning OFFLINE sampler with C={C_param}, lambda_k+1={lambda_k_plus_1_param}")
    H_offline = offline_edge_sampler(G_sbm.copy(), C_param, lambda_k_plus_1_param)
    print(f"Offline sampled graph H_offline has {H_offline.number_of_edges()} edges.")

    static_degree_oracle_map = {node: G_sbm.degree(node, weight='weight') for node in G_sbm.nodes()}
    original_edges_stream = list(G_sbm.edges(data=True))
    random.shuffle(original_edges_stream)

    print(f"\nRunning ONLINE sampler with C={C_param}, lambda_k+1={lambda_k_plus_1_param}")
    current_nodes_list = list(G_sbm.nodes())

    online_sampler = OnlineStaticOracleSampler(
        nodes_list=current_nodes_list,
        static_degree_oracle=static_degree_oracle_map,
        C=C_param,
        lambda_k_plus_1=lambda_k_plus_1_param
    )

    for u, v, data in original_edges_stream:
        weight = data.get('weight', 1.0)
        online_sampler.process_edge_online(u, v, weight)

    H_online = online_sampler.get_sampled_graph()
    print(f"Online sampled graph H_online has {H_online.number_of_edges()} edges.")

    print("\n--- Comparison ---")
    print(f"Number of edges in original G_sbm: {G_sbm.number_of_edges()}")
    print(f"Number of edges in H_offline:      {H_offline.number_of_edges()}")
    print(f"Number of edges in H_online:       {H_online.number_of_edges()}")
