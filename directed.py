import networkx as nx
import localgraphclustering as lgc
import numpy as np
import time
from plots.plot_utils import *
from plots.plot_style import *

set_plot_style()

no_nodes = [500,600,700,800,900,1000]
etas = [0.7, 0.8, 0.9]
save_path = "Directed Plots"

for eta in etas:
    print(f"\n=== Running experiments for eta = {eta} ===")
    
    pt_time_all, our_time_all = [], []
    pt_flr_min, our_flr_min = [], []

    for n in no_nodes:
        print(f"\nGenerating graph with {n} nodes per partition...")
        prob_matrix = [[9 / n, eta], [1 - eta, 9 / n]]
        G = nx.stochastic_block_model([n, n], prob_matrix, directed=True)

        edge_list_file = "directed_sbm_edgelist.edgelist"
        nx.write_edgelist(G, edge_list_file, data=False)
        directed_graph = lgc.GraphLocal(edge_list_file, 'edgelist', separator=' ', semi_double_cover=True)

        pt_times, our_times = [], []
        pt_flrs, our_flrs = [], []

        for i in range(5):
            

            start = time.time()
            L, R, _ = lgc.find_bipartite_clusters.evo_cut_directed(directed_graph, [1], 0.1, T=2)
            runtime_pt = time.time() - start
            flr_pt = directed_graph.compute_conductance(L + [v + 2 * n for v in R])
            pt_times.append(runtime_pt)
            pt_flrs.append(flr_pt)
            

            start = time.time()
            L, R, _ = lgc.find_bipartite_clusters.new_evo_cut_directed(directed_graph, [1], 0.1, T=2)
            runtime_our = time.time() - start
            flr_our = directed_graph.compute_conductance(L + [v + 2 * n for v in R])
            our_times.append(runtime_our)
            our_flrs.append(flr_our)
            print(f"  Run {i + 1}/5: complete")

        pt_time_all.append(pt_times)
        our_time_all.append(our_times)
        pt_flr_min.append(np.mean(pt_flrs))
        our_flr_min.append(np.min(our_flrs))
        print(f"    ECD    — Time: {np.mean(runtime_pt):.4f}s, Flow Ratio: {np.mean(flr_pt):.4f}")
        print(f"    ECD+Our — Time: {np.mean(runtime_our):.4f}s, Flow Ratio: {np.min(flr_our):.4f}")
    pt_time_all = np.array(pt_time_all)
    our_time_all = np.array(our_time_all)

    pt_time_mean = pt_time_all.mean(axis=1)
    pt_time_std = pt_time_all.std(axis=1)
    our_time_mean = our_time_all.mean(axis=1)
    our_time_std = our_time_all.std(axis=1)

    plot_runtime_comparison(no_nodes, pt_time_mean, pt_time_std, our_time_mean, our_time_std, eta, save_path)
    plot_flow_ratio_comparison(no_nodes, pt_flr_min, our_flr_min, eta, save_path)
