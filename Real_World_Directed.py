import networkx as nx
import matplotlib.pyplot as plt
import localgraphclustering as lgc
from scipy.io import loadmat
import numpy as np
import time



n=3141

starting_vertex=3010

target_phi =0.1

itr= 30

edge_list_file = "Realworld_graph_bulider/census_data_graph_construction/US_Migration_directed_graph.edgelist"
directed_sbm_semi_DC = lgc.GraphLocal(edge_list_file, 'edgelist', separator=' ', semi_double_cover=True)
best_L = None
best_R = None
best_fr = None
start_time=time.time()
for i in range(itr):
    L, R, _= lgc.find_bipartite_clusters.evo_cut_directed(directed_sbm_semi_DC, [starting_vertex], target_phi, T=2)
    #print(f"L: {L}")
    #print(f"R: {R}")
    flow_ratio= directed_sbm_semi_DC.compute_conductance(L + [(v+n)  for v in R])
    
    if best_fr is None or flow_ratio < best_fr:
        best_fr = flow_ratio
        best_L = L
        best_R = R
    #print(f"Peter's Cluster one: {(L)}")
    #print(" ")
    #print(f"Peter's Cluster two: {(R)}")
#print(f" Iteration: {ind + 1} ")
print("--------------------------------------------------------------------------------------------------")
print(f"ECD algo with {n} nodes")
print(f"Flow Ratio: {best_fr}")
#pt_flr.append(best_fr)
#print(f"L: {best_L}")
#print(f"R: {best_R}")
end_time=time.time()
print(f"Time taken : {end_time - start_time : 4f} secs")
#pt_time.append(end_time - start_time )

print(" ")
best_L = None
best_R = None
best_fr = None
start_time=time.time()
print(f"ECD+Our algo with {n} nodes")

for i in range(itr):
    L, R, _= lgc.find_bipartite_clusters.new_evo_cut_directed(directed_sbm_semi_DC, [starting_vertex], target_phi, T=2)
    flow_ratio= directed_sbm_semi_DC.compute_conductance(L + [(v+n)  for v in R])
    
    if best_fr is None or flow_ratio < best_fr:
        best_fr = flow_ratio
        best_L = L
        best_R = R
    #print(f"L: {(L)}")
    #print(" ")
    #print(f"R: {(R)}")
    #print(" ")
print(f"Flow Ratio: {best_fr}")
#our_flr.append(best_fr)
#print(f"L: {best_L}")
#print(f"R: {best_R}")
end_time=time.time()
print(f"Time taken : {end_time - start_time : 4f} secs")
#our_time.append(end_time - start_time )

print("---------------------------------------------------------------------------------------------------------------")
print(" ")
