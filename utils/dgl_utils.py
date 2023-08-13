import numpy as np
import scipy.sparse as ssp
import random

"""All functions in this file are from  dgl.contrib.data.knowledge_graph"""


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs.     max_nodes_per_hop默认是200个
    Modified from dgl.contrib.data.knowledge_graph to accomodate node sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    i = 0
    while current_lvl:

        max_nodes_per_hop_ = max_nodes_per_hop[i]
        i+=1

        for v in current_lvl:
            visited.add(v) # set(0,1,2,3,4,6,7,8,9)

        next_lvl = _get_neighbors(adj, current_lvl) 
        next_lvl -= visited  # set difference 

        next_lvl = list(next_lvl)
        count = np.full(10,len(next_lvl)//10,dtype=int)
        count[:len(next_lvl) % 10] += 1

        start = 0
        stop = 0
        next_lvl_set = set()
        for j in range(10): 
            stop += count[j] 
            next_lvl_ = set(next_lvl[start:stop])
            start += count[j]
            if max_nodes_per_hop and max_nodes_per_hop_ < len(next_lvl_):
                next_lvl_ = set(random.sample(next_lvl_, max_nodes_per_hop_))
            next_lvl_set = set.union(next_lvl_set,next_lvl_)
        yield next_lvl_set

        current_lvl = set.union(next_lvl_set) 


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj) 
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices #  ([0,1,2,3,5,6,7,8,9,...],)
    return neighbors


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)
