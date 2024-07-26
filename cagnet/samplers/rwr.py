import math
import torch
import torch.distributed as dist
import torch_sparse
import numpy as np
from collections import defaultdict
from cagnet.samplers.utils import *
import random

from sparse_coo_tensor_cpp import downsample_gpu, compute_darts_gpu, throw_darts_gpu, \
                                    compute_darts_select_gpu, throw_darts_select_gpu, \
                                    compute_darts1d_gpu, throw_darts1d_gpu, normalize_gpu, \
                                    shift_rowselect_gpu, shift_colselect_gpu, \
                                    scatteri_add_gpu, rowselect_coo_gpu, \
                                    sparse_coo_tensor_gpu


timing = True
baseline_compare = True

def start_time(timer, timing_arg=None):
    if timing_arg is not None:
        start_timing = timing_arg
    else:
        start_timing = timing
    if start_timing:
        timer.record()

def stop_time(start_timer, stop_timer, barrier=False, timing_arg=None):
    if timing_arg is not None:
        start_timing = timing_arg
    else:
        start_timing = timing
    if start_timing:
        stop_timer.record()
        torch.cuda.synchronize()
        time_taken = start_timer.elapsed_time(stop_timer)
        if barrier:
            dist.barrier()
        return time_taken
    else:
        return 0.0
    
def random_walk_with_restart(adj_matrix, seed_nodes, alpha=0.85, max_iter=100, tol=1e-3):
    """
    Perform Random Walk with Restart (RWR) on the adjacency matrix starting from the seed nodes.

    Parameters:
    adj_matrix (torch.sparse_csr_tensor): The adjacency matrix of the graph.
    seed_nodes (list of int): The seed nodes for the RWR.
    alpha (float): The restart probability.
    max_iter (int): The maximum number of iterations.
    tol (float): The tolerance for convergence.

    Returns:
    torch.tensor: The steady-state distribution vector.
    """
    n = adj_matrix.size(0)
    r = torch.zeros(n, device=adj_matrix.device)
    r[seed_nodes] = 1.0 / len(seed_nodes)  # initialize seed nodes

    p = r.clone()
    for i in range(max_iter):
        p_next = alpha * torch.sparse.mm(adj_matrix, p.unsqueeze(1)).squeeze() + (1 - alpha) * r
        if torch.norm(p_next - p, p=1) < tol:
            break
        p = p_next
        # print(i)
    
    return p

def get_subgraph(adj_matrix, nodes):
    """
    Extract the subgraph containing the given nodes.

    Parameters:
    adj_matrix (torch.sparse_csr_tensor): The adjacency matrix of the graph.
    nodes (torch.tensor): The nodes in the subgraph.

    Returns:
    torch.sparse_csr_tensor: The adjacency matrix of the subgraph.
    """
    mask = torch.zeros(adj_matrix.size(0), dtype=torch.bool, device=adj_matrix.device)
    mask[nodes] = True
    sub_indices = mask.nonzero(as_tuple=True)[0]

    sub_adj_matrix = adj_matrix.to_dense()[sub_indices][:, sub_indices]
    sub_adj_matrix = sub_adj_matrix.to_sparse_csr()
    
    return sub_adj_matrix

def rwr_subgraphs(adj_matrix, seed_node_count, alpha=0.85, max_iter=100, tol=1e-6, top_k=100):
    """
    Perform RWR from randomly selected seed nodes and return subgraphs.

    Parameters:
    adj_matrix (torch.sparse_csr_tensor): The adjacency matrix of the graph.
    seed_node_count (int): Number of seed nodes to select randomly.
    alpha (float): The restart probability.
    max_iter (int): The maximum number of iterations.
    tol (float): The tolerance for convergence.
    top_k (int): Number of top nodes to consider for subgraph extraction.

    Returns:
    list of torch.sparse_csr_tensor: List of adjacency matrices of the subgraphs.
    """
    n = adj_matrix.size(0)
    seed_nodes = torch.randint(0, n, (seed_node_count,), device=adj_matrix.device)
    
    subgraphs = []
    for seed_node in seed_nodes:
        print(seed_node)
        p = random_walk_with_restart(adj_matrix, [seed_node], alpha, max_iter, tol)
        top_nodes = torch.topk(p, top_k).indices
        subgraph_adj_matrix = get_subgraph(adj_matrix, top_nodes)
        subgraphs.append(subgraph_adj_matrix)
    
    return seed_nodes, subgraphs

def rwr_sampler(adj_matrix, seeds, batch_size, frontier_sizes, mb_count_total, n_layers, n_darts_list, \
                        replication, sa_masks, rank, size, row_groups, col_groups,
                        timing_arg, baseline_arg, replicate_graph):

    global timing
    global baseline_compare

    timing = timing_arg
    baseline_compare = baseline_arg

    total_start_timer = torch.cuda.Event(enable_timing=True)
    total_stop_timer = torch.cuda.Event(enable_timing=True)

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    timing_dict = defaultdict(list)

    start_time(start_timer)
    node_count = adj_matrix.size(0)
    node_count_total = adj_matrix.size(1)
    mb_count = frontier_sizes

    rank_c = rank // replication
    rank_col = rank % replication

    # adj_matrices = [[None] * n_layers for x in range(mb_count)] 
    # adj_matrices[i][j] --  mb i layer j
    adj_matrices = [None] * 1 # adj_matrices[i] --  bulk mb mtx for layer j
    frontiers = [None] * (1 + 1) # frontiers[i] -- bulk mb frontiers for layer j

    gpu = torch.device(f"cuda:{torch.cuda.current_device()}")

    # batches_expand_rows = torch.arange(batch_size, dtype=torch.int32, device=gpu)
    # batches_expand_idxs = torch.stack((batches_expand_rows, seeds._indices()[1, :]))
    # # batches_expand = torch.sparse_coo_tensor(
    # #                         batches_expand_idxs,
    # #                         batches._values(), 
    # #                         size=(mb_count * batch_size, node_count_total))
    # batches_expand = sparse_coo_tensor_gpu(batches_expand_idxs, seeds._values(), 
    #                                         torch.Size([mb_count * batch_size, node_count_total]))

    # if not replicate_graph:
    #     batches_expand = batches_expand.to_sparse_csr()
    timing_dict["sage-preamble"].append(stop_time(start_timer, stop_timer))
    # breakpoint()
    seeds,clusters = rwr_subgraphs(adj_matrix,5)
    # breakpoint()
    # return current_frontier_select, next_frontier_select, adj_matrices
    return [clusters]*(n_layers+1), clusters