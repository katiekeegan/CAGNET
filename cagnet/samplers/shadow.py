import math
import torch
import torch.distributed as dist
import torch_sparse
import numpy as np
from collections import defaultdict
from cagnet.samplers.utils import *
from sparse_coo_tensor_cpp import *

timing = True

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

def shadow_sampler(adj_matrix, batches, batch_size, frontier_sizes, mb_count_total, n_layers, n_darts_list, \
                        replication, rank, size, row_groups, col_groups,
                        timing_arg, replicate_graph):

    global timing

    timing = timing_arg

    total_start_timer = torch.cuda.Event(enable_timing=True)
    total_stop_timer = torch.cuda.Event(enable_timing=True)

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    timing_dict = defaultdict(list)

    rank_c = rank // replication
    rank_col = rank % replication
    mb_count = batches.size(0)
    node_count = adj_matrix.size(0)
    node_count_total = adj_matrix.size(1)
    gpu = torch.device(f"cuda:{torch.cuda.current_device()}")

    batches_expand_rows = torch.arange(mb_count * batch_size, dtype=torch.int32, device=gpu)
    batches_expand_idxs = torch.stack((batches_expand_rows, batches._indices()[1, :]))
    batches_expand = sparse_coo_tensor_gpu(batches_expand_idxs, batches._values(), 
                                            torch.Size([mb_count * batch_size, node_count_total]))

    print(f"adj_matrix: {adj_matrix}", flush=True)
    print(f"batches_expand: {batches_expand}", flush=True)
    print(f"n_layers: {n_layers}", flush=True)

    current_frontier = batches_expand

    frontiers = torch.cuda.LongTensor(mb_count, node_count_total).fill_(0) # TODO: make sparse as mb_count grows
    for i in range(n_layers):
        print(f"layer {i}", flush=True)

        p = gen_prob_dist(current_frontier, adj_matrix, mb_count, node_count_total,
                                replication, rank, size, row_groups, col_groups,
                                None, timing_dict, "sage", timing_arg, replicate_graph)

        print(f"p: {p}", flush=True)

        frontier_size = frontier_sizes[i]
        n_darts = n_darts_list[i]
        next_frontier = sample(p, frontier_size, mb_count, node_count_total, n_darts,
                                    replication, rank, size, row_groups, col_groups,
                                    timing_dict, "sage")
        
        print(f"next_frontier: {next_frontier}", flush=True)
        print(f"next_frontier._indices.dtype: {next_frontier._indices().dtype}", flush=True)

        # collapse next_frontier to mb_count x node_count_total sparse matrix
        # TODO: Change this when collapsed frontier is sparse
        next_frontier_mask = next_frontier._values().nonzero().squeeze()
        collapsed_frontier_rows = next_frontier._indices()[0, next_frontier_mask]
        collapsed_frontier_rows = collapsed_frontier_rows.div(batch_size, rounding_mode="floor")
        collapsed_frontier_cols = next_frontier._indices()[1, next_frontier_mask]
        collapsed_frontier_idxs = torch.stack((collapsed_frontier_rows, collapsed_frontier_cols))
        collapsed_frontier_vals = torch.cuda.LongTensor(collapsed_frontier_idxs.size(1)).fill_(1)

        collapsed_frontier = torch.sparse_coo_tensor(collapsed_frontier_idxs, 
                                                        collapsed_frontier_vals,
                                                        torch.Size([mb_count, node_count_total]))
        collapsed_frontier_dense = collapsed_frontier.to_dense()
    
        frontiers = frontiers + collapsed_frontier_dense

    frontiers = frontiers.view(-1).squeeze()

    # Assumes nbulkmb=1 for now TODO: change when nbulkmb>1
    rowselect_mask = torch.cuda.BoolTensor(adj_matrix._nnz()).fill_(False)
    sampled_frontiers = frontiers.nonzero().squeeze()
    print(f"sampled_frontiers.size: {sampled_frontiers.size()}", flush=True)
    rowselect_csr_gpu(sampled_frontiers, adj_matrix.crow_indices(), rowselect_mask, 
                        sampled_frontiers.size(0), adj_matrix._nnz())

    row_lengths = adj_matrix.crow_indices()[1:] - adj_matrix.crow_indices()[:-1]
    rowselect_adj_crows = torch.cuda.IntTensor(sampled_frontiers.size(0) + 1).fill_(0)
    print(f"row_lengths[sampled_frontiers]: {row_lengths[sampled_frontiers]}", flush=True)
    print(f"row_lengths[sampled_frontiers].sum: {row_lengths[sampled_frontiers].sum()}", flush=True)
    rowselect_adj_crows[1:] = torch.cumsum(row_lengths[sampled_frontiers], dtype=torch.int32, dim=0)
    rowselect_adj_crows[0] = 0

    rowselect_adj_cols = adj_matrix.col_indices()[rowselect_mask]
    rowselect_adj_vals = adj_matrix.values()[rowselect_mask]

    sampled_frontier_size = sampled_frontiers.size(0)
    print(f"sampled_frontier_size: {sampled_frontier_size}", flush=True)
    row_select_adj = torch.sparse_csr_tensor(rowselect_adj_crows, rowselect_adj_cols, rowselect_adj_vals,
                                                torch.Size([sampled_frontier_size, node_count_total]))

    print(f"row_select_adj: {row_select_adj}", flush=True)
    row_select_adj = row_select_adj.t().to_sparse_csr()

    rowselect_mask = torch.cuda.BoolTensor(row_select_adj._nnz()).fill_(False)
    rowselect_csr_gpu(sampled_frontiers, row_select_adj.crow_indices().long(), rowselect_mask, 
                        sampled_frontiers.size(0), row_select_adj._nnz())

    row_lengths = row_select_adj.crow_indices()[1:] - row_select_adj.crow_indices()[:-1]
    rowselect_adj_crows = torch.cuda.IntTensor(sampled_frontiers.size(0) + 1)
    rowselect_adj_crows[1:] = torch.cumsum(row_lengths[sampled_frontiers], dtype=torch.int32, dim=0)
    rowselect_adj_crows[0] = 0

    rowselect_adj_cols = row_select_adj.col_indices()[rowselect_mask]
    rowselect_adj_vals = row_select_adj.values()[rowselect_mask]

    sampled_frontier_size = sampled_frontiers.size(0)
    print(f"sampled_frontier_size: {sampled_frontier_size}", flush=True)
    sampled_adj = torch.sparse_csr_tensor(rowselect_adj_crows, rowselect_adj_cols, rowselect_adj_vals,
                                                torch.Size([sampled_frontier_size, sampled_frontier_size]))

    sampled_adj = sampled_adj.t().to_sparse_csr()

    print(f"sampled_adj: {sampled_adj}", flush=True)

    if timing:
        for k, v in sorted(timing_dict.items()):
            if (k.startswith("spgemm") and k != "spgemm-misc") or k == "probability-spgemm" or k == "row-select-spgemm" or k == "col-select-spgemm" or k == "sampling-iters" or k == "frontier-row-col-select" or k == "adj-row-col-select" or k.startswith("sample") or k == "compute-p" or k == "sage-startiter" or k == "sage-csr2coo" or k == "sage-preamble" or k == "sage-samplingiter":
                v_tens = torch.cuda.FloatTensor(1).fill_(sum(v))
                v_tens_recv = []
                for i in range(size):
                    v_tens_recv.append(torch.cuda.FloatTensor(1).fill_(0))
                dist.all_gather(v_tens_recv, v_tens)

                if rank == 0:
                    min_time = min(v_tens_recv).item()
                    max_time = max(v_tens_recv).item()
                    avg_time = sum(v_tens_recv).item() / size
                    med_time = sorted(v_tens_recv)[size // 2].item()

                    print(f"{k} min: {min_time} max: {max_time} avg: {avg_time} med: {med_time}")
            dist.barrier()
        for k, v in timing_dict.items():
            if len(v) > 0:
                avg_time = sum(v) / len(v)
            else:
                avg_time = -1.0
            print(f"{k} total_time: {sum(v)} avg_time {avg_time} len: {len(v)}")
    return frontiers, sampled_adjs
