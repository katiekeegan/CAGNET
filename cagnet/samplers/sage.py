import math
import torch
import torch.distributed as dist
import torch_sparse
from collections import defaultdict
from cagnet.samplers.utils import *

from sparse_coo_tensor_cpp import downsample_gpu, compute_darts_gpu, throw_darts_gpu, \
                                    compute_darts_select_gpu, throw_darts_select_gpu, \
                                    compute_darts1d_gpu, throw_darts1d_gpu, normalize_gpu, \
                                    shift_rowselect_gpu, shift_colselect_gpu, \
                                    scatterd_add_gpu, scatteri_add_gpu, rowselect_coo_gpu, \
                                    sparse_coo_tensor_gpu


timing = True
baseline_compare = True

def start_time(timer):
    if timing:
        timer.record()

def stop_time(start_timer, stop_timer, barrier=False):
    if timing:
        stop_timer.record()
        torch.cuda.synchronize()
        time_taken = start_timer.elapsed_time(stop_timer)
        if barrier:
            dist.barrier()
        return time_taken
    else:
        return 0.0

def sage_sampler(adj_matrix, batches, batch_size, frontier_size, mb_count_total, n_layers, n_darts, \
                        replication, sa_masks, rank, size, row_groups, col_groups,
                        timing_arg, baseline_arg):

    global timing
    global baseline_compare

    timing = timing_arg
    baseline_compare = baseline_arg

    total_start_timer = torch.cuda.Event(enable_timing=True)
    total_stop_timer = torch.cuda.Event(enable_timing=True)

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    timing_dict = defaultdict(list)

    node_count = adj_matrix.size(0)
    node_count_total = adj_matrix.size(1)
    mb_count = batches.size(0)

    rank_c = rank // replication
    rank_col = rank % replication

    n_darts_col = n_darts // replication
    if rank_col == replication - 1:
        n_darts_col = n_darts - (replication - 1) * n_darts_col
    n_darts_col = n_darts

    # adj_matrices = [[None] * n_layers for x in range(mb_count)] 
    # adj_matrices[i][j] --  mb i layer j
    adj_matrices = [None] * n_layers # adj_matrices[i] --  bulk mb mtx for layer j

    gpu = torch.device(f"cuda:{torch.cuda.current_device()}")

    batches_expand_rows = torch.arange(mb_count * batch_size, device=gpu)
    batches_expand_idxs = torch.stack((batches_expand_rows, batches._indices()[1, :]))
    batches_expand = torch.sparse_coo_tensor(
                            batches_expand_idxs,
                            batches._values(), 
                            size=(mb_count * batch_size, node_count_total))

    batches_expand = batches_expand.to_sparse_csr()
    adj_matrix = adj_matrix.to_sparse_csr()
    current_frontier = batches_expand
    frontiers = [None] * (n_layers + 1)

    if baseline_compare:
        total_start_timer.record()
    for i in range(n_layers):
        if i == 0:
            nnz = batch_size
        else:
            current_frontier_nnzmask = current_frontier._values().nonzero().squeeze()
            current_frontier_nnzinds = current_frontier._indices()[:, current_frontier_nnzmask]
            current_frontier_nnzvals = current_frontier._values()[current_frontier_nnzmask].double()
            current_frontier = torch.sparse_coo_tensor(current_frontier_nnzinds, current_frontier_nnzvals,
                                                            size=current_frontier.size())
            nnz = current_frontier._nnz()
            current_frontier = current_frontier.to_sparse_csr()

        print(f"i: {i} nnz: {nnz}", flush=True)
        # Expand batches matrix
        if baseline_compare:
            total_start_timer.record()
        p = gen_prob_dist(current_frontier, adj_matrix, mb_count, node_count_total,
                                replication, rank, size, row_groups, col_groups,
                                sa_masks, timing_dict, "sage",
                                timing_arg)
        adj_matrix = adj_matrix.to_sparse_coo()
        if p.layout == torch.sparse_csr:
            p = p.to_sparse_coo()
        # batches = batches.to_sparse_coo()

        next_frontier = sample(p, frontier_size, mb_count, node_count_total, n_darts,
                                    replication, rank, size, row_groups, col_groups,
                                    timing_dict, "sage")

        start_time(start_timer)
        # add explicit 0's to next_frontier
        next_frontier_nnz = next_frontier._values().nonzero().squeeze()
        frontier_nnz_sizes = torch.histc(next_frontier._indices()[0,next_frontier_nnz], bins=p.size(0))

        frontier_nnz_sizes = torch.clamp(frontier_nnz_sizes, max=frontier_size)
        next_frontier_rows = torch.repeat_interleave(
                                torch.arange(nnz * mb_count, device=gpu),
                                # torch.arange(batch_size * mb_count, device=gpu),
                                frontier_size)
        nextf_cols_idxs = torch.arange(next_frontier_nnz.size(0), device=gpu)
        frontier_remainder = frontier_size - frontier_nnz_sizes
        ps_f_remain = torch.cumsum(frontier_remainder, dim=0).roll(1)
        ps_f_remain[0] = 0
        nextf_cols_idxs += torch.repeat_interleave(ps_f_remain, frontier_nnz_sizes)
        next_frontier_cols = torch.cuda.LongTensor(next_frontier_rows.size(0)).fill_(0)
        next_frontier_cols.scatter_(0, nextf_cols_idxs, next_frontier._indices()[1,next_frontier_nnz])

        next_frontier_idxs = torch.stack((next_frontier_rows, next_frontier_cols))
        next_frontier_values = torch.cuda.LongTensor(next_frontier_rows.size(0)).fill_(0)
        next_frontier_values[nextf_cols_idxs] = 1

        # Construct sampled adj matrix
        next_frontier = torch.sparse_coo_tensor(next_frontier_idxs, 
                                            next_frontier_values,
                                            size=(nnz * mb_count, node_count_total))
                                            # size=(batch_size * mb_count, node_count_total))

        # next_frontier_select = next_frontier._indices()[1,:].view(mb_count * batch_size, frontier_size)
        next_frontier_select = next_frontier._indices()[1,:].view(mb_count * nnz, frontier_size)
        # batches_select = torch.masked_select(batches._indices()[1,:], \
        #                                         batches._values().bool()).view(mb_count * batch_size, 1)
        current_frontier_select = torch.masked_select(current_frontier.col_indices(), \
                                                current_frontier.values().bool()).view(mb_count * batch_size, 1)
        next_frontier_select = torch.cat((next_frontier_select, current_frontier_select), dim=1)
        batch_vals = torch.cuda.LongTensor(current_frontier_select.size()).fill_(1)
        # next_frontier_select_vals = next_frontier._values().view(mb_count * batch_size, frontier_size)
        next_frontier_select_vals = next_frontier._values().view(mb_count * nnz, frontier_size)
        next_frontier_select_vals = torch.cat((next_frontier_select_vals, batch_vals), dim=1).view(-1)

        # batch_rows = torch.arange(mb_count * batch_size).cuda().view(mb_count * batch_size, 1)
        batch_rows = torch.arange(mb_count * nnz).cuda().view(mb_count * nnz, 1)
        # next_frontier_select_rows = next_frontier._indices()[0,:].view(mb_count * batch_size, frontier_size)
        next_frontier_select_rows = next_frontier._indices()[0,:].view(mb_count * nnz, frontier_size)
        next_frontier_select_rows = torch.cat((next_frontier_select_rows, batch_rows), dim=1).view(-1)

        nnz_mask = next_frontier_select_vals.nonzero().squeeze()
        adj_sample_rows = next_frontier_select_rows[nnz_mask]
        adj_sample_cols = torch.arange(next_frontier_select.numel()).cuda()
        # adj_sample_cols = adj_sample_cols.remainder(next_frontier_select.size(1) * batch_size)
        adj_sample_cols = adj_sample_cols.remainder(next_frontier_select.size(1) * nnz)
        adj_sample_cols = adj_sample_cols[nnz_mask]
        adj_matrices_indices = torch.stack((adj_sample_rows, adj_sample_cols))
        adj_matrices_values = torch.cuda.DoubleTensor(adj_sample_rows.size(0)).fill_(1)

        # adj_matrices = [torch.sparse_coo_tensor(adj_matrices_indices, adj_matrices_values, 
        #                         size=torch.Size([mb_count * nnz, next_frontier_select.size(1) * batch_size]))]
        adj_matrix_sample = torch.sparse_coo_tensor(adj_matrices_indices, adj_matrices_values, 
                                # size=torch.Size([mb_count * nnz, next_frontier_select.size(1) * batch_size]))
                                size=torch.Size([mb_count * nnz, next_frontier_select.size(1) * nnz]))
        adj_matrices[i] = adj_matrix_sample
        frontiers[i] = current_frontier_select.clone()
        current_frontier = next_frontier
        timing_dict["row-col-select"].append(stop_time(start_timer, stop_timer, barrier=True))
    frontiers[n_layers] = next_frontier_select.clone()

    # print(f"total_time: {stop_time(total_start_timer, total_stop_timer)}", flush=True)
    if baseline_compare:
        total_stop_timer.record()
        torch.cuda.synchronize()
        total_time = total_start_timer.elapsed_time(total_stop_timer)
    print(f"total_time: {total_time}", flush=True)
    if timing:
        for k, v in sorted(timing_dict.items()):
            if (k.startswith("spgemm") and k != "spgemm-misc") or k == "probability-spgemm" or k == "row-select-spgemm" or k == "col-select-spgemm":
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
    # return current_frontier_select, next_frontier_select, adj_matrices
    return frontiers, adj_matrices
