import argparse
import copy
import math
import os
import os.path as osp
import time
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.autograd.profiler as profiler
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.loader import NeighborSampler, NeighborLoader
from torch_geometric.utils import add_remaining_self_loops
import torch_sparse

from cagnet.nn.conv import GCNConv
from cagnet.partitionings import Partitioning
from cagnet.samplers import ladies_sampler, sage_sampler
from cagnet.samplers.utils import *
import cagnet.nn.functional as CAGF
import torch.nn.functional as F

from sparse_coo_tensor_cpp import sort_dst_proc_gpu

import socket
import yaml

class InteractionGNN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, nb_node_layer, nb_edge_layer, n_graph_iters, 
                                        aggr, rank, size, partitioning, replication, device, 
                                        group=None, row_groups=None, col_groups=None):
        super(InteractionGNN, self).__init__()
        # self.layers = nn.ModuleList()
        self.nb_node_layer = nb_node_layer
        self.nb_edge_layer = nb_edge_layer
        self.n_graph_iters = n_graph_iters
        self.aggr = aggr
        self.rank = rank
        self.size = size
        self.group = group
        self.row_groups = row_groups
        self.col_groups = col_groups
        self.device = device
        self.partitioning = partitioning
        self.replication = replication
        self.timings = dict()

        torch.manual_seed(0)
        aggr_list = ["sum", "mean", "max", "std"]
        self.aggregation = torch_geometric.nn.aggr.MultiAggregation(aggr_list, mode="cat")

        network_input_size = (1 + 2 * len(aggr_list)) * n_hidden
        self.node_encoder = self.make_mlp(
            in_feats,
            [n_hidden] * nb_node_layer,
            layer_norm=True,
            batch_norm=False,
        )
        
        self.edge_encoder = self.make_mlp(
            2 * n_hidden,
            [n_hidden] * nb_edge_layer,
            layer_norm=True,
            batch_norm=False,
        )
        
        self.edge_network = self.make_mlp(
            3 * n_hidden,
            [n_hidden] * nb_edge_layer,
            layer_norm=True,
            batch_norm=False,
        )
        
        self.node_network = self.make_mlp(
            network_input_size,
            [n_hidden] * nb_node_layer,
            layer_norm=True,
            batch_norm=False,
        )
        
        self.output_edge_classifier = self.make_mlp(
            3 * n_hidden,
            [n_hidden] * nb_edge_layer + [1],
            layer_norm=True,
            batch_norm=False,
            output_activation=None,
        )
        
    def make_mlp(
        self,
        input_size,
        sizes,
        output_activation=torch.nn.Tanh,
        hidden_activation=torch.nn.SiLU,
        layer_norm=False,
        batch_norm=False,
    ):
        """Construct an MLP with specified fully-connected layers."""
        # if hidden_activation is not None:
        # hidden_activation = torch.nn.SiLU
        # output_activation = torch.nn.Tanh
        layers = []
        n_layers = len(sizes)
        sizes = [input_size] + sizes
        # Hidden layers
        for i in range(n_layers - 1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            if layer_norm:
                layers.append(torch.nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(sizes[i + 1], track_running_stats=False, affine=False))
            layers.append(hidden_activation())
        # Final layer
        layers.append(torch.nn.Linear(sizes[-2], sizes[-1]))
        if output_activation is not None:
            if layer_norm:
                layers.append(torch.nn.LayerNorm(sizes[-1], elementwise_affine=False))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(sizes[-1], track_running_stats=False, affine=False))
            layers.append(output_activation())
        return torch.nn.Sequential(*layers)

    def message_step(self, x, start, end, e):
        # Compute new node features
        edge_messages = torch.cat([
            self.aggregation(e, end, dim_size=x.shape[0]),
            self.aggregation(e, start, dim_size=x.shape[0]),
        ], dim=-1)

        node_inputs = torch.cat([x, edge_messages], dim=-1)

        x_out = self.node_network(node_inputs)

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)
        e_out = self.edge_network(edge_inputs)

        return x_out, e_out

    def output_step(self, x, start, end, e):
        classifier_inputs = torch.cat([x[start], x[end], e], dim=1)
        classifier_output = self.output_edge_classifier(classifier_inputs).squeeze(-1)
        return classifier_output

    def forward(self, batch, epoch):
        x = torch.stack([batch["z"]], dim=-1).float()
        start, end = batch.edge_index
        x.requires_grad = True
        x = self.node_encoder(x)
        e = self.edge_encoder(torch.cat([x[start], x[end]], dim=1))
        for _ in range(self.n_graph_iters):
            x, e = self.message_step(x, start, end, e)

        output = self.output_step(x, start, end, e)
        return output

    def loss_function(self, output, batch):
        """
        Applies the loss function to the output of the model and the truth labels.
        To balance the positive and negative contribution, simply take the means of each separately.
        Any further fine tuning to the balance of true target, true background and fake can be handled 
        with the `weighting` config option.
        """

        assert hasattr(batch, "y"), "The batch does not have a truth label"
        assert hasattr(batch, "weights"), "The batch does not have a weighting label"
        
        negative_mask = ((batch.y == 0) & (batch.weights != 0)) | (batch.weights < 0) 
        
        negative_loss = F.binary_cross_entropy_with_logits(
            output[negative_mask], torch.zeros_like(output[negative_mask]), weight=batch.weights[negative_mask].abs()
        )

        positive_mask = (batch.y == 1) & (batch.weights > 0)
        positive_loss = F.binary_cross_entropy_with_logits(
            output[positive_mask], torch.ones_like(output[positive_mask]), weight=batch.weights[positive_mask].abs()
        )

        return positive_loss + negative_loss

    @torch.no_grad()
    def evaluate(self, graph, features, test_idx, labels):
        # subgraph_loader = NeighborSampler(graph, node_idx=None,
        #                                   sizes=[-1], batch_size=2048,
        #                                   shuffle=False, num_workers=6)

        # for i in range(self.n_layers):
        #     xs = []
        #     for batch_size, n_id, adj in subgraph_loader:
        #         edge_index, _, size = adj.to(self.device)
        #         x = features[n_id].to(self.device)
        #         # edge_index, _, size = adj
        #         # x = features[n_id]
        #         # x_target = x[:size[1]]
        #         # x = self.convs[i]((x, x_target), edge_index)
        #         x = self.layers[i](x, edge_index)
        #         if i != self.n_layers - 1:
        #             x = F.relu(x)
        #         # xs.append(x)
        #         xs.append(x[:batch_size])

        #     features = torch.cat(xs, dim=0)

        # return features

        subgraph_loader = NeighborSampler(graph, node_idx=None,
                                          sizes=[-1], batch_size=2048,
        # subgraph_loader = NeighborSampler(graph, node_idx=test_idx,
                                          # sizes=[-1, -1], batch_size=512,
                                          shuffle=False)
        non_eval_timings = copy.deepcopy(self.timings)
        for l, layer in enumerate(self.layers):
            hs = []
            for batch_size, n_id, adj in subgraph_loader:
                # edge_index, _, size = adj.to(self.device)
                edge_index, _, size = adj
                adj_batch = torch.sparse_coo_tensor(edge_index, 
                                                        torch.FloatTensor(edge_index.size(1)).fill_(1.0),
                                                        size)
                adj_batch = adj_batch.t().coalesce()
                h = features[n_id]

                # h = layer(self, adj_batch, h_batch, epoch=-1) # GCNConv
                h = self.layers[l](h, edge_index)
                if l != len(self.layers) - 1:
                    h = CAGF.relu(h, self.partitioning)
                # hs.append(h) # GCNConv
                hs.append(h[:batch_size]) # SAGEConv
            features = torch.cat(hs, dim=0)
        return features

        # print(f"test_idx: {test_idx}", flush=True)
        # non_eval_timings = copy.deepcopy(self.timings)
        # correct_count = 0
        # for batch_size, n_id, adjs in subgraph_loader:
        #     h = features[n_id]
        #     for l, (edge_index, _, size) in enumerate(adjs):
        #         # h = layer(self, adj_batch, h_batch, epoch=-1) # GCNConv
        #         h = self.layers[l](h, edge_index) # SAGEConv
        #         if l != len(self.layers) - 1:
        #             h = F.relu(h)
        #     h = F.log_softmax(h, dim=1)
        #     batch = n_id[:batch_size]
        #     print(f"batch: {batch}", flush=True)
        #     print(f"h[:batch_size]: {h[:batch_size]}", flush=True)
        #     print(f"h[:batch_size].argmax(dim=-1): {h[:batch_size].argmax(dim=-1)}", flush=True)
        #     preds = h[:batch_size].argmax(dim=-1) == labels[batch]
        #     correct_count += preds.sum()
        # return correct_count

        # """ SAGEConv inference """
        # subgraph_loader = NeighborSampler(graph, node_idx=None,
        #                                   sizes=[-1], batch_size=2048,
        #                                   shuffle=False)

        # # Compute representations of nodes layer by layer, using *all*
        # # available edges. This leads to faster computation in contrast to
        # # immediately computing the final representations of each batch.
        # total_edges = 0
        # for i in range(self.n_layers):
        #     xs = []
        #     for batch_size, n_id, adj in subgraph_loader:
        #         edge_index, _, size = adj.to(self.device)
        #         total_edges += edge_index.size(1)
        #         x = features[n_id].to(self.device)
        #         # x_target = x[:size[1]]
        #         # x = self.convs[i]((x, x_target), edge_index)
        #         x = self.layers[i](x, edge_index) # SAGEConv
        #         if i != self.n_layers - 1:
        #             x = F.relu(x)
        #         x = x[:batch_size]
        #         xs.append(x)
        #     features = torch.cat(xs, dim=0)

        # return features

class LADIES(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, aggr, rank, size, partitioning, replication, 
                                        device, group=None, row_groups=None, col_groups=None):
        super(LADIES, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.aggr = aggr
        self.rank = rank
        self.size = size
        self.group = group
        self.row_groups = row_groups
        self.col_groups = col_groups
        self.device = device
        self.partitioning = partitioning
        self.replication = replication
        self.timings = dict()

        self.timings["total"] = []
        self.timings["sample"] = []
        self.timings["extract"] = []
        self.timings["extract-select"] = []
        self.timings["extract-inst"] = []
        self.timings["extract-coalesce"] = []
        self.timings["train"] = []
        self.timings["selectfeats"] = []
        self.timings["fwd"] = []
        self.timings["bwd"] = []
        self.timings["loss"] = []
        self.timings["fakemats"] = []

        self.timings["precomp"] = []
        self.timings["spmm"] = []
        self.timings["gemm_i"] = []
        self.timings["gemm_w"] = []
        self.timings["aggr"] = []

        # # input layer
        # self.layers.append(GCNConv(in_feats, n_hidden, self.partitioning, self.device))
        # # hidden layers
        # for i in range(n_layers - 2):
        #         self.layers.append(GCNConv(n_hidden, n_hidden, self.partitioning, self.device))
        # # output layer
        # self.layers.append(GCNConv(n_hidden, n_classes, self.partitioning, self.device))
        self.layers.append(GCNConv(in_feats, n_classes, self.partitioning, self.device))

    def forward(self, graphs, inputs, epoch):
        h = inputs
        for l, layer in enumerate(self.layers):
            # graphs[l].t_()
            # edge_index = graphs[l]._indices()
            h = layer(self, graphs[l], h, epoch) # GCNConv
            if l != len(self.layers) - 1:
                # h = CAGF.relu(h, self.partitioning)
                h = F.relu(h)

        # h = CAGF.log_softmax(self, h, self.partitioning, dim=1)
        h = F.log_softmax(h, dim=1)
        return h

    @torch.no_grad()
    def evaluate(self, graph, features, test_idx, labels):
        # subgraph_loader = NeighborSampler(graph, node_idx=None,
        #                                   sizes=[-1], batch_size=2048,
        #                                   shuffle=False, num_workers=6)

        # for i in range(self.n_layers):
        #     xs = []
        #     for batch_size, n_id, adj in subgraph_loader:
        #         edge_index, _, size = adj.to(self.device)
        #         x = features[n_id].to(self.device)
        #         # edge_index, _, size = adj
        #         # x = features[n_id]
        #         # x_target = x[:size[1]]
        #         # x = self.convs[i]((x, x_target), edge_index)
        #         x = self.layers[i](x, edge_index)
        #         if i != self.n_layers - 1:
        #             x = F.relu(x)
        #         # xs.append(x)
        #         xs.append(x[:batch_size])

        #     features = torch.cat(xs, dim=0)

        # return features

        subgraph_loader = NeighborSampler(graph, node_idx=None,
                                          sizes=[-1], batch_size=2048,
        # subgraph_loader = NeighborSampler(graph, node_idx=test_idx,
                                          # sizes=[-1, -1], batch_size=512,
                                          shuffle=False)
        non_eval_timings = copy.deepcopy(self.timings)
        for l, layer in enumerate(self.layers):
            hs = []
            for batch_size, n_id, adj in subgraph_loader:
                # edge_index, _, size = adj.to(self.device)
                edge_index, _, size = adj
                adj_batch = torch.sparse_coo_tensor(edge_index, 
                                                        torch.FloatTensor(edge_index.size(1)).fill_(1.0),
                                                        size)
                adj_batch = adj_batch.t().coalesce()
                h = features[n_id]

                # h = layer(self, adj_batch, h_batch, epoch=-1) # GCNConv
                h = self.layers[l](h, edge_index)
                if l != len(self.layers) - 1:
                    h = CAGF.relu(h, self.partitioning)
                # hs.append(h) # GCNConv
                hs.append(h[:batch_size]) # SAGEConv
            features = torch.cat(hs, dim=0)
        return features

def get_proc_groups(rank, size, replication):
    rank_c = rank // replication
     
    row_procs = []
    for i in range(0, size, replication):
        row_procs.append(list(range(i, i + replication)))

    col_procs = []
    for i in range(replication):
        col_procs.append(list(range(i, size, replication)))

    row_groups = []
    for i in range(len(row_procs)):
        row_groups.append(dist.new_group(row_procs[i]))

    col_groups = []
    for i in range(len(col_procs)):
        col_groups.append(dist.new_group(col_procs[i]))

    return row_groups, col_groups

# Normalize all elements according to KW's normalization rule
def scale_elements(adj_matrix, adj_part, node_count, row_vtx, col_vtx, normalization):
    if not normalization:
        return adj_part

    adj_part = adj_part.coalesce()
    deg = torch.histc(adj_matrix[0].float(), bins=node_count)
    deg = deg.pow(-0.5)

    row_len = adj_part.size(0)
    col_len = adj_part.size(1)

    dleft = torch.sparse_coo_tensor([np.arange(0, row_len).tolist(),
                                     np.arange(0, row_len).tolist()],
                                     deg[row_vtx:(row_vtx + row_len)].float(),
                                     size=(row_len, row_len),
                                     requires_grad=False, device=torch.device("cpu"))

    dright = torch.sparse_coo_tensor([np.arange(0, col_len).tolist(),
                                     np.arange(0, col_len).tolist()],
                                     deg[col_vtx:(col_vtx + col_len)].float(),
                                     size=(col_len, col_len),
                                     requires_grad=False, device=torch.device("cpu"))
    # adj_part = torch.sparse.mm(torch.sparse.mm(dleft, adj_part), dright)
    ad_ind, ad_val = torch_sparse.spspmm(adj_part._indices(), adj_part._values(), 
                                            dright._indices(), dright._values(),
                                            adj_part.size(0), adj_part.size(1), dright.size(1))

    adj_part_ind, adj_part_val = torch_sparse.spspmm(dleft._indices(), dleft._values(), 
                                                        ad_ind, ad_val,
                                                        dleft.size(0), dleft.size(1), adj_part.size(1))

    adj_part = torch.sparse_coo_tensor(adj_part_ind, adj_part_val, 
                                                size=(adj_part.size(0), adj_part.size(1)),
                                                requires_grad=False, device=torch.device("cpu"))

    return adj_part

# Split a COO into partitions of size n_per_proc
# Basically torch.split but for Sparse Tensors since pytorch doesn't support that.
def split_coo(adj_matrix, node_count, n_per_proc, dim):
    vtx_indices = list(range(0, node_count, n_per_proc))
    vtx_indices.append(node_count)

    am_partitions = []
    for i in range(len(vtx_indices) - 1):
        am_part = adj_matrix[:,(adj_matrix[dim,:] >= vtx_indices[i]).nonzero().squeeze(1)]
        am_part = am_part[:,(am_part[dim,:] < vtx_indices[i + 1]).nonzero().squeeze(1)]
        am_part[dim] -= vtx_indices[i]
        am_partitions.append(am_part)

    return am_partitions, vtx_indices

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.sparse.sum(mx, 1)
    r_inv = torch.float_power(rowsum, -1).flatten()
    # r_inv._values = r_inv._values()[torch.isinf(r_inv._values())] = 0.
    # r_mat_inv = torch.diag(r_inv._values())
    r_inv_values = torch.cuda.DoubleTensor(r_inv.size(0)).fill_(0)
    r_inv_values[r_inv._indices()[0,:]] = r_inv._values()
    # r_inv_values = r_inv._values()
    r_inv_values[torch.isinf(r_inv_values)] = 0
    r_mat_inv = torch.sparse_coo_tensor([np.arange(0, r_inv.size(0)).tolist(),
                                     np.arange(0, r_inv.size(0)).tolist()],
                                     r_inv_values,
                                     size=(r_inv.size(0), r_inv.size(0)))
    # mx = r_mat_inv.mm(mx.float())
    mx_indices, mx_values = torch_sparse.spspmm(r_mat_inv._indices(), r_mat_inv._values(), 
                                                    mx._indices(), mx._values(),
                                                    r_mat_inv.size(0), r_mat_inv.size(1), mx.size(1),
                                                    coalesced=True)
    mx = torch.sparse_coo_tensor(indices=mx_indices, values=mx_values.double(), size=(r_mat_inv.size(0), mx.size(1)))
    return mx

def one5d_partition(rank, size, inputs, adj_matrix, data, features, classes, replication, \
                            normalize, replicate_graph):
    node_count = inputs.size(0)
    # n_per_proc = math.ceil(float(node_count) / size)
    # n_per_proc = math.ceil(float(node_count) / (size / replication))
    n_per_proc = node_count // (size // replication)

    am_partitions = None
    am_pbyp = None

    # inputs = inputs.to(torch.device("cpu"))
    # adj_matrix = adj_matrix.to(torch.device("cpu"))
    # torch.cuda.synchronize()

    rank_c = rank // replication
    # Compute the adj_matrix and inputs partitions for this process
    # TODO: Maybe I do want grad here. Unsure.
    with torch.no_grad():
        # Column partitions
        if not replicate_graph:
            am_partitions, vtx_indices = split_coo(adj_matrix, node_count, n_per_proc, 1)
        else:
            am_partitions = None
        # # proc_node_count = vtx_indices[rank_c + 1] - vtx_indices[rank_c]
        # # am_pbyp, _ = split_coo(am_partitions[rank_c], node_count, n_per_proc, 0)
        # # print(f"before", flush=True)
        # # for i in range(len(am_pbyp)):
        # #     if i == size // replication - 1:
        # #         last_node_count = vtx_indices[i + 1] - vtx_indices[i]
        # #         am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
        # #                                                 size=(last_node_count, proc_node_count),
        # #                                                 requires_grad=False)

        # #         am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
        # #                                         vtx_indices[rank_c], normalize)
        # #     else:
        # #         am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
        # #                                                 size=(n_per_proc, proc_node_count),
        # #                                                 requires_grad=False)

        # #         am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
        # #                                         vtx_indices[rank_c], normalize)

        if not replicate_graph:
            for i in range(len(am_partitions)):
                proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], 
                                                        torch.ones(am_partitions[i].size(1)).double(), 
                                                        size=(node_count, proc_node_count), 
                                                        requires_grad=False)
                am_partitions[i] = scale_elements(adj_matrix, am_partitions[i], node_count,  0, vtx_indices[i], \
                                                        normalize)
            adj_matrix_loc = am_partitions[rank_c]
        else:
            adj_matrix_loc = None

        # # input_partitions = torch.split(inputs, math.ceil(float(inputs.size(0)) / (size / replication)), dim=0)
        input_partitions = torch.split(inputs, inputs.size(0) // (size // replication), dim=0)
        if len(input_partitions) > (size // replication):
            input_partitions_fused = [None] * (size // replication)
            input_partitions_fused[:-1] = input_partitions[:-2]
            input_partitions_fused[-1] = torch.cat(input_partitions[-2:], dim=0)
            input_partitions = input_partitions_fused

        inputs_loc = input_partitions[rank_c]

    # print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    print(f"rank: {rank} inputs_loc.size: {inputs_loc.size()}", flush=True)
    return inputs_loc, adj_matrix_loc, am_partitions, input_partitions

def one5d_partition_mb(rank, size, batches, replication, mb_count):
    rank_c = rank // replication
    batch_partitions = torch.split(batches, int(mb_count // (size / replication)), dim=0)
    return batch_partitions[rank_c]
    # batch_partitions = torch.split(batches, int(mb_count // size), dim=0)
    # return batch_partitions[rank]

def main(args, batches=None):
    # load and preprocess dataset
    # Initialize distributed environment with SLURM
    if "SLURM_PROCID" in os.environ.keys():
        os.environ["RANK"] = os.environ["SLURM_PROCID"]

    if "SLURM_NTASKS" in os.environ.keys():
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    os.environ["MASTER_ADDR"] = args.hostname 
    os.environ["MASTER_PORT"] = "1234"
    
    print(f"device_count: {torch.cuda.device_count()}")
    print(f"hostname: {socket.gethostname()}", flush=True)
    if not dist.is_initialized():
        dist.init_process_group(backend=args.dist_backend)
    rank = dist.get_rank()
    size = dist.get_world_size()
    print(f"hostname: {socket.gethostname()} rank: {rank} size: {size}", flush=True)
    torch.cuda.set_device(rank % args.gpu)

    device = torch.device(f'cuda:{rank % args.gpu}')

    start_timer = torch.cuda.Event(enable_timing=True)
    stop_timer = torch.cuda.Event(enable_timing=True)

    start_inner_timer = torch.cuda.Event(enable_timing=True)
    stop_inner_timer = torch.cuda.Event(enable_timing=True)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', args.dataset)

    if args.dataset == "cora" or args.dataset == "reddit":
        if args.dataset == "cora":
            dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        elif args.dataset == "reddit":
            dataset = Reddit(path)

        data = dataset[0]
        data = data.to(device)
        # data.x.requires_grad = True
        inputs = data.x.to(device)
        # inputs.requires_grad = True
        data.y = data.y.to(device)
        edge_index = data.edge_index
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        adj_matrix = edge_index
    elif args.dataset == "physics_ex3":
        print(f"Loading coo...", flush=True)
        input_dir = "/pscratch/sd/a/alokt/data_dir/Example_3/metric_learning/"
        with open("gnn_train.yaml") as stream:
            hparams = yaml.safe_load(stream)

        print(f"hparams: {hparams}", flush=True)

        dataset = GraphDataset(input_dir, "trainset", 80, "fit", hparams)

        print(f"dataset: {dataset}", flush=True)
        trainset = []
        for data in dataset:
            data_obj = Data(hit_id=data["hit_id"],
                                x=data["x"], 
                                y=data["y"], 
                                z=data["z"], 
                                edge_index=data["edge_index"], 
                                truth_map=data["truth_map"],
                                weights=data["weights"])

            trainset.append(data_obj)
        trainset = Batch.from_data_list(trainset)
        trainset = trainset.to(device)
        print(f"trainset: {trainset}", flush=True)

        num_features = 1
        num_classes = 2

    row_groups, col_groups = get_proc_groups(rank, size, args.replication)

    model = InteractionGNN(num_features,
                      args.n_hidden,
                      num_classes,
                      args.nb_node_layer,
                      args.nb_edge_layer,
                      args.n_graph_iters,
                      args.aggr,
                      rank,
                      size,
                      Partitioning.NONE,
                      args.replication,
                      device,
                      row_groups=row_groups,
                      col_groups=col_groups)

    model = model.to(device)

    # use optimizer
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=True,
        )

    train_loader = NeighborLoader(trainset,  
                                    num_neighbors=[5,5,5], 
                                    batch_size=10000, 
                                    num_workers=1) 

    for epoch in range(args.n_epochs):
        print(f"Epoch: {epoch}", flush=True)
        if epoch >= 1:
            epoch_start = time.time()

        model.train()
        for batch in train_loader:
            # frontiers_bulk, adj_matrices_bulk = sage_sampler(g_loc, batches_loc, 
            #                                                     args.batch_size,
            #                                                     args.samp_num, 
            #                                                     args.n_bulkmb,
            #                                                     args.n_layers, 
            #                                                     args.n_darts,
            #                                                     rep_pass, 
            #                                                     nnz_row_masks, 
            #                                                     rank, size, row_groups, 
            #                                                     col_groups, args.timing, 
            #                                                     args.baseline,
            #                                                     args.replicate_graph)

            print(f"batch: {batch}", flush=True)
            print(f"batch.edge_index: {batch.edge_index}", flush=True)
            optimizer.zero_grad()
            logits = model(batch, epoch)
            print(f"logits: {logits}", flush=True)
            print(f"logits.sum: {logits.sum()}", flush=True)
            loss = model.loss_function(logits, batch)     
            print(f"loss: {loss}", flush=True)
            loss.backward()

    total_stop = time.time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IGNN')
    parser.add_argument("--dataset", type=str, default="Cora",
                        help="dataset to train")
    parser.add_argument("--sample-method", type=str, default="ladies",
                        help="sampling algorithm for training")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=4,
                        help="gpus per node")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="number of vertices in minibatch")
    parser.add_argument("--samp-num", type=str, default="2-2",
                        help="number of vertices per layer of minibatch")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggr", type=str, default="mean",
                        help="Aggregator type: mean/sum")
    parser.add_argument('--world-size', default=-1, type=int,
                         help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                         help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                         help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
    parser.add_argument('--hostname', default='127.0.0.1', type=str,
                            help='hostname for rank 0')
    parser.add_argument('--normalize', action="store_true",
                            help='normalize adjacency matrix')
    parser.add_argument('--partitioning', default='ONE5D', type=str,
                            help='partitioning strategy to use')
    parser.add_argument('--replication', default=1, type=int,
                            help='partitioning strategy to use')
    parser.add_argument('--n-bulkmb', default=1, type=int,
                            help='number of minibatches to sample in bulk')
    parser.add_argument('--bulk-batch-fetch', default=1, type=int,
                            help='number of minibatches to fetch features for in bulk')
    parser.add_argument('--n-darts', default=-1, type=int,
                            help='number of darts to throw per minibatch in LADIES sampling')
    parser.add_argument('--semibulk', default=128, type=int,
                            help='number of batches to column extract from in bulk')
    parser.add_argument('--timing', action="store_true",
                            help='whether to turn on timers')
    parser.add_argument('--baseline', action="store_true",
                            help='whether to avoid col selection for baseline comparison')
    parser.add_argument('--replicate-graph', action="store_true",
                            help='replicate adjacency matrix on each device')
    parser.add_argument("--nb-node-layer", type=int, default=2,
                        help="number of hidden node MLP layers")
    parser.add_argument("--nb-edge-layer", type=int, default=2,
                        help="number of hidden edge MLP layers")
    parser.add_argument("--n-graph-iters", type=int, default=8,
                        help="number of message passing iterations")
    args = parser.parse_args()
    args.samp_num = [int(i) for i in args.samp_num.split('-')]
    print(args)

    main(args)
