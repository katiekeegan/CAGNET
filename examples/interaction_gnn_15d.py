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
from torch_geometric.loader import DataLoader, NeighborSampler, NeighborLoader
# from torch_geometric.loader.shadow import ShaDowKHopSampler
from shadow import ShaDowKHopSampler
from torch_geometric.transforms import LargestConnectedComponents
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
import torch_sparse

from cagnet.nn.conv import GCNConv
from cagnet.partitionings import Partitioning
from cagnet.samplers import shadow_sampler
from cagnet.samplers.utils import *
import cagnet.nn.functional as CAGF
import torch.nn.functional as F

from sparse_coo_tensor_cpp import sort_dst_proc_gpu

import socket
import yaml

# import wandb
# wandb.init(project="exatrkx")

class InteractionGNN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, nb_node_layer, nb_edge_layer, n_graph_iters, 
                                        aggr, rank, size, partitioning, replication, device, 
                                        group=None, row_groups=None, col_groups=None, impl="cagnet"):
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
        self.impl = impl

        # torch.manual_seed(0)
        # aggr_list = ["sum", "mean", "max", "std"]
        # self.aggregation = torch_geometric.nn.aggr.MultiAggregation(aggr_list, mode="cat")

        # network_input_size = (1 + 2 * len(aggr_list)) * n_hidden

        self.node_encoder = self.make_mlp(
            input_size=in_feats,
            sizes=[n_hidden] * nb_node_layer,
            output_activation="Tanh",
            hidden_activation="SiLU",
            layer_norm=True,
            batch_norm=False,
        )
        
        self.edge_encoder = self.make_mlp(
            input_size=2 * n_hidden,
            sizes=[n_hidden] * nb_edge_layer,
            output_activation="Tanh",
            hidden_activation="SiLU",
            layer_norm=True,
            batch_norm=False,
        )
        
        in_edge_net = n_hidden * 6
        self.edge_network = nn.ModuleList(
            [
                self.make_mlp(
                    input_size=in_edge_net,
                    sizes=[n_hidden] * nb_edge_layer,
                    output_activation="Tanh",
                    hidden_activation="SiLU",
                    layer_norm=True,
                    batch_norm=False,
                )
                for i in range(n_graph_iters)
            ]
        )
        
        in_node_net = n_hidden * 4
        self.node_network = nn.ModuleList(
            [
                self.make_mlp(
                    input_size=in_node_net,
                    sizes=[n_hidden] * nb_node_layer,
                    output_activation="Tanh",
                    hidden_activation="SiLU",
                    layer_norm=True,
                    batch_norm=False,
                )
                for i in range(n_graph_iters)
            ]
        )
        
        # edge decoder
        self.edge_decoder = self.make_mlp(
            input_size=n_hidden,
            sizes=[n_hidden] * nb_edge_layer,
            output_activation="Tanh",
            hidden_activation="SiLU",
            layer_norm=True,
            batch_norm=False,
        )

        # edge output transform layer
        self.edge_output_transform = self.make_mlp(
            input_size=n_hidden,
            sizes=[n_hidden, 1],
            output_activation=None,
            hidden_activation="SiLU",
            layer_norm=True,
            batch_norm=False,
        )

        # dropout layer
        self.dropout = nn.Dropout(p=0.1)
        
    def make_mlp(
        self, 
        input_size,
        sizes,
        hidden_activation="ReLU",
        output_activation=None,
        layer_norm=False,  # TODO : change name to hidden_layer_norm while ensuring backward compatibility
        output_layer_norm=False,
        batch_norm=False,  # TODO : change name to hidden_batch_norm while ensuring backward compatibility
        output_batch_norm=False,
        input_dropout=0,
        hidden_dropout=0,
        track_running_stats=False,
    ):
        """Construct an MLP with specified fully-connected layers."""
        hidden_activation = getattr(nn, hidden_activation)
        if output_activation is not None:
            output_activation = getattr(nn, output_activation)
        layers = []
        n_layers = len(sizes)
        sizes = [input_size] + sizes
        # Hidden layers
        for i in range(n_layers - 1):
            if i == 0 and input_dropout > 0:
                layers.append(nn.Dropout(input_dropout))
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if layer_norm:  # hidden_layer_norm
                layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
            if batch_norm:  # hidden_batch_norm
                layers.append(
                    nn.BatchNorm1d(
                        sizes[i + 1],
                        eps=6e-05,
                        track_running_stats=track_running_stats,
                        affine=True,
                    )  # TODO : Set BatchNorm and LayerNorm parameters in config file ?
                )
            layers.append(hidden_activation())
            if hidden_dropout > 0:
                layers.append(nn.Dropout(hidden_dropout))
        # Final layer
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        if output_activation is not None:
            if output_layer_norm:
                layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
            if output_batch_norm:
                layers.append(
                    nn.BatchNorm1d(
                        sizes[-1],
                        eps=6e-05,
                        track_running_stats=track_running_stats,
                        affine=True,
                    )  # TODO : Set BatchNorm and LayerNorm parameters in config file ?
                )
            layers.append(output_activation())
        return nn.Sequential(*layers)

    def message_step(self, x, e, src, dst, i):
        # Compute new node features
        edge_inputs = torch.cat([e, x[src], x[dst]], dim=-1)  # order dst src x ?
        e_updated = self.edge_network[i](edge_inputs)
        edge_messages_from_src = scatter_add(e_updated, dst, dim=0, dim_size=x.shape[0])
        edge_messages_from_dst = scatter_add(e_updated, src, dim=0, dim_size=x.shape[0])

        node_inputs = torch.cat(
            [edge_messages_from_src, edge_messages_from_dst, x], dim=-1
        )  # to check : the order dst src  x ?
        x_updated = self.node_network[i](node_inputs)
        return (
            x_updated,
            e_updated,
            self.edge_output_transform(self.edge_decoder(e_updated)),
        )

    def output_step(self, x, start, end, e):
        classifier_inputs = torch.cat([x[start], x[end], e], dim=1)
        classifier_output = self.output_edge_classifier(classifier_inputs).squeeze(-1)
        return classifier_output

    def forward(self, batch, epoch=0):
        x = torch.stack([batch["z"]], dim=-1).float()
        src, dst = batch.edge_index
        x.requires_grad = True
        x = self.node_encoder(x)
        e = self.edge_encoder(torch.cat([x[src], x[dst]], dim=-1))
        
        # if concat
        input_x = x
        input_e = e
        outputs = []
        for i in range(self.n_graph_iters):
            # if concat
            x = torch.cat([x, input_x], dim=-1)
            e = torch.cat([e, input_e], dim=-1)
            x, e, out = self.message_step(x, e, src, dst, i)
            outputs.append(out)

        return outputs[-1].squeeze(-1)

    def loss_function(self, output, batch, balance="proportional"):
        """
        Applies the loss function to the output of the model and the truth labels.
        To balance the positive and negative contribution, simply take the means of each separately.
        Any further fine tuning to the balance of true target, true background and fake can be handled 
        with the `weighting` config option.
        """

        assert hasattr(batch, "y"), "The batch does not have a truth label"
        assert hasattr(batch, "weights"), "The batch does not have a weighting label"
        
        assert hasattr(batch, "y"), (
            "The batch does not have a truth label. Please ensure the batch has a `y`"
            " attribute."
        )
        assert hasattr(batch, "weights"), (
            "The batch does not have a weighting label. Please ensure the batch"
            " weighting is handled in preprocessing."
        )

        if balance not in ["equal", "proportional"]:
            warnings.warn(
                f"{balance} is not a proper choice for the loss balance. Use either 'equal' or 'proportional'. Automatically switching to 'proportional' instead."
            )
            balance = "proportional"

        negative_mask = ((batch.y == 0) & (batch.weights != 0)) | (batch.weights < 0)

        negative_loss = F.binary_cross_entropy_with_logits(
            output[negative_mask],
            torch.zeros_like(output[negative_mask]),
            weight=batch.weights[negative_mask].abs(),
            reduction="sum",
        )

        positive_mask = (batch.y == 1) & (batch.weights > 0)
        positive_loss = F.binary_cross_entropy_with_logits(
            output[positive_mask],
            torch.ones_like(output[positive_mask]),
            weight=batch.weights[positive_mask].abs(),
            reduction="sum",
        )

        if balance == "proportional":
            n = positive_mask.sum() + negative_mask.sum()
            return (
                (positive_loss + negative_loss) / n,
                positive_loss.detach() / n,
                negative_loss.detach() / n,
            )
        else:
            n_pos, n_neg = positive_mask.sum(), negative_mask.sum()
            n = n_pos + n_neg
            return (
                positive_loss / n_pos + negative_loss / n_neg,
                positive_loss.detach() / n,
                negative_loss.detach() / n,
            )

    @torch.no_grad()
    def evaluate(self, test_loader):
        """
        The gateway for the evaluation stage. This class method is called from the eval_stage.py script.
        """
        test_batches = []
        for i, batch in enumerate(test_loader):
            batch = batch.to(self.device)
            print(f"test_batch: {batch}", flush=True)
            output = self(batch)
            loss, pos_loss, neg_loss = self.loss_function(output, batch)
            print(f"test loss: {loss} pos_los: {pos_loss} neg_loss: {neg_loss}", flush=True)

            scores = torch.sigmoid(output)
            batch.scores = scores.detach()
            test_batches.append(batch)

        full_auc, masked_auc = graph_roc_curve("testset", test_batches, "Interaction GNN ROC curve", self.impl)
        print(f"full_auc: {full_auc} type(full_auc): {type(full_auc)}", flush=True)
        print(f"masked_auc: {masked_auc} type(masked_auc): {type(masked_auc)}", flush=True)
        return full_auc, masked_auc

        # # Load data from testset directory
        # graph_constructor = cls(config).to(device)
        # if checkpoint is not None:
        #     print(f"Restoring model from {checkpoint}")
        #     graph_constructor = cls.load_from_checkpoint(checkpoint, hparams=config).to(
        #         device
        #     )
        # graph_constructor.setup(stage="test")

        # plots:
        #   graph_scoring_efficiency: 
        #     title: Interaction GNN Edge-wise Efficiency
        #     pt_units: GeV
        #   graph_roc_curve:
        #     title: Interaction GNN ROC curve
        # all_plots = config["plots"]

        ## TODO: Handle the list of plots properly
        #for plot_function, plot_config in all_plots.items():
        #    if hasattr(eval_utils, plot_function):
        #        getattr(eval_utils, plot_function)(
        #            graph_constructor, plot_config, config
        #        )
        #    else:
        #        print(f"Plot {plot_function} not implemented")

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

            # data_obj = dataset.preprocess_event(data_obj)
            trainset.append(data_obj)
        trainset = Batch.from_data_list(trainset)
        # trainset = trainset.to(device)
        print(f"trainset: {trainset}", flush=True)
        print(f"trainset.x.sum: {trainset.x.sum()}", flush=True)
        print(f"trainset.y.sum: {trainset.y.sum()}", flush=True)
        print(f"trainset.z.sum: {trainset.z.sum()}", flush=True)
        print(f"trainset.truth_map.sum: {trainset.truth_map.sum()}", flush=True)
        print(f"trainset.weights.sum: {trainset.weights.sum()}", flush=True)

        node_count = torch.max(trainset.edge_index) + 1
        edge_count = trainset.edge_index.size(1)

        testset = GraphDataset(input_dir, "testset", 10, "fit", hparams)
        print(f"testset: {testset}", flush=True)
        num_features = 1
        num_classes = 2
    elif args.dataset == "ctd":
        print(f"Loading coo...", flush=True)
        input_dir = "/global/cfs/cdirs/m4439/CTD2023_results/module_map/"
        with open("gnn_train.yaml") as stream:
            hparams = yaml.safe_load(stream)

        print(f"hparams: {hparams}", flush=True)

        print(f"before graphdataset", flush=True)
        dataset = GraphDataset(input_dir, "trainset", 4053, "fit", hparams)
        print(f"after graphdataset", flush=True)

        print(f"dataset: {dataset}", flush=True)
        trainset = []
        for data in dataset:
            if data is None:
                continue
            data_obj = Data(hit_id=data["hit_id"],
                                x=data["x"], 
                                y=data["y"], 
                                z=data["z"], 
                                edge_index=data["edge_index"], 
                                truth_map=data["truth_map"],
                                weights=data["weights"])

            # data_obj = dataset.preprocess_event(data_obj)
            trainset.append(data_obj)
        trainset = Batch.from_data_list(trainset)
        # trainset = trainset.to(device)
        print(f"trainset: {trainset}", flush=True)
        print(f"trainset.x.sum: {trainset.x.sum()}", flush=True)
        print(f"trainset.y.sum: {trainset.y.sum()}", flush=True)
        print(f"trainset.z.sum: {trainset.z.sum()}", flush=True)
        print(f"trainset.truth_map.sum: {trainset.truth_map.sum()}", flush=True)
        print(f"trainset.weights.sum: {trainset.weights.sum()}", flush=True)

        node_count = torch.max(trainset.edge_index) + 1
        edge_count = trainset.edge_index.size(1)

        testset = GraphDataset(input_dir, "testset", 200, "fit", hparams)
        print(f"testset: {testset}", flush=True)
        num_features = 1
        num_classes = 2

    proc_row = size // args.replication
    rank_row = rank // args.replication
    rank_col = rank % args.replication
    if rank_row >= (size // args.replication):
        return

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
                      col_groups=col_groups,
                      impl=args.impl)
    print(f"model: {model}", flush=True)

    model = model.to(device)

    # use optimizer
    print(f"lr: {args.lr}", flush=True)
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=True,
            weight_decay=0.01
        )
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=10,
                    gamma=0.9,
                )

    # torch.manual_seed(0)
    if args.impl == "torch":
        train_loader = ShaDowKHopSampler(trainset, 
                                            depth=args.n_layers, 
                                            num_neighbors=args.num_neighbors, 
                                            batch_size=args.batch_size, 
                                            num_workers=1,
                                            shuffle=False,
                                            drop_last=True)
        print(f"len(train_loader): {len(train_loader)}", flush=True)
    test_loader = DataLoader(testset, batch_size=1, num_workers=1)


    g_loc_indices = trainset.edge_index.to(device)
    # g_loc_values = torch.cuda.FloatTensor(g_loc_indices.size(1)).fill_(1.0)
    g_loc_values = torch.arange(g_loc_indices.size(1), dtype=torch.int64).to(device)
    g_loc = torch.sparse_coo_tensor(g_loc_indices, g_loc_values)
    g_loc = g_loc.to_sparse_csr()
    print(f"g_loc: {g_loc}", flush=True)

    components_small = LargestConnectedComponents(args.batch_size - 1)
    components_large = LargestConnectedComponents(args.batch_size)

    row_n_bulkmb = int(args.n_bulkmb / (size / args.replication))
    if rank == size - 1:
        row_n_bulkmb = args.n_bulkmb - row_n_bulkmb * (proc_row - 1)

    rank_n_bulkmb = int(row_n_bulkmb / args.replication)
    if rank == size - 1:
        rank_n_bulkmb = row_n_bulkmb - rank_n_bulkmb * (args.replication - 1)

    print(f"rank_n_bulkmb: {rank_n_bulkmb}")

    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        print(f"Epoch: {epoch}", flush=True)
        if epoch >= 1:
            epoch_start = time.time()

        batches_all = torch.arange(node_count).to(device)
        # batches_all = torch.randperm(node_count).to(device)
        batch_count = -(node_count // -args.batch_size) # ceil(train_nid.size(0) / batch_size)
        print(f"node_count: {node_count}", flush=True)
        print(f"batch_count: {batch_count}", flush=True)

        if args.impl == "torch":
            for batch, n_id in train_loader:
                optimizer.zero_grad()
                batch = batch.to(device)
                logits = model(batch, epoch)
                loss, pos_loss, neg_loss = model.loss_function(logits, batch)     
                print(f"loss: {loss} pos_loss: {pos_loss} neg_loss: {neg_loss}", flush=True)
                # wandb.log({'loss': loss.item(),
                #                 'pos_loss': pos_loss.item(), 
                #                 'neg_loss': neg_loss.item(), 
                #                 'epoch': epoch})
                loss.backward()
                optimizer.step()

        elif args.impl == "cagnet":
            for b in range(0, batch_count, args.n_bulkmb):
                if b + args.n_bulkmb >= batch_count:
                    break
                batches_start = b * args.batch_size
                batches_stop = (b + args.n_bulkmb) * args.batch_size
                batches = batches_all[batches_start:batches_stop].view(args.n_bulkmb, args.batch_size)
                batches_loc = one5d_partition_mb(rank, size, batches, 1, args.n_bulkmb)

                print(f"batches: {batches}", flush=True)
                print(f"batches_loc: {batches_loc}", flush=True)
                batches_indices_rows = torch.arange(batches_loc.size(0), dtype=torch.int32, device=device)
                batches_indices_rows = batches_indices_rows.repeat_interleave(batches_loc.size(1))
                batches_indices_cols = batches_loc.view(-1)
                batches_indices = torch.stack((batches_indices_rows, batches_indices_cols))
                batches_values = torch.cuda.FloatTensor(batches_loc.size(1) * batches_loc.size(0), 
                                                            device=device).fill_(1.0)
                batches_loc = torch.sparse_coo_tensor(batches_indices, batches_values, 
                                                            (batches_loc.size(0), node_count))

                node_count = trainset.x.size(0)
                if args.n_darts == -1:
                    avg_degree = int(edge_count / node_count)
                    args.n_darts = []
                    avg_degree = edge_count / node_count
                    for d in range(args.n_layers):
                        dart_count = int(avg_degree * args.samp_num / avg_degree)
                        args.n_darts.append(dart_count)

                if args.replicate_graph:
                    rep_pass = 1
                else:
                    rep_pass = args.replication

                frontiers_bulk, adj_matrices_bulk = shadow_sampler(g_loc, batches_loc, args.batch_size, \
                                                                        [args.num_neighbors], args.n_bulkmb, \
                                                                        # 2, args.n_darts, \
                                                                        args.n_layers, args.n_darts, \
                                                                        rep_pass, rank, size, row_groups, \
                                                                        col_groups, args.timing, \
                                                                        args.replicate_graph)

                g_loc_coo = g_loc.to_sparse_coo()

                for i in range(args.n_bulkmb):
                    adj_matrix = adj_matrices_bulk[i].to_sparse_coo()

                    frontier_cpu = frontiers_bulk[i].cpu()
                    edge_ids_cpu = adj_matrix._values().cpu()

                    batch = Batch(batch=frontiers_bulk[i], 
                                    edge_index=adj_matrices_bulk[i]._indices(),
                                    y=trainset.y[edge_ids_cpu],
                                    z=trainset.z[frontier_cpu],
                                    weights=trainset.weights[edge_ids_cpu])

                    optimizer.zero_grad()
                    batch = batch.to(device)
                    logits = model(batch, epoch)
                    loss, pos_loss, neg_loss = model.loss_function(logits, batch)     
                    print(f"loss: {loss} pos_loss: {pos_loss} neg_loss: {neg_loss}", flush=True)
                    # wandb.log({'loss': loss.item(),
                    #                 'pos_loss': pos_loss.item(), 
                    #                 'neg_loss': neg_loss.item(), 
                    #                 'epoch': epoch})
                    loss.backward()
                    optimizer.step()

        if epoch % 5 == 0:
            print(f"Evaluating", flush=True)
            full_auc, masked_auc = model.evaluate(test_loader)

            # for name, W in model.named_parameters():
            #     print(f"name: {name} W.grad.sum: {W.grad.sum()}", flush=True)

        scheduler.step()
        curr_lr = scheduler.get_last_lr()
        print(f"Epoch: {epoch} lr: {curr_lr}", flush=True)
        if epoch >= 1:
            dur.append(time.time() - epoch_start)
        if epoch >= 1 and epoch % 5 == 0:
            print(f"Epoch time: {np.sum(dur) / epoch}", flush=True)
            # wandb.log({'full_auc': full_auc.item(),
            #                 'masked_auc': masked_auc.item(), 
            #                 'time': np.sum(dur).item()})
            if epoch == 5:
                break
    total_stop = time.time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IGNN')
    parser.add_argument("--dataset", type=str, default="Cora",
                        help="dataset to train")
    parser.add_argument("--sample-method", type=str, default="shadow",
                        help="sampling algorithm for training")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=4,
                        help="gpus per node")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="number of vertices in minibatch")
    parser.add_argument("--num-neighbors", type=int, default=1,
                        help="number of neigbors per vertex of minibatch")
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
    parser.add_argument('--impl', default='cagnet', type=str,
                            help='sampling implementation to use (torch/cagnet)')
    args = parser.parse_args()
    args.samp_num = args.num_neighbors
    print(args)

    main(args)
