import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import ArgumentParser
from dual_gnn.cached_gcn_conv import CachedGCNConv
from dual_gnn.dataset.DomainDataNew import DomainDataNew
from dual_gnn.ppmi_conv import PPMIConv
from torch.nn import Parameter
import random
import numpy as np
import torch
import torch.functional as F
from torch import nn
import torch.nn.functional as F
import itertools
from scipy.stats import wasserstein_distance
import networkx as nx
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn.conv import GINConv, SAGEConv
from tqdm import tqdm
import scipy.linalg as spl
from dual_gnn.models.augmentation import MMD, CMD, CDAN
from sample_graph import MHRW, RWRW, Sampler_Our, Sampler_OT
from torch_geometric.utils import subgraph
from torch_geometric.data import Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument("--source", type=str, default='acm')
parser.add_argument("--target", type=str, default='citation')
parser.add_argument("--name", type=str, default='UDAGCN')
parser.add_argument("--seed", type=int, default=200)
parser.add_argument("--UDAGCN", type=bool, default=False)
parser.add_argument("--DANN", type=bool, default=False)
parser.add_argument("--CDAN", type=bool, default=False)
parser.add_argument("--MMD", type=bool, default=False)
parser.add_argument("--CMD", type=bool, default=False)
parser.add_argument("--reg", type=float, default=1.0)
parser.add_argument("--encoder_dim", type=int, default=16)
parser.add_argument("--aug1", type=str, default='permE')
parser.add_argument("--type", type=str, default='gcn')
parser.add_argument("--aug_ratio", type=float, default=0.2)
parser.add_argument("--sample", type=str, default='OT-T')
parser.add_argument("--ptype", type=str, default='degree')
parser.add_argument("--itype", type=str, default='wl')
parser.add_argument("--eq", type=int, default=2)
parser.add_argument("--bsz", type=int, default=10)
parser.add_argument("--device", type=int, default=7)

args = parser.parse_args()
seed = args.seed
use_UDAGCN = args.UDAGCN
use_CDAN = args.CDAN
use_DANN = args.DANN
use_MMD = args.MMD
use_CMD = args.CMD
print("use_MMD:", use_MMD)
print("use_CMD:", use_CMD)
print("use_DANN:", use_DANN)
print("use_CDAN:", use_CDAN)
print("use_UDAGCN:", use_CDAN)
if use_MMD:
    method = 'MMD'
elif use_CMD:
    method = 'CMD'
elif use_DANN:
    method = 'DANN'
elif use_CDAN:
    method = 'CDAN'
elif use_UDAGCN:
    method = 'UDAGCN'
else:
    method = 'GCN'

encoder_dim = args.encoder_dim
reg = args.reg
device_idx = args.device

id = "source: {}, target: {}, seed: {}, UDAGCN: {}, encoder_dim: {}" \
    .format(args.source, args.target, seed, use_UDAGCN, encoder_dim)

print(id)

root = "."
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
dataset = DomainDataNew(root + "/data/{}".format(args.source), name=args.source)
source_data = dataset[0]
dataset = DomainDataNew(root + "/data/{}".format(args.target),
                        name=args.target)
target_data = dataset[0]

source_G_before = to_networkx(source_data)
source_G = to_networkx(source_data)
target_G = to_networkx(target_data)
source_data = source_data.to(device)
target_data = target_data.to(device)


class GNN(torch.nn.Module):
    def __init__(self, base_model=None, type="gcn", **kwargs):
        super(GNN, self).__init__()

        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]

        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.type = type

        if type == "ppmi":
            model_cls = PPMIConv
        elif type == 'gcn':
            model_cls = CachedGCNConv
        elif type == 'gin':
            model_cls = GINConv
        elif type == 'sage':
            model_cls = SAGEConv

        self.conv_layers = nn.ModuleList([
            model_cls(dataset.num_features, 128,
                      weight=weights[0],
                      bias=biases[0],
                      **kwargs),
            model_cls(128, encoder_dim,
                      weight=weights[1],
                      bias=biases[1],
                      **kwargs)
        ])

        if type == 'gin':
            self.conv_layers[0].nn = nn.Linear(dataset.num_features, 128)
            self.conv_layers[1].nn = nn.Linear(128, encoder_dim)

    def forward(self, x, edge_index, cache_name):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x

    def forward_other(self, x, edge_index):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)


loss_func = nn.CrossEntropyLoss().to(device)

GNN_type = args.type
encoder = GNN(type=GNN_type).to(device)
if use_UDAGCN:
    ppmi_encoder = GNN(base_model=encoder, type="ppmi", path_len=10).to(device)

cls_model = nn.Sequential(
    nn.Linear(encoder_dim, dataset.num_classes),
).to(device)

if use_CDAN == False:
    domain_model = nn.Sequential(
        GRL(),
        nn.Linear(encoder_dim, 40),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(40, 2),
    ).to(device)
else:
    domain_model = nn.Sequential(
        GRL(),
        nn.Linear(80, 40),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(40, 2),
    ).to(device)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


# att_model = Attention(encoder_dim).cuda()
att_model = Attention(encoder_dim).to(device)

models = [encoder, cls_model, domain_model]
if use_UDAGCN:
    models.extend([ppmi_encoder, att_model])
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=3e-3)


def gcn_encode(data, cache_name, mask=None):
    if GNN_type == 'gin' or GNN_type == 'sage':
        encoded_output = encoder.forward_other(data.x, data.edge_index)
    else:
        encoded_output = encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def ppmi_encode(data, cache_name, mask=None):
    encoded_output = ppmi_encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def encode(data, cache_name, mask=None):
    gcn_output = gcn_encode(data, cache_name, mask)
    if use_UDAGCN:
        ppmi_output = ppmi_encode(data, cache_name, mask)
        outputs = att_model([gcn_output, ppmi_output])
        return outputs
    else:
        return gcn_output


def predict(data, cache_name, mask=None):
    encoded_output = encode(data, cache_name, mask)
    logits = cls_model(encoded_output)
    return logits


def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return accuracy


def test(data, cache_name, mask=None):
    for model in models:
        model.eval()
    logits = predict(data, cache_name, mask)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    accuracy = evaluate(preds, labels)
    return accuracy


epochs = 200


def to_nxgraph(pyg_graph):
    edge_index = pyg_graph.edge_index.cpu().numpy().transpose()
    edge_list = edge_index.tolist()
    nx_G = nx.Graph(edge_list)
    return nx_G


def check_empty(graph):
    nodelist = list(graph)
    nlen = len(nodelist)
    if nlen == 0:
        return True
    else:
        return False


def calculate_hop_laplacian(source_graph, target_graph):
    distance_list = []
    source_list = []
    target_list = []
    for i in tqdm(range(1000)):
        source_nd = random.randint(0, source_graph.num_nodes - 1)
        source_p = k_hop_subgraph(source_nd, 2, source_graph.edge_index, relabel_nodes=True)
        source_node_set, source_edge_index = source_p[0], source_p[1]

        target_nd = random.randint(0, target_graph.num_nodes - 1)
        target_p = k_hop_subgraph(target_nd, 2, target_graph.edge_index, relabel_nodes=True)
        target_node_set, target_edge_index = target_p[0], target_p[1]

        source_sub_g = nx.Graph()
        source_sub_g.add_edges_from(source_edge_index.t().tolist())
        if check_empty(source_sub_g):
            continue
        source_L = nx.normalized_laplacian_matrix(source_sub_g)
        source_L = torch.from_numpy(source_L.todense()).float()
        eva_L_source, evt = spl.eigh(source_L)

        target_sub_g = nx.Graph()
        target_sub_g.add_edges_from(target_edge_index.t().tolist())
        if check_empty(target_sub_g):
            continue
        target_L = nx.normalized_laplacian_matrix(target_sub_g)
        target_L = torch.from_numpy(target_L.todense()).float()
        eva_L_target, evt = spl.eigh(target_L)

        distance = wasserstein_distance(eva_L_source, eva_L_target)
        distance_list.append(distance)
        source_list.append(eva_L_source.tolist())
        target_list.append(eva_L_target.tolist())

    source_flat_list = [item for sublist in source_list for item in sublist]
    target_flat_list = [item for sublist in target_list for item in sublist]
    # return np.mean(distance_list),wasserstein_distance(source_flat_list,target_flat_list)
    return source_flat_list, target_flat_list, np.mean(distance_list)


def calculate_distance(source_graph, target_graph, source_G, target_G, type):
    print(type)
    if type == 'degree_centrality':
        source_value = nx.degree_centrality(source_graph)
        target_value = nx.degree_centrality(target_graph)
        distance = wasserstein_distance(source_value, target_value)
    elif type == 'closeness_centrality':
        source_value = nx.closeness_centrality(source_graph)
        target_value = nx.closeness_centrality(target_graph)
        distance = wasserstein_distance(source_value, target_value)
    elif type == 'eigenvector_centrality':
        source_value = nx.eigenvector_centrality(source_graph)
        target_value = nx.eigenvector_centrality(target_graph)
        distance = wasserstein_distance(source_value, target_value)
    elif type == 'betweenness_centrality':
        source_value = nx.betweenness_centrality(source_graph)
        target_value = nx.betweenness_centrality(target_graph)
        distance = wasserstein_distance(source_value, target_value)
    elif type == 'degree_his':
        source_value = nx.degree_histogram(source_G)
        target_value = nx.degree_histogram(target_G)
        distance = wasserstein_distance(source_value, target_value)
    elif type == 'degree':
        source_value = [val for (node, val) in source_G.degree()]
        target_value = [val for (node, val) in target_G.degree()]
        distance = wasserstein_distance(source_value, target_value)
    elif type == 'eigen':
        source_value, target_value, distance = calculate_hop_laplacian(source_graph, target_graph)
    elif type == 'eigen_sub_all':
        source_value, target_value, distance = calculate_hop_laplacian(source_graph, target_graph)
        distance = wasserstein_distance(source_value, target_value)
    elif type == 'eigen_all':
        edge_index = source_graph.edge_index.numpy().transpose()
        edge_list = edge_index.tolist()
        source_G_tmp = nx.Graph(edge_list)

        source_L = nx.normalized_laplacian_matrix(source_G_tmp)
        source_L = torch.from_numpy(source_L.todense()).float()
        source_value, evt = spl.eigh(source_L)

        edge_index = target_graph.edge_index.numpy().transpose()
        edge_list = edge_index.tolist()
        target_G_tmp = nx.Graph(edge_list)

        target_L = nx.normalized_laplacian_matrix(target_G_tmp)
        target_L = torch.from_numpy(target_L.todense()).float()
        target_value, evt = spl.eigh(target_L)
        distance = wasserstein_distance(source_value, target_value)
    return source_value, target_value, distance


def sampling_graph(source_value, target_value, source, type, ip_type, ratio=0.1):
    ep = 10
    bsz = 100
    if type == 'MHRW':
        N = source['x'].shape[0]
        source_G = to_nxgraph(source)
        sampler = MHRW()
        sample_graph = sampler.mhrw(source_G, 0, N * ratio)
        node_list = torch.LongTensor(list(sample_graph.nodes()))
        subgraph_edge_index, _ = subgraph(list(sample_graph.nodes()), source['edge_index'], relabel_nodes=True)
        pyg_sample_graph = Data(x=source['x'][node_list], edge_index=subgraph_edge_index, y=source['y'][node_list])
        pyg_sample_graph.train_mask = source['train_mask'][node_list]
        pyg_sample_graph.val_mask = source['val_mask'][node_list]
        pyg_sample_graph.test_mask = source['test_mask'][node_list]
        return pyg_sample_graph
    elif type == 'RW':
        N = source['x'].shape[0]
        node_list = torch.LongTensor(np.random.choice(N - 1, int(N * ratio), replace=False))
        subgraph_edge_index, _ = subgraph(node_list, source['edge_index'], relabel_nodes=True)
        pyg_sample_graph = Data(x=source['x'][node_list], edge_index=subgraph_edge_index, y=source['y'][node_list])
        pyg_sample_graph.train_mask = source['train_mask'][node_list]
        pyg_sample_graph.val_mask = source['val_mask'][node_list]
        pyg_sample_graph.test_mask = source['test_mask'][node_list]
        return pyg_sample_graph
    elif type == 'MS':
        N = source['x'].shape[0]
        source_G = to_nxgraph(source)
        sampler = MHRW()
        sample_graph = sampler.mhrw(source_G, 0, N * ratio)

        node_list = list(sample_graph.nodes())
        new_node_list = [i for i in range(len(node_list))]
        mapping_labels = dict(zip(node_list, new_node_list))
        G = nx.relabel_nodes(sample_graph, mapping_labels)
        subgraph_edge_index = torch.tensor(list(G.edges)).t().contiguous()
        pyg_sample_graph = Data(x=source['x'][node_list], edge_index=subgraph_edge_index, y=source['y'][node_list])
        pyg_sample_graph.train_mask = source['train_mask'][node_list]
        pyg_sample_graph.val_mask = source['val_mask'][node_list]
        pyg_sample_graph.test_mask = source['test_mask'][node_list]
        return pyg_sample_graph
    elif type == 'OT':
        N = source['x'].shape[0]
        source_G = to_nxgraph(source)
        sampler = Sampler_OT(source_value, target_value, device, ep, bsz, ip_type)
        sample_graph = sampler.sample(source_G, 0, N * ratio)

        node_list = list(sample_graph.nodes())
        new_node_list = [i for i in range(len(node_list))]
        mapping_labels = dict(zip(node_list, new_node_list))
        G = nx.relabel_nodes(sample_graph, mapping_labels)
        subgraph_edge_index = torch.tensor(list(G.edges)).t().contiguous()
        pyg_sample_graph = Data(x=source['x'][node_list], edge_index=subgraph_edge_index, y=source['y'][node_list])
        pyg_sample_graph.train_mask = source['train_mask'][node_list]
        pyg_sample_graph.val_mask = source['val_mask'][node_list]
        pyg_sample_graph.test_mask = source['test_mask'][node_list]
        return pyg_sample_graph
    elif type == 'GDAS-random':
        N = source['x'].shape[0]
        sampler = Sampler_OT(source_value, target_value, device, ep, bsz, ip_type)
        p = sampler.importance
        s = sum(p)
        p = [i / s for i in p]
        node_list = torch.LongTensor(np.random.choice(N, int(N * ratio), replace=False, p=p))
        subgraph_edge_index, _ = subgraph(node_list, source['edge_index'], relabel_nodes=True)
        pyg_sample_graph = Data(x=source['x'][node_list], edge_index=subgraph_edge_index, y=source['y'][node_list])
        pyg_sample_graph.train_mask = source['train_mask'][node_list]
        pyg_sample_graph.val_mask = source['val_mask'][node_list]
        pyg_sample_graph.test_mask = source['test_mask'][node_list]
        return pyg_sample_graph
    elif type == 'GDAS':
        N = source['x'].shape[0]
        source_G = to_nxgraph(source)
        sampler = Sampler_OT(source_value, target_value, device, ep, bsz, ip_type)
        sample_graph = sampler.sample(source_G, 0, N * ratio)
        node_list = list(sample_graph.nodes())
        subgraph_edge_index, _ = subgraph(node_list, source['edge_index'], relabel_nodes=True)
        pyg_sample_graph = Data(x=source['x'][node_list], edge_index=subgraph_edge_index, y=source['y'][node_list])
        pyg_sample_graph.train_mask = source['train_mask'][node_list]
        pyg_sample_graph.val_mask = source['val_mask'][node_list]
        pyg_sample_graph.test_mask = source['test_mask'][node_list]
        return pyg_sample_graph


def train(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()

    global rate
    rate = min((epoch + 1) / epochs, 0.05)

    encoded_source = encode(source_data, "source")
    encoded_target = encode(target_data, "target")
    source_logits = cls_model(encoded_source)

    # use source classifier loss:
    cls_loss = loss_func(source_logits, source_data.y)

    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name:
                cls_loss = cls_loss + param.mean() * 3e-3

    if use_MMD:
        mmd_loss = MMD(encoded_source, encoded_target)
        cls_loss = args.reg * mmd_loss + cls_loss
    elif use_CMD:
        cmd_loss = CMD(encoded_source, encoded_target)
        cls_loss = args.reg * cmd_loss + cls_loss

    if use_UDAGCN:
        # use domain classifier loss:
        source_domain_preds = domain_model(encoded_source)
        target_domain_preds = domain_model(encoded_target)

        source_domain_cls_loss = loss_func(
            source_domain_preds,
            torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(device)
        )
        target_domain_cls_loss = loss_func(
            target_domain_preds,
            torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device)
        )
        loss_grl = source_domain_cls_loss + target_domain_cls_loss
        loss = cls_loss + loss_grl

        # use target classifier loss:
        target_logits = cls_model(encoded_target)
        target_probs = F.softmax(target_logits, dim=-1)
        target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

        loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

        loss = loss + loss_entropy * (epoch / epochs * 0.01)
    elif use_CDAN:
        target_logits = cls_model(encoded_target)

        feature = torch.cat((encoded_source, encoded_target), 0)
        softmax_output = torch.cat((source_logits, target_logits), 0)
        label = torch.cat((torch.zeros(source_logits.size(0)).type(torch.LongTensor),
                           torch.ones(target_logits.size(0)).type(torch.LongTensor)), 0).to(device)
        cdan_pred = CDAN(feature, softmax_output, domain_model)
        loss_cdan = loss_func(cdan_pred, label)
        loss = cls_loss + args.reg * loss_cdan
    elif use_DANN:
        source_domain_preds = domain_model(encoded_source)
        target_domain_preds = domain_model(encoded_target)

        source_domain_cls_loss = loss_func(
            source_domain_preds,
            torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(device)
        )
        target_domain_cls_loss = loss_func(
            target_domain_preds,
            torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device)
        )
        loss_grl = source_domain_cls_loss + target_domain_cls_loss
        loss = cls_loss + args.reg * loss_grl
    else:
        loss = cls_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
source_acc = []
target_acc = []

type = args.ptype
sampling_type = args.sample
im_type = args.itype
source_value, target_value, previous_distance = calculate_distance(source_data, target_data, source_G, target_G, type)
print(source_data)
print('before distance:', previous_distance)
source_data = sampling_graph(source_value, target_value, source_data, sampling_type, im_type, 1 - args.aug_ratio)
source_G = to_networkx(source_data)
target_G = to_networkx(target_data)
source_value, target_value, distance = calculate_distance(source_data, target_data, source_G, target_G, type)
source_data = source_data.to(device)
target_data = target_data.to(device)
print(source_data)
print('after distance:', distance)

for epoch in tqdm(range(1, epochs)):
    # print(encoder.conv_layers[0].nn.weight)
    train(epoch)
    source_correct = test(source_data, "source", source_data.test_mask)
    source_acc.append(source_correct)
    target_correct = test(target_data, "target")
    target_acc.append(target_correct)
    # print("Epoch: {}, source_acc: {}, target_acc: {}".format(epoch, source_correct, target_correct))
    if target_correct > best_target_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_epoch = epoch

print("=============================================================")
line = "best_source_acc: {}, best_target_acc: {}, type: {}, source: {}, target: {}, UDA: {}, method: {}".format(
    best_source_acc,
    best_target_acc,
    args.type,
    args.source,
    args.target,
    args.UDAGCN, method)

print(line)
print("=============================================================")
print("#result#" + str(previous_distance) + "#" + str(distance) + "#" + str(best_target_acc.cpu().item()))
