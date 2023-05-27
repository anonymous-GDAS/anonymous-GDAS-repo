import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import ArgumentParser
from dual_gnn.cached_gcn_conv import CachedGCNConv
from dual_gnn.dataset.DomainData import DomainData
from dual_gnn.ppmi_conv import PPMIConv
from dual_gnn.utils.switch_aug import switch_aug
from dual_gnn.utils.visual import visualization
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
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn.conv import GINConv, SAGEConv
from torch_geometric.nn.inits import glorot, zeros
from dual_gnn.dataset.DomainDataNew import DomainDataNew
from tqdm import tqdm
from dual_gnn.models.augmentation import MMD, CMD, CDAN
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='acm')
parser.add_argument("--target", type=str, default='citation')
parser.add_argument("--name", type=str, default='UDAGCN')
parser.add_argument("--seed", type=int, default=200)
parser.add_argument("--UDAGCN", type=bool, default=False)
parser.add_argument("--CDAN", type=bool, default=True)
parser.add_argument("--MMD", type=bool, default=False)
parser.add_argument("--CMD", type=bool, default=False)
parser.add_argument("--reg", type=float, default=0.1)
parser.add_argument("--encoder_dim", type=int, default=16)
parser.add_argument("--aug1", type=str, default='permE')
parser.add_argument("--type", type=str, default='gcn')
parser.add_argument("--aug_ratio1", type=float, default=0.0)

args = parser.parse_args()
seed = args.seed
use_UDAGCN = args.UDAGCN
use_CDAN = args.CDAN
print(use_CDAN)
encoder_dim = args.encoder_dim
reg = args.reg

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
        loss = cls_loss + reg * loss_cdan
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
line = "best_source_acc: {}, best_target_acc: {}, type: {}, source: {}, target: {}, UDA: {}".format(best_source_acc,
                                                                                                    best_target_acc,
                                                                                                    args.type,
                                                                                                    args.source,
                                                                                                    args.target,
                                                                                                    args.UDAGCN)

print(line)
print("=============================================================")
