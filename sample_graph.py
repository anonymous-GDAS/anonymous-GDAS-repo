import random
import time
import networkx as nx
import numpy as np
from tqdm import tqdm
from scipy.stats import wasserstein_distance
import copy
import torch

from OT.Sinkhorn_distance import SinkhornDistance
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

eplisons = 1.0
criterion_label = SinkhornDistance(eps=eplisons, max_iter=200, reduction='sum', dis='euc').to('cuda:0')

from torch.autograd import Variable


def to_var(x, device, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda(device)
    return Variable(x, requires_grad=requires_grad)


def softmax_normalize(weights, temperature=1.):
    return torch.nn.functional.softmax(weights / temperature, dim=0)


def linear_normalize(weights):
    weights = torch.max(weights, torch.zeros_like(weights))
    if torch.sum(weights) > 1e-8:
        return weights / torch.sum(weights)


class SRW_RWF_ISRW():
    def __init__(self):
        self.T = 50  # number of iterations
        self.growth_size = 2

    def random_walk_sampling_simple(self, complete_graph, pos_node, step_each_pos_node_to_walk, num_epoches=1):
        nr_nodes = len(complete_graph.nodes())
        edges_before_t_iter = 0
        sampled_graph = nx.Graph()

        for ep in range(num_epoches):
            for index_of_first_random_node in pos_node:
                sampled_graph.add_node(index_of_first_random_node)
                curr_node = index_of_first_random_node
                for iteration in range(step_each_pos_node_to_walk):
                    edges = [n for n in complete_graph.neighbors(curr_node)]
                    index_of_edge = random.randint(0, len(edges) - 1)
                    chosen_node = edges[index_of_edge]
                    sampled_graph.add_node(chosen_node)
                    sampled_graph.add_edge(curr_node, chosen_node)
                    curr_node = chosen_node

                    if iteration % self.T == self.T - 1:
                        if ((sampled_graph.number_of_edges() - edges_before_t_iter) < self.growth_size):
                            curr_node = random.randint(0, nr_nodes - 1)
                        edges_before_t_iter = sampled_graph.number_of_edges()

        return sampled_graph


class MHRW():
    def __init__(self):
        self.G1 = nx.Graph()

    def mhrw(self, G, node, size):
        dictt = {}
        node_list = set()
        node_list.add(node)
        parent_node = node_list.pop()
        dictt[parent_node] = parent_node
        degree_p = G.degree(parent_node)
        related_list = list(G.neighbors(parent_node))
        node_list.update(related_list)

        history = 0
        count = 0
        while (len(self.G1.nodes()) < size):
            value = len(self.G1.nodes()) / size
            if value == history:
                count = count + 1
            else:
                count = 0
                history = value
            if count >= 100:
                break
            if (len(node_list) > 0):
                child_node = node_list.pop()
                p = round(random.uniform(0, 1), 4)
                if (child_node not in dictt):
                    related_listt = list(G.neighbors(child_node))
                    degree_c = G.degree(child_node)
                    dictt[child_node] = child_node
                    if (p <= min(1, degree_p / degree_c) and child_node in list(G.neighbors(parent_node))):
                        self.G1.add_edge(parent_node, child_node)
                        parent_node = child_node
                        degree_p = degree_c
                        node_list.clear()
                        node_list.update(related_listt)
                    else:
                        del dictt[child_node]
            # node_list set becomes empty or size is not reached
            # insert some random nodes into the set for next processing
            else:
                node_list.update(random.sample(set(G.nodes()) - set(self.G1.nodes()), 3))
                parent_node = node_list.pop()
                G.add_node(parent_node)
                related_list = list(G.neighbors(parent_node))
                node_list.clear()
                node_list.update(related_list)
        return self.G1


class Sampler_Our():
    def __init__(self, source_value, target_value, ip_type='wl'):
        self.G1 = nx.Graph()
        self.source_value = source_value
        self.target_value = target_value
        self.ip_type = ip_type
        self.importance = self.calculate_importance(self.source_value, self.target_value)

    def calculate_wl_importance(self, source_value, target_value):
        gap = []
        origin = wasserstein_distance(source_value, target_value)
        for i in tqdm(range(len(source_value))):
            tmp = copy.deepcopy(source_value)
            tmp[i] = tmp[i] - 1
            new_distance = wasserstein_distance(tmp, target_value)
            gap.append(new_distance - origin)
        return gap

    def get_kde(self, x, data_array, bandwidth=0.1):
        def gauss(x):
            import math
            return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (x ** 2))

        N = len(data_array)
        res = 0
        if len(data_array) == 0:
            return 0
        for i in range(len(data_array)):
            res += gauss((x - data_array[i]) / bandwidth)
        res /= (N * bandwidth)
        return res

    def importance_KDE(self, source_value, target_value):
        result_list = []
        for i in tqdm(source_value):
            result = self.get_kde(i, target_value)
            result_list.append(result)
        return np.array(source_value) - np.array(result_list)

    def normalization(self, vector):
        minimum = min(vector)
        maximum = max(vector)
        return [(val - minimum) / (maximum - minimum) + 1e-6 for val in vector]

    def calculate_importance(self, source, target):
        if self.ip_type == 'kde':
            score = self.importance_KDE(source, target)
        elif self.ip_type == 'wl':
            score = self.calculate_wl_importance(source, target)
        score_softmax = self.normalization(score)
        return score_softmax

    def sample(self, G, node, size):
        dictt = {}
        node_list = set()
        node_list.add(node)
        parent_node = node_list.pop()
        dictt[parent_node] = parent_node
        degree_p = G.degree(parent_node)
        related_list = list(G.neighbors(parent_node))
        node_list.update(related_list)

        history = 0
        count = 0
        while (len(self.G1.nodes()) < size):
            value = len(self.G1.nodes()) / size
            if value == history:
                count = count + 1
            else:
                count = 0
                history = value
            if count >= 100:
                break
            if (len(node_list) > 0):
                child_node = node_list.pop()
                p = round(random.uniform(0, 1), 4)
                if (child_node not in dictt):
                    related_listt = list(G.neighbors(child_node))
                    degree_c = G.degree(child_node)
                    dictt[child_node] = child_node
                    if (p <= min(1, self.importance[child_node] * degree_p / (
                            degree_c * self.importance[parent_node])) and child_node in list(G.neighbors(parent_node))):
                        self.G1.add_edge(parent_node, child_node)
                        parent_node = child_node
                        degree_p = degree_c
                        node_list.clear()
                        node_list.update(related_listt)
                    else:
                        del dictt[child_node]
            # node_list set becomes empty or size is not reached
            # insert some random nodes into the set for next processing
            else:
                node_list.update(random.sample(set(G.nodes()) - set(self.G1.nodes()), 3))
                parent_node = node_list.pop()
                G.add_node(parent_node)
                related_list = list(G.neighbors(parent_node))
                node_list.clear()
                node_list.update(related_list)
        return self.G1


class Sampler_OT():
    def __init__(self, source_value, target_value, device, ep, bsz=10, ip_type='wl'):
        self.G1 = nx.Graph()
        self.source_value = source_value
        self.target_value = target_value
        self.device = device
        self.epoch = ep
        self.bsz = bsz
        self.importance = self.calculate_importance(self.source_value, self.target_value)

    def calculate_ot_importance(self, source, target):
        if len(source) == 0 or len(target) == 0:
            return []
        source_value = torch.Tensor(source)
        target_value = torch.Tensor(target)
        source_value = source_value.reshape(-1, 1)
        target_value = target_value.reshape(-1, 1)
        tmp = torch.rand((source_value.shape[0], target_value.shape[0]))
        weight = to_var(tmp, self.device)
        # print('before:', torch.sum(linear_normalize(weight), dim=1))
        Attoptimizer = torch.optim.Adam([weight], lr=1.0, weight_decay=5e-4)
        # normal_list = []
        # our_list = []
        for epoch in range(self.epoch):
            # print(weight)
            probability_train = linear_normalize(weight)
            # print(probability_train)
            OTloss = criterion_label(torch.tensor(source_value, dtype=float).cuda(),
                                     torch.tensor(target_value, dtype=float).cuda(),
                                     probability_train.squeeze())
            # our_list.append(OTloss.cpu().item())

            Attoptimizer.zero_grad()
            OTloss.backward()
            Attoptimizer.step()
            # with torch.no_grad():
            #     tmp = torch.ones((source_value.shape[0], target_value.shape[0])) / (
            #             source_value.shape[0] * target_value.shape[0])
            #     probability_train = to_var(tmp, self.device)
            #     probability_train = linear_normalize(probability_train)
            #     normal = criterion_label(torch.tensor(source_value, dtype=float).cuda(),
            #                              torch.tensor(target_value, dtype=float).cuda(),
            #                              probability_train.squeeze())
            #     normal_list.append(normal.cpu().item())
            # print(weight.data)
        # print('after:', torch.sum(linear_normalize(weight), dim=1))
        # return torch.sum(linear_normalize(weight), dim=1).cpu().detach().numpy().tolist(), normal_list[-1], our_list[-1]
        return torch.sum(linear_normalize(weight), dim=1).cpu().detach().numpy().tolist()

    def normalization(self, vector):
        minimum = min(vector)
        maximum = max(vector)
        return [(val - minimum) / (maximum - minimum) + 1e-6 for val in vector]

    def calculate_importance(self, source, target):
        ip_list = []
        before_list = []
        after_list = []
        for i in tqdm(range(len(source) // self.bsz + 1)):
            score = self.calculate_ot_importance(source[self.bsz * i:min(self.bsz * (i + 1), len(source))],
                                                 random.sample(list(target), 50))
            # score= self.calculate_ot_importance(source[self.bsz * i:min(self.bsz * (i + 1), len(source))],
            #                                            random.sample(target, self.bsz))
            # before_list.append(a)
            # after_list.append(b)
            for k in score:
                ip_list.append(k)
        score_softmax = self.normalization(ip_list)
        # print("Mean difference:", np.mean(after_list) - np.mean(before_list))
        return score_softmax

    def sample(self, G, node, size):
        dictt = {}
        node_list = set()
        node_list.add(node)
        parent_node = node_list.pop()
        dictt[parent_node] = parent_node
        degree_p = G.degree(parent_node)
        related_list = list(G.neighbors(parent_node))
        node_list.update(related_list)

        history = 0
        count = 0
        while (len(self.G1.nodes()) < size):
            value = len(self.G1.nodes()) / size
            if value == history:
                count = count + 1
            else:
                count = 0
                history = value
            if count >= 100:
                break
            if (len(node_list) > 0):
                child_node = node_list.pop()
                p = round(random.uniform(0, 1), 4)
                if (child_node not in dictt):
                    related_listt = list(G.neighbors(child_node))
                    degree_c = G.degree(child_node)
                    dictt[child_node] = child_node
                    if (p <= min(1, self.importance[child_node] * degree_p / (
                            degree_c * self.importance[parent_node])) and child_node in list(G.neighbors(parent_node))):
                        self.G1.add_edge(parent_node, child_node)
                        parent_node = child_node
                        degree_p = degree_c
                        node_list.clear()
                        node_list.update(related_listt)
                    else:
                        del dictt[child_node]
            # node_list set becomes empty or size is not reached
            # insert some random nodes into the set for next processing
            else:
                node_list.update(random.sample(set(G.nodes()) - set(self.G1.nodes()), 3))
                parent_node = node_list.pop()
                G.add_node(parent_node)
                related_list = list(G.neighbors(parent_node))
                node_list.clear()
                node_list.update(related_list)
        # p = self.G1.nodes
        # for node in p:
        #     related_list = list(G.neighbors(node))
        #     for k in related_list:
        #         if k in p:
        #             self.G1.add_edge(node, k)
        #             self.G1.add_edge(k, node)
        return self.G1


class Sampler_Tune():
    def __init__(self, source_value, target_value, device, ep, bsz=10, ip_type='wl'):
        self.G1 = nx.Graph()
        self.source_value = source_value
        self.target_value = target_value
        self.device = device
        self.epoch = ep
        self.bsz = bsz
        self.importance = self.calculate_importance(self.source_value, self.target_value)

    def calculate_ot_importance(self, source, target):
        if len(source) == 0:
            return []
        source_value = torch.Tensor(source)
        target_value = torch.Tensor(target)
        source_value = source_value.reshape(-1, 1)
        target_value = target_value.reshape(-1, 1)
        tmp = torch.rand((source_value.shape[0], target_value.shape[0]))
        weight = to_var(tmp, self.device)
        # print('before:', torch.sum(linear_normalize(weight), dim=1))
        Attoptimizer = torch.optim.Adam([weight], lr=1.0, weight_decay=5e-4)
        # normal_list = []
        # our_list = []
        for epoch in range(self.epoch):
            probability_train = linear_normalize(weight)
            OTloss = criterion_label(torch.tensor(source_value, dtype=float).cuda(),
                                     torch.tensor(target_value, dtype=float).cuda(),
                                     probability_train.squeeze())
            # our_list.append(OTloss.cpu().item())

            Attoptimizer.zero_grad()
            OTloss.backward()
            Attoptimizer.step()
            # with torch.no_grad():
            #     tmp = torch.ones((source_value.shape[0], target_value.shape[0])) / (
            #             source_value.shape[0] * target_value.shape[0])
            #     probability_train = to_var(tmp, self.device)
            #     probability_train = linear_normalize(probability_train)
            #     normal = criterion_label(torch.tensor(source_value, dtype=float).cuda(),
            #                              torch.tensor(target_value, dtype=float).cuda(),
            #                              probability_train.squeeze())
            #     normal_list.append(normal.cpu().item())
            # print(weight.data)
        # print('after:', torch.sum(linear_normalize(weight), dim=1))
        # return torch.sum(linear_normalize(weight), dim=1).cpu().detach().numpy().tolist(), normal_list[-1], our_list[-1]
        return torch.sum(linear_normalize(weight), dim=1).cpu().detach().numpy().tolist()

    def normalization(self, vector):
        minimum = min(vector)
        maximum = max(vector)
        return [(val - minimum) / (maximum - minimum) + 1e-6 for val in vector]

    def calculate_importance(self, source, target):
        ip_list = []
        before_list = []
        after_list = []
        for i in tqdm(range(len(source) // self.bsz + 1)):
            score = self.calculate_ot_importance(source[self.bsz * i:min(self.bsz * (i + 1), len(source))],
                                                 random.sample(list(target), 50))
            for k in score:
                ip_list.append(k)
        score_softmax = self.normalization(ip_list)
        return score_softmax

    def sample(self, G, node, size):
        dictt = {}
        node_list = set()
        node_list.add(node)
        parent_node = node_list.pop()
        dictt[parent_node] = parent_node
        degree_p = G.degree(parent_node)
        related_list = list(G.neighbors(parent_node))
        node_list.update(related_list)

        history = 0
        count = 0
        while (len(self.G1.nodes()) < size):
            value = len(self.G1.nodes()) / size
            if value == history:
                count = count + 1
            else:
                count = 0
                history = value
            if count >= 100:
                break
            if (len(node_list) > 0):
                child_node = node_list.pop()
                p = round(random.uniform(0, 1), 4)
                if (child_node not in dictt):
                    related_listt = list(G.neighbors(child_node))
                    degree_c = G.degree(child_node)
                    dictt[child_node] = child_node
                    if (p <= min(1, self.importance[child_node] * degree_p / (
                            degree_c * self.importance[parent_node])) and child_node in list(G.neighbors(parent_node))):
                        self.G1.add_edge(parent_node, child_node)
                        parent_node = child_node
                        degree_p = degree_c
                        node_list.clear()
                        node_list.update(related_listt)
                    else:
                        del dictt[child_node]
            else:
                node_list.update(random.sample(set(G.nodes()) - set(self.G1.nodes()), 3))
                parent_node = node_list.pop()
                G.add_node(parent_node)
                related_list = list(G.neighbors(parent_node))
                node_list.clear()
                node_list.update(related_list)
        return self.G1


class RWRW():
    def __init__(self):
        self.T = 50  # number of iterations
        self.growth_size = 2

    def random_walk_sampling_simple(self, complete_graph, pos_node, step_each_pos_node_to_walk, num_epoches=1):
        nr_nodes = len(complete_graph.nodes())
        edges_before_t_iter = 0
        sampled_graph = nx.Graph()

        for ep in range(num_epoches):
            for index_of_first_random_node in pos_node:
                sampled_graph.add_node(index_of_first_random_node)
                curr_node = index_of_first_random_node
                for iteration in range(step_each_pos_node_to_walk):
                    edges = [n for n in complete_graph.neighbors(curr_node)]
                    index_of_edge = random.randint(0, len(edges) - 1)
                    chosen_node = edges[index_of_edge]
                    sampled_graph.add_node(chosen_node)
                    sampled_graph.add_edge(curr_node, chosen_node)
                    curr_node = chosen_node

                    if iteration % self.T == self.T - 1:
                        if ((sampled_graph.number_of_edges() - edges_before_t_iter) < self.growth_size):
                            curr_node = random.randint(0, nr_nodes - 1)
                        edges_before_t_iter = sampled_graph.number_of_edges()

        return sampled_graph


class MHRW_aug():
    def __init__(self):
        self.G1 = nx.Graph()

    def mhrw(self, G, node, size):
        dictt = {}
        node_list = set()
        node_list.add(node)
        parent_node = node_list.pop()
        dictt[parent_node] = parent_node
        degree_p = G.degree(parent_node)
        related_list = list(G.neighbors(parent_node))
        node_list.update(related_list)

        history = 0
        count = 0
        while (len(self.G1.nodes()) < size):
            value = len(self.G1.nodes()) / size
            if value == history:
                count = count + 1
            else:
                count = 0
                history = value
            if count >= 100:
                break
            if (len(node_list) > 0):
                child_node = node_list.pop()
                p = round(random.uniform(0, 1), 4)
                if (child_node not in dictt):
                    related_listt = list(G.neighbors(child_node))
                    degree_c = G.degree(child_node)
                    dictt[child_node] = child_node
                    if (p <= min(1, degree_p / degree_c) and child_node in list(G.neighbors(parent_node))):
                        self.G1.add_edge(parent_node, child_node)
                        parent_node = child_node
                        degree_p = degree_c
                        node_list.clear()
                        node_list.update(related_listt)
                    else:
                        del dictt[child_node]
            # node_list set becomes empty or size is not reached
            # insert some random nodes into the set for next processing
            else:
                node_list.update(random.sample(set(G.nodes()) - set(self.G1.nodes()), 3))
                parent_node = node_list.pop()
                G.add_node(parent_node)
                related_list = list(G.neighbors(parent_node))
                node_list.clear()
                node_list.update(related_list)

        p = self.G1.nodes
        for node in p:
            related_list = list(G.neighbors(node))
            for k in related_list:
                if k in p:
                    self.G1.add_edge(node, k)
                    self.G1.add_edge(k, node)
        return self.G1


def __get_ppdai_sampled_nodes(user_info, payment, threshold=3):
    ret = set()
    with open(user_info, 'r') as f:
        cnt = 0
        for line in f:
            if cnt == 0:
                cnt += 1
                continue
            line = line.rstrip('\n').split(',')
            uid, agency = line[0], int(line[-1])
            if agency == 1:
                ret.add(uid)
        f.close()
    print('agency: {}'.format(len(ret)))
    cnt = 0
    with open(payment, 'r') as f:
        for line in f:
            line = line.rstrip('\n').split(' ')
            uid, due = line[0], int(line[1])
            if due == 1:
                if cnt % threshold == 0:
                    ret.add(uid)
                cnt += 1
        f.close()
    print('due: {}'.format(cnt))
    print('sampled id: {}'.format(len(ret)))
    return ret


def sample(fpath, sampled_id=None, lcc=True):
    node_dict, index, edge_list = {}, 0, []
    with open(fpath, 'r') as f:
        for line in f:
            line = line.rstrip('\n').split(' ')
            src, dst = line[0], line[1]
            if src not in node_dict:
                node_dict[src] = index
                index += 1
            if dst not in node_dict:
                node_dict[dst] = index
                index += 1
            src_idx, dst_idx = node_dict[src], node_dict[dst]
            edge_list.append([src_idx, dst_idx])
        f.close()
    g = nx.Graph()
    g.add_nodes_from(list(range(index)))
    g.add_edges_from(edge_list)

    idx2node = {val: key for key, val in node_dict.items()}
    print("# of userid:", len(idx2node))

    # g = max(nx.connected_component_subgraphs(g), key=len)
    g_nodes = g.nodes()
    num_nodes = len(g_nodes)
    num_edges = len(g.edges())
    print("Number of nodes=", num_nodes)
    print("Number of edges=", num_edges)
    average_degree = sum(dict(g.degree()).values()) / num_nodes
    print("Average degree=", average_degree)

    # read positive-labeled node set
    # pos_node = random.sample(range(num_nodes), 1000)
    if sampled_id is None:
        pos_node = random.sample(g_nodes, 2500)
    else:
        pos_node = []
        for ids in sampled_id:
            if ids in node_dict:
                pos_node.append(node_dict[ids])
        pos_node = np.array(pos_node)
    print('sampled {} nodes'.format(len(pos_node)))
    step_each_pos_node_to_walk = 10

    print("Start Sampling Graph")
    start = time.time()
    # make an object and call function SRW
    obj = SRW_RWF_ISRW()
    sampled_graph = obj.random_walk_sampling_simple(g, pos_node, step_each_pos_node_to_walk)
    if lcc:
        sampled_graph = max(nx.connected_component_subgraphs(sampled_graph), key=len)

    edges_sampled = sampled_graph.edges()

    # features = [user_feat[user_feat['id'] == int2userid[i]].values.tolist()[0][1:]
    #             for i in range(len(userid))]

    # with open("result/ppd_new.edgelist.sample", 'w') as f:
    output_fpath = "data/ppdai_network_data.sample" if lcc else "data/ppdai_network_data_full.sample"
    with open(output_fpath, 'w') as f:
        for line in edges_sampled:
            f.write(idx2node[line[0]] + '\t' + idx2node[line[1]] + '\n')

    print("Finish Sampling Graph")
    print(time.time() - start)
    print("Number of nodes sampled=", len(sampled_graph.nodes()))
    print("Number of edges sampled=", len(sampled_graph.edges()))
    average_degree = sum(dict(sampled_graph.degree()).values()) / len(sampled_graph.nodes())
    print("Average degree sampled=", average_degree)


if __name__ == "__main__":
    sampled_id = __get_ppdai_sampled_nodes(user_info='data/ppdai_user_info.csv',
                                           payment='data/ppdai_payment_labels.sample')
    sample(fpath='data/ppdai_network_data.raw', sampled_id=sampled_id)
    sampled_id = __get_ppdai_sampled_nodes(user_info='data/ppdai_user_info.csv',
                                           payment='data/ppdai_payment_labels.sample', threshold=8)
    sample(fpath='data/ppdai_network_data.raw', sampled_id=sampled_id, lcc=False)
