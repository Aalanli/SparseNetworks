# %%
import copy
from typing import Dict, List, Tuple, Union
from cv2 import sqrt

import torch
import torch.nn as nn


def soft_gate(x, a, b):
    return 1 / (1 + (x / a).abs() ** (2 * b))

def collect_decisions(a, b):
    for k in b:
        if k not in a:
            a[k] = b[k]
        else:
            a[k][0].extend(b[k][0])
            a[k][1].extend(b[k][1])
    return a

def reset_parameters(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()

def dense_conv(dim, out_dim, kernel_size=7):
    return nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, out_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(out_dim))

def replicate_layer(layer, n):
    return nn.ModuleList([copy.deepcopy(layer).apply(reset_parameters) for i in range(n)])

def print_types(x):
    if type(x) == list:
        h = "["
        for i in x:
            h += print_types(i) + ", "
        return h + "]"
    elif type(x) == dict:
        h = "["
        for i in x:
            h += print_types(x[i]) + ", "
        return h + "]"
    else:
        return str(type(x))
        

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Comp(nn.Module):
    def __init__(self, dim, kernel_size) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, padding='same')
    
    def reset_parameters(self):
        self.conv.reset_parameters()
    
    def forward(self, x):
        return self.conv(x).sum(0) + x.sum(0)
        

class DecisionNode(nn.Module):
    """output decision matrix"""
    def __init__(self, sample_dims, n_nodes, kernel_size, dilation) -> None:
        super().__init__()
        self.sample_dims = sample_dims
        self.conv = nn.Conv2d(sample_dims, n_nodes, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(n_nodes, n_nodes, kernel_size)
    
    def forward(self, x):
        # x.shape = [batch, d_model, y, x]
        x = self.conv(x[:, :self.sample_dims])  # [batch, n_nodes, y', x']
        x = self.conv2(x.flatten(2)).sum(-1)  # [batch, n_nodes]
        return x


class DecisionGate(nn.Module):
    """splits given tensor into different nodes given by weightings"""
    def __init__(self, init_a, init_b, thres) -> None:
        super().__init__()
        self.a = init_a
        self.b = init_b
        self.thres = thres
    
    def forward(self, x, act, batch_inds) -> Tuple[torch.Tensor, Dict[int, Tuple[List[torch.Tensor], List[int]]]]:
        x = soft_gate(x, self.a, self.b) # [batch, out_inds]
        inds = (x >= self.thres).nonzero()
        out = {}
        for i in inds:
            p = int(i[1])
            b = int(i[0])
            if p not in out:
                out[p] = ([act[i[0]] * x[i[0], i[1]]], [batch_inds[b]])
            else:
                out[p][0].append(act[i[0]] * x[i[0], i[1]])
                out[p][1].append(batch_inds[b])
        return x, out


class CoalsceDecison(nn.Module):
    """collect tensors from previous nodes and processes them for the current node"""
    def __init__(self, nodes: Union[int, List[int]], compression) -> None:
        super().__init__()
        if type(nodes) == int:
            self.nodes = [nodes]
        else:
            self.nodes = nodes
        self.compress = compression
    
    def forward(self, x) -> Tuple[torch.Tensor, List[int]]:
        x1 = []
        batch_inds = []
        for i in self.nodes:
            x1.extend(x[i][0])
            batch_inds.extend(x[i][1])
        batches = {i: [] for i in batch_inds}
        for batch, act in zip(batch_inds, x1):
            batches[batch].append(act)
        results = []
        indices = list(batches.keys())
        indices.sort()
        for b in indices:
            h0 = torch.stack(batches[b], dim=0)
            h1 = self.compress(h0)
            results.append(h1)
        h = torch.stack(results, dim=0)
        return h, indices


class SparseConvLayer(nn.Module):
    def __init__(self, nodes, nodes_out, conv, dec_gate, comp, sample_dims, kernel_size, dilation) -> None:
        super().__init__()
        self.nodes = nodes
        self.dec_nodes = replicate_layer(DecisionNode(sample_dims, nodes_out, kernel_size, dilation), nodes)
        self.conv = replicate_layer(conv, nodes)
        self.col = nn.ModuleList([CoalsceDecison(i, copy.deepcopy(comp)) for i in range(nodes)]).apply(reset_parameters)
        self.dec_gate = dec_gate
        
        self.step_count = 0
        self.act_tracker = {n: 0 for n in range(nodes)}
    
    def execute_node(self, x, i):
        out, ind = self.col[i](x)
        out = self.conv[i](out)
        h = self.dec_nodes[i](out)
        return self.dec_gate(h, out, ind)
    
    def forward(self, x: Dict[int, Tuple[List[torch.Tensor], List[int]]]):
        node_acts = {}
        dec = {}
        print(self.nodes)
        print(x.keys())
        for i in range(self.nodes):
            if len(x[i][0]) > 0:
                n_act, dec1 = self.execute_node(x, i)
                node_acts[i] = n_act
                collect_decisions(dec, dec1)
        for i in dec:
            self.act_tracker[i] += len(dec[i][1])
        self.step_count += 1
        return dec, node_acts

# TODO pruning unused nodes

class SparseConvNet(nn.Module):
    def __init__(self, split_patch, patch_size, dim, conv, dec_gate, comp, layers, sample_dims, kernel_size, dilation) -> None:
        super().__init__()
        split_nodes = split_patch ** 2
        self.split_patch = split_patch
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim))
        self.node_layers = nn.ModuleList([
            SparseConvLayer(split_nodes, split_nodes, conv, dec_gate,
                            comp, sample_dims, kernel_size, dilation) for i in range(layers)])
        self.comp = comp
    
    def forward(self, x):
        x = self.patch_embed(x)
        patches = map(lambda x: list(torch.chunk(x, self.split_patch, -2)), list(torch.chunk(x, self.split_patch, -1)))
        x0 = {i: (h, list(range(x.shape[0]))) for i, h in enumerate(patches)}
        decisions = []
        for f in self.node_layers:
            x0, node_dec = f(x0)
            decisions.append(node_dec)
        by_batch = [[] for i in range(x.shape[0])]
        for k in x0:
            for (h, b) in zip(x0[k][0], x0[k][1]):
                by_batch[b].append(h)
        for i in by_batch:
            h = torch.stack(by_batch[i], dim=0)
            h = self.comp(h).sum(0) + h.sum(0)
            by_batch[i] = h
        out = torch.stack(h, dim=0)
        return out, decisions


"""conv = dense_conv(128)
dec_gate = DecisionGate(1, 1, 0.001)
comp = nn.Conv2d(128, 128, 3, padding='same')

model = SparseConvNet(3, 7, 128, conv, dec_gate, comp, 6, 4, 7, 4)
x = torch.rand(4, 3, 512, 512)

y = model(x)
"""

# %%
