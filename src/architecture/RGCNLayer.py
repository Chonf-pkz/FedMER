import torch
import torch.nn as nn
import torch.nn.functional as F

class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, dropout=0.1):
        super().__init__()
        self.num_rels = num_rels
        self.weight = nn.Parameter(torch.Tensor(num_rels, in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        self.dropout = nn.Dropout(dropout)
        self.self_loop = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_list):
        out = torch.zeros_like(self.self_loop(x))
        for rel, adj in enumerate(adj_list):
            out += torch.bmm(adj, torch.matmul(x, self.weight[rel]))
        out = out / self.num_rels
        out = out + self.self_loop(x)
        return F.relu(self.dropout(out))
