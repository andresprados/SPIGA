from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F

from spiga.models.gnn.layers import MLP


class GAT(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads=4):
        super().__init__()

        num_heads_in = num_heads
        self.reshape = None
        if input_dim != output_dim:
            for num_heads_in in range(num_heads, 0, -1):
                if input_dim % num_heads_in == 0:
                    break
            self.reshape = MLP([input_dim, output_dim])

        self.attention = MessagePassing(input_dim, num_heads_in, out_dim=output_dim)

    def forward(self, features):
        message, prob = self.attention(features)
        if self.reshape:
            features = self.reshape(features)
        output = features + message
        return output, prob


class MessagePassing(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, out_dim=None):
        super().__init__()
        self.attn = Attention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, out_dim])

    def forward(self, features):
        message, prob = self.attn(features, features, features)
        return self.mlp(torch.cat([features, message], dim=1)), prob


class Attention(nn.Module):
    def __init__(self, num_heads: int, feature_dim: int):
        super().__init__()
        assert feature_dim % num_heads == 0
        self.dim = feature_dim // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = self.attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1)), prob

    def attention(self, query, key, value):
        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
        prob = F.softmax(scores, dim=-1)
        return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob
