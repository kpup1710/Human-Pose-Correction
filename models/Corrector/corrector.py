import torch
import torch.nn as nn
from .vit import Transformer
from ..modulated_gcn_conv import ModulatedGraphConv
from torch.nn import Parameter

class Corrector(nn.Module):
    def __init__(self, adj, num_tokens, dim_enc, depth, heads, dim_head, mlp_dim, num_labels, dropout=0.0, out_dim=2) -> None:
        super().__init__()
        self.adj = adj
        self.dim_enc = dim_enc
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.transfromer = Transformer(self.dim_enc, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout)
        self.label_emb = nn.Linear(num_labels, self.dim_enc)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens + 1, dim_enc))
        self.mlp = ModulatedGraphConv(in_features=self.dim_enc, out_features=out_dim, adj=self.adj)

    def forward(self, label, skt):
        embed_lb = self.label_emb(label)
        # print(embed_lb.shape, skt.shape)
        x = torch.concat((embed_lb, skt), dim=1)
        x += self.pos_embedding
        x = self.transfromer(x)
        out = self.mlp(x[:,1:,:])
        return out

        
