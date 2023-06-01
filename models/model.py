import torch
import torch.nn as nn
from.Predictor import ModulatedGCN
from.Corrector import Corrector

class Predictor_Corrector(nn.Module):
    def __init__(self, adj, args) -> None:
        super().__init__()
        self.adj = adj
        self.hid_dim = args['pred_hid_dim']
        self.coords_dim = args['pred_coords_dim']
        self.num_layers_pred = args['pred_num_layers']
        self.nodes_group = args['pred_nodes_group']
        self.p_dropout = args['p_dropout']
        self.dim_enc = args['cor_dim_enc']
        self.depth = args['cor_depth']
        self.heads = args['cor_heads']
        self.dim_head = args['cor_dim_head']
        self.out_dim = args['out_dim']
        self.dropout = args['cor_dropout']
        self.out_dim = args['cor_out_dim']
        self.num_tokens = args['num_tokens']
        self.num_classes = args['num_classes']

        self.predictor = ModulatedGCN(adj=self.adj, hid_dim=self.hid_dim, coords_dim=self.coords_dim, num_layers=self.num_layers_pred, nodes_group=self.nodes_group, p_dropout=self.p_dropout, num_classes=self.num_classes).double()
        
        self.corrector = Corrector(adj = self.adj, num_tokens=self.num_tokens, dim_enc=self.coords_dim[1], depth=self.depth, heads=self.heads, dim_head=self.dim_head, mlp_dim=self.coords_dim[0], num_labels=self.num_classes, out_dim=self.out_dim)

    def load_predictor(self, path):
        self.predictor.load_state_dict(torch.load(path))
        for param in self.predictor.parameters():
          param.requires_grad = False

    def forward(self, input):
        x, y = input
        out_pred, tokens = self.predictor(x.type(torch.float64))
        corrected_pose = self.corrector(y.type(torch.float64), tokens.type(torch.float64))
        out_corr,_ = self.predictor(corrected_pose.type(torch.float64))
        return out_corr, corrected_pose
