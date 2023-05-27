import torch
import torch.nn as nn
from.Predictor import ModulatedGCN
from.Corrector import Corrector

class Predictor_Corrector(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 128), num_layers_pred=4, nodes_group=None, p_dropout=None, num_classes=82,
                 num_tokens=18, depth=4, heads=1, dim_head=64) -> None:
        super().__init__()
        self.adj = adj
        self.hid_dim = hid_dim
        self.coords_dim = coords_dim
        self.num_lalyers_pred = num_layers_pred
        self.nodes_group = nodes_group
        self.p_dropout = p_dropout
        self.num_classes = num_classes
        self.num_tokens = num_tokens
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head

        self.predictor = ModulatedGCN(adj=self.adj, hid_dim=self.hid_dim, coords_dim=self.coords_dim, num_layers=self.num_lalyers_pred, nodes_group=self.nodes_group, p_dropout=self.p_dropout, num_classes=self.num_classes).double()
        self.predictor.load_state_dict(torch.load('/content/drive/MyDrive/test/best_predictor.pth'))
        for param in self.predictor.parameters():
          param.requires_grad = False
        self.corrector = Corrector(adj = self.adj, num_tokens=self.num_tokens, dim_enc=self.coords_dim[1], depth=self.depth, heads=self.heads, dim_head=self.dim_head, mlp_dim=self.coords_dim[0], num_labels=self.num_classes)

    def forward(self, input):
        x, y = input
        out_pred, tokens = self.predictor(x.type(torch.float64))
        corrected_pose = self.corrector(y.type(torch.float64), tokens.type(torch.float64))
        out_corr,_ = self.predictor(corrected_pose.type(torch.float64))
        return out_corr, corrected_pose
