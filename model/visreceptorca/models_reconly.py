import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


class Combine(nn.Module):
    def __init__(self, target_model, d_model=256, nhead = 8, num_encoder_layers=3,  num_decoder_layers=3, dropout_rate=0.15):
        super(Combine, self).__init__()
        
        self.target_model = target_model
        # print(d_model, nhead, num_encoder_layers, num_decoder_layers, dropout_rate)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=d_model*2,
                                                   dropout=dropout_rate)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer,
                                                         num_layers=num_decoder_layers)
        self.z_pi = nn.Linear(d_model, 1)
    
    def forward(self, data_target, y=None):

        h_t = self.target_model(data_target)
        h_t_x, t_mask = to_dense_batch(h_t[0], h_t[1], fill_value=0)
        # print(h_l_x,h_t_x)
        h_t_x = torch.transpose(h_t_x, 1, 0)
        memory = torch.zeros_like(h_t_x)

        C = self.transformer_decoder(h_t_x, memory)
        C = C.squeeze()
        pi = self.z_pi(C)
        pi = torch.transpose(pi, 1, 0)
        pi_out = pi[t_mask]
        pi_out = torch.sigmoid(pi_out)
        
        return pi_out

    def compute_euclidean_distances_matrix(self, X, Y):
        # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        # (X-Y)^2 = X^2 + Y^2 -2XY
        X = X.double()
        Y = Y.double()
                # return the minimal distance of all X to per Y
        return torch.min(torch.sum(X**2, axis=-1).unsqueeze(1) + torch.sum(Y**2, axis=-1).unsqueeze(-1) - 2 * torch.bmm(X, Y.permute(0, 2, 1)), axis=0)[0]

        dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2,    axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
        return dists**0.5
