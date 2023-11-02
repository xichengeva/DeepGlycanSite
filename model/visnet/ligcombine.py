
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


    
class Combine(nn.Module):
    def __init__(self, ligand_model, target_model, d_model=256, nhead = 8, num_encoder_layers=3,  num_decoder_layers=3, dropout_rate=0.15):
        super(Combine, self).__init__()
        
        self.ligand_model = ligand_model
        self.target_model = target_model
        print(d_model, nhead, num_encoder_layers, num_decoder_layers, dropout_rate)

        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          dim_feedforward=d_model*2,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dropout=dropout_rate,
                                          )
  
        self.z_pi = nn.Linear(d_model, 1)
    
    def forward(self, data_ligand, data_target, y=None):
        # print(data_ligand,'data_ligand')
        # print(data_ligand,'data_ligand')
        # reshape according to batch
        # data_ligand.K_pre = data_ligand.K_pre.reshape(data_ligand.batch.max().item()+1, data_ligand.K_pre.size(0)//(data_ligand.batch.max().item()+1))
        data_ligand.U_pre = data_ligand.U_pre.reshape(data_ligand.batch.max().item()+1, data_ligand.U_pre.size(0)//(data_ligand.batch.max().item()+1)).to(torch.float32)
        # data_ligand.KU_pre = data_ligand.KU_pre.reshape(data_ligand.batch.max().item()+1, data_ligand.KU_pre.size(0)//(data_ligand.batch.max().item()+1)).to(torch.float32)
        # print(data_ligand,'data_ligand')        
        h_l, out_graph_vec = self.ligand_model(data_ligand)
        # print(h_l)
        # print(out_graph_vec.size(),'out_graph_vec size')
        # print(data_target.x.size(),'data_target.x size before')
        h_t_x1, t_mask1 = to_dense_batch(data_target.x, data_target.batch, fill_value=0)
        h_t_x1 = torch.transpose(h_t_x1, 1, 0)
        # print(h_t_x1.size(),'h_t_x1 size')
        out_graph_vec_expanded = out_graph_vec.unsqueeze(0).expand(h_t_x1.size(0), out_graph_vec.size(0), out_graph_vec.size(1))

        out_graph_vec_expanded = torch.transpose(out_graph_vec_expanded, 1, 0)
        out_graph_vec_expanded = out_graph_vec_expanded[t_mask1]
        h_t = self.target_model(data_target,out_graph_vec_expanded)

        h_l_x, l_mask = to_dense_batch(h_l.x, h_l.batch, fill_value=0)
        h_t_x, t_mask = to_dense_batch(h_t[0], h_t[1], fill_value=0)
        # print(h_l_x,h_t_x)

        h_l_x = torch.transpose(h_l_x, 1, 0)
        h_t_x = torch.transpose(h_t_x, 1, 0)
        C = self.transformer(h_l_x, h_t_x)
        pi = torch.transpose(C, 1, 0)
        pi_out = pi[t_mask]

        pi_out_1 = self.z_pi(pi_out)
        pi_out_1 = torch.sigmoid(pi_out_1)
        # print(h_l_x)

        
        return pi_out_1

