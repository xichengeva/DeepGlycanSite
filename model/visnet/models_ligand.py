import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from torch_geometric.nn import Set2Set


def compute_cluster_batch_index(cluster, batch):
    max_prev_batch = 0
    for i in range(batch.max().item()+1):
        cluster[batch == i] += max_prev_batch
        max_prev_batch = cluster[batch == i].max().item() + 1
    return cluster


class NodeSampling(nn.Module):
    def __init__(self, nodes_per_graph):
        super(NodeSampling, self).__init__()
        
        self.num = nodes_per_graph
          
    def forward(self, x):
        if self.training:
            max_prev_batch = 0
            idx = []
            counts = torch.unique(x.batch, return_counts=True)[1]
            
            for i in range(x.batch.max().item()+1):
                idx.append(torch.randperm(counts[i])[:self.num] + max_prev_batch)
                max_prev_batch += counts[i]
            idx = torch.cat(idx)
            
            x.batch = x.batch[idx]
            # x.pos = x.pos[idx]
            x.x = x.x[idx]
            
        return x

    
class ResBlock(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.15):
        super(ResBlock, self).__init__()
        
        self.projectDown_node = nn.Linear(in_channels, in_channels//4)
        self.projectDown_edge = nn.Linear(in_channels, in_channels//4)
        self.bn1_node = nn.BatchNorm1d(in_channels//4)
        self.bn1_edge = nn.BatchNorm1d(in_channels//4)
        
        self.conv = MetaLayer(edge_model=EdgeModel(in_channels//4), node_model=NodeModel(in_channels//4), global_model=None)
                
        self.projectUp_node = nn.Linear(in_channels//4, in_channels)
        self.projectUp_edge = nn.Linear(in_channels//4, in_channels)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2_node = nn.BatchNorm1d(in_channels)
        nn.init.zeros_(self.bn2_node.weight)
        self.bn2_edge = nn.BatchNorm1d(in_channels)
        nn.init.zeros_(self.bn2_edge.weight)
                
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        h_node = F.elu(self.bn1_node(self.projectDown_node(x)))
        h_edge = F.elu(self.bn1_edge(self.projectDown_edge(edge_attr)))
        h_node, h_edge, _ = self.conv(h_node, edge_index, h_edge, None, batch)
        
        h_node = self.dropout(self.bn2_node(self.projectUp_node(h_node)))
        data.x = F.elu(h_node + x)
        
        h_edge = self.dropout(self.bn2_edge(self.projectUp_edge(h_edge))) 
        data.edge_attr = F.elu(h_edge + edge_attr)
        
        return data


class EdgeModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(in_channels*3, in_channels), nn.BatchNorm1d(in_channels), nn.ELU())

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        # print(src.shape, dest.shape, edge_attr.shape,'src, dest, edge_attr')
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)

    
class NodeModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = nn.Sequential(nn.Linear(in_channels*2, in_channels), nn.BatchNorm1d(in_channels), nn.ELU())
        self.node_mlp_2 = nn.Sequential(nn.Linear(in_channels*2, in_channels), nn.BatchNorm1d(in_channels), nn.ELU())

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

                        
class LigandNet(nn.Module):
    def __init__(self, in_channels, edge_features=6, hidden_dim=128, residual_layers=20, dropout_rate=0.15):
        super(LigandNet, self).__init__()

        self.node_encoder = nn.Linear(in_channels, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        self.conv1 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        self.conv2 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        self.conv3 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        layers = [ResBlock(in_channels=hidden_dim, dropout_rate=dropout_rate) for i in range(residual_layers)] 
        self.resnet = nn.Sequential(*layers)
        self.pool = Set2Set(hidden_dim, processing_steps=3)
        self.pool1 = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.pool2 = nn.Sequential(
                    nn.Linear(hidden_dim*2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.pool3 = nn.Linear(hidden_dim*3, hidden_dim)
        self.kpool = nn.Linear(512, hidden_dim)


    def forward(self, data):
        # print(data)
        # print(data.x.shape)
        data.x = self.node_encoder(data.x)
        # print(data.x.shape)
        data.edge_attr = self.edge_encoder(data.edge_attr)
        data.x, data.edge_attr, _ = self.conv1(data.x, data.edge_index, data.edge_attr, None, data.batch)
        data.x, data.edge_attr, _ = self.conv2(data.x, data.edge_index, data.edge_attr, None, data.batch)
        data.x, data.edge_attr, _ = self.conv3(data.x, data.edge_index, data.edge_attr, None, data.batch)
        data = self.resnet(data)
        kout = self.kpool(data.U_pre)
        output_graph = self.pool(data.x, data.batch).squeeze(dim=-1)
        # kout add to output_graph
        # print(output_graph.shape,'output_graph shape')
        # print(kout.shape,'kout shape')
        out_add_kout = torch.cat((output_graph, kout), dim=1)
        # print(out_add_kout.shape,'out_add_kout shape')

        output_graph3 = self.pool3(out_add_kout)
        return data, output_graph3
    
    
def mdn_loss_fn(pi, y):
    # two class loss
    # y = 0 or 1
    return -torch.log(pi) * y
    diff = pi - y
    squared_diff = torch.square(diff)
    return squared_diff
