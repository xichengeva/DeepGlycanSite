from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from .utils import (
    NeighborEmbedding,
    EdgeEmbedding,
    CosineCutoff,
    Distance,
    Sphere,
    VecLayerNorm,
    IntEmbedding,
    rbf_class_mapping,
    act_class_mapping,
)

EPS = 1e-12

class ViSNetBlock(nn.Module):

    def __init__(
        self,
        hidden_channels=128,
        num_layers=6,
        num_rbf=50,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        attn_activation="silu",
        neighbor_embedding=True,
        num_heads=8,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        lmax=1,
        vecnorm_type="max_min",
        vecnorm_trainable=True,
        x_dimension=41,
        edge_dimension=3,
        dropout=0.0,
    ):
        super(ViSNetBlock, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.lmax = lmax
        self.vecnorm_type = vecnorm_type
        self.vecnorm_trainable = vecnorm_trainable
        self.x_dimension = x_dimension
        self.edge_dimension = edge_dimension

        act_class = act_class_mapping[activation]

        self.distance = Distance(
            cutoff_lower,
            cutoff_upper,
            return_vecs=True,
        )

        self.sphere = Sphere(l=self.lmax)
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )

        self.rbf_proj = nn.Linear(num_rbf, hidden_channels)

        if self.x_dimension > 0:
            self.atom_embedding = IntEmbedding(self.x_dimension, hidden_channels, usage='x')
        else:
            raise ValueError('atom_feature must be specified')

        if self.edge_dimension > 0:
            self.bond_embedding = IntEmbedding(self.edge_dimension, hidden_channels, usage='edge')
        else:
            self.bond_embedding = None
        
        self.neighbor_embedding = NeighborEmbedding(hidden_channels, cutoff_lower, cutoff_upper, self.x_dimension).jittable() if neighbor_embedding else None

        self.edge_embedding = EdgeEmbedding().jittable()

        self.attention_layers = nn.ModuleList()
        block_params = dict(hidden_channels=hidden_channels, distance_influence=distance_influence,
                            num_heads=num_heads, activation=act_class, attn_activation=attn_activation,
                            cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper,
                            vecnorm_trainable=vecnorm_trainable, vecnorm_type=vecnorm_type,
                            dropout=dropout)
        
        for _ in range(num_layers - 1):
            layer = EquivariantMultiHeadAttention(**block_params,last_layer=False).jittable()
            self.attention_layers.append(layer)
        self.attention_layers.append(EquivariantMultiHeadAttention(**block_params,last_layer=True).jittable())

        self.x_out_norm = nn.LayerNorm(hidden_channels)
        self.v_out_norm = VecLayerNorm(hidden_channels, vecnorm_trainable, vecnorm_type)

        self.reset_parameters()

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        if self.bond_embedding is not None:
            self.bond_embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.x_out_norm.reset_parameters()
        self.v_out_norm.reset_parameters()

    def forward(self,
                data: Data,
                **kwargs
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        x = self.atom_embedding(data)
        edge_index, edge_weight, edge_vec = self.distance(data)  

        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"



        edge_attr = self.rbf_proj(self.distance_expansion(edge_weight))


        if self.bond_embedding is not None:
            edge_attr += self.bond_embedding(data)  

        edge_vec = edge_vec / torch.norm(edge_vec, dim=1).unsqueeze(1).clamp(min=1e-8)  
        edge_vec = self.sphere(edge_vec)

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(data, x, edge_weight, edge_attr)   

        vec = torch.zeros(x.size(0), ((self.lmax + 1) ** 2) - 1, x.size(1), device=x.device)
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)  
        
        for attn in self.attention_layers[:-1]:
            dx, dvec, dedge_attr = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec)  

            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.attention_layers[-1](x, vec, edge_index, edge_weight, edge_attr, edge_vec)
        x = x + dx
        vec = vec + dvec
        
        x = self.x_out_norm(x)
        vec = self.v_out_norm(vec)
        
        return x, vec, data.x, data.pos, data.batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper})"
        )


class EquivariantMultiHeadAttention(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        distance_influence,
        num_heads,
        activation,
        attn_activation,
        cutoff_lower,
        cutoff_upper,
        vecnorm_type,
        vecnorm_trainable,
        last_layer=False,
        dropout=0.0,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer
        self.vecnorm_type = vecnorm_type
        self.vecnorm_trainable = vecnorm_trainable

        self.x_layernorm = nn.LayerNorm(hidden_channels)  # 归一化x
        self.f_layernorm = nn.LayerNorm(hidden_channels)
        self.v_layernorm = VecLayerNorm(hidden_channels, vecnorm_trainable, vecnorm_type)
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)

        self.v_dot_proj = nn.Linear(hidden_channels, hidden_channels)
        
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels * 2)
            self.src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.w_dot_proj = nn.Linear(hidden_channels, hidden_channels)

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(hidden_channels, hidden_channels)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(hidden_channels, hidden_channels)
            
        self.scalar_dropout = nn.Dropout(dropout)
        self.vector_dropout = nn.Dropout2d(dropout)

        self.reset_parameters()
        
    def vector_rejection(self, vec, d_ij):
        # print(vec.shape,'vec shape')
        # print(d_ij.shape,'d_ij shape')

        
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        # print(vec_proj.shape,'vec_proj shape')
        # vec 78, 3, 256, d_ij,78, 3, vec_proj 78, 1, 256
        # 先用边向量乘vec（节点的vector属性，过了神经网络），
        # 然后求和，然后再乘边向量，然后再用节点的vector属性减去这个值
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        self.x_layernorm.reset_parameters()
        self.f_layernorm.reset_parameters()
        self.v_layernorm.reset_parameters()

        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.s_proj.weight)
        self.s_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_dot_proj.weight)
        self.v_dot_proj.bias.data.fill_(0)
        
        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.src_proj.weight)
            nn.init.xavier_uniform_(self.trg_proj.weight)
            nn.init.xavier_uniform_(self.w_dot_proj.weight)
            self.w_dot_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        #x, vec（点的向量表征）, edge_index, edge_weight（r_ij距离）, 
        # edge_attr（f_ij边属性）, edge_vec（d_ij边向量）
        # print()
        x = self.x_layernorm(x)
        f_ij = self.f_layernorm(f_ij)
        vec = self.v_layernorm(vec)
        # print("vec",vec.shape) 
        # vec不是edge_vec，最开始都是0，shape是x.size(0), 
        # ((self.lmax + 1) ** 2) - 1, hidden_channels,lmax=1时是3，lmax=2时是8
        
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim) 
        # 对x进行投影，然后reshape成num_heads个head_dim维的向量

        # print("q",q.shape) 
        # print("k",k.shape)
        # print("v",v.shape)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        # print(self.vec_proj(vec).shape) #把256通过linear变成768，然后分成3份，每份256，
        # 没有bias，意味着nn中只有乘法没有加法
        # print("vec1",vec1.shape) #vec1 shape是x.size(0), ((self.lmax + 1) ** 2) - 1, hidden_channels
        # print("vec2",vec2.shape) #vec2 shape是x.size(0), ((self.lmax + 1) ** 2) - 1, hidden_channels
        # print("vec3",vec3.shape) #vec3 shape是x.size(0), ((self.lmax + 1) ** 2) - 1, hidden_channels
        vec_dot = (vec1 * vec2).sum(dim=1)  #每个位置的元素相乘，然后在第一维上求和
        # print(vec1,vec2,vec1 * vec2,vec_dot)
        # print("vec_dot",vec_dot.shape) 38*256 if x.size(0)=38 and hidden_channels=256
        vec_dot = self.act(self.v_dot_proj(vec_dot))  # 过一个nn和激活函数

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)        #边属性投影1，转为和x一样的shape
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)          # 边属性投影2，转为和x一样的shape
            if self.dv_proj is not None
            else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec_out = self.propagate(  # 在propagate中调用message和update，并不显式编辑propgate函数，
        # 运行时，qkv分别是每一条边的起点，终点的qkv，最终使用message和edge_update达到更新节点和边的效果
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        # edge_updater_type: (vec: Tensor, d_ij: Tensor, f_ij: Tensor)
        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)  
        # output投影，分成3份，其中2份用来更新dx，1份用来更新dvec，dvec同时与vec_out也有关
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):  #i是源节点，j是目标节点
        # attention mechanism
        # 假如有36个节点，78条边，那么此处q_i的shape是78, 32， 8，和节点数无关，32是32个注意力头，8是8个维度
        # print("q_i",q_i.shape) #q_i shape是num_heads, head_dim
        # print('dk',dk.shape) #dk 来自于边属性的projection
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk).sum(dim=-1)  
         # 注意力系数，q_i和k_j是源节点和目标节点的特征，dk是边属性投影1

        # attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)  
        # 根据距离r_ij加权处理激活过后的attn

        # value pathway
        if dv is not None:
            v_j = v_j * dv   #目标节点特征乘以边属性投影2
        # print("v_j",v_j.shape) #v_j shape是num_heads, head_dim

        # update scalar features
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)          
        # 用目标节点特征乘以attn，然后reshape成hidden_channels维的向量，
        #  use attention to update target node features
        #v_j是目标节点的value特征
        # print(v_j.shape, vec_j.shape, d_ij.shape,'v_j, vec_j, d_ij')

        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels, dim=1)  
        # 把v_j通过linear变成2份，每份hidden_channels，没有bias，意味着nn中只有乘法没有加法
        
        # update vector features
        # vec_j是目标节点的节点vec，d_ij是边向量。

        # 此处使用目标节点value特征的更新（通过s_proj变成两份），加权平均产生新的vec
        # print(s1.shape, s2.shape, d_ij.shape, vec_j.shape,'s1, s2, d_ij, vec_j')
        # print(s1.unsqueeze(1).shape, s2.unsqueeze(1).shape, d_ij.unsqueeze(2).shape,'s1.unsqueeze(1), s2.unsqueeze(1), d_ij.unsqueeze(2)')
        # vec_j * s1.unsqueeze(1) ,如果vec_j是78, 3, 256，s1是78, 256，那么s1.unsqueeze(1)就是78, 1, 256，相乘就是78, 3, 256，也就是对于每一个s1的256，都去乘vec_j里的256，乘3遍（78*3*256的3），得到的3*256个数
        # s2.unsqueeze(1) * d_ij.unsqueeze(2),如果d_ij是38*3， s2是38*256，那么d_ij.unsqueeze(2)就是38*3*1，然后s2.unsqueeze(1)就是38*1*256，相乘就是38*3*256，3行的每个元素都去乘256个元素之一，然后连接起来，做38次，得到38*3*256

        vec = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

        # 总之，vec的更新方式是原来的vec乘以s1，加上d_ij乘以s2，s1和s2是通过v_j的更新得到的
        
        return v_j, vec
    
    def edge_update(self, vec_i, vec_j, d_ij, f_ij):

        w1 = self.vector_rejection(self.trg_proj(vec_i), d_ij)   # 把源节点的vec投影到边向量上，得到w1，
        # 边向量有方向，因此源节点出发是d_ij，目标节点出发是-d_ij
        w2 = self.vector_rejection(self.src_proj(vec_j), -d_ij) # 把目标节点的vec投影到边向量上，得到w2
        w_dot = (w1 * w2).sum(dim=1)  # 把w1和w2的每个元素相乘，然后求和，得到w_dot
        w_dot = self.act(self.w_dot_proj(w_dot))  # 把w_dot过一个nn

        # vector_rejection的内容：
        # def vector_rejection(self, vec, d_ij):
        # vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        # # print(vec_proj.shape,'vec_proj shape')
        # # vec 78, 3, 256, d_ij,78, 3, vec_proj 78, 1, 256
        # # 先用边向量乘vec（节点的vector属性，过了神经网络），
        # # 然后求和，然后再乘边向量，然后再用节点的vector属性减去这个值
        # return vec - vec_proj * d_ij.unsqueeze(2)


        f1, f2 = torch.split(
            self.act(self.f_proj(f_ij)),
            self.hidden_channels,
            dim=1
        )
        
        return f1 * w_dot + f2  #对边属性做残差操作，加上源节点和目标节点vec的投影


    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs
