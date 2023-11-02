import math
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
import math

EPS = 1e-8

class VecLayerNorm(nn.Module):
    def __init__(self, hidden_channels, trainable, norm_type="max_min"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_channels), requires_grad=trainable)
        
        if norm_type == "rms":
            self.norm = self.rms_norm
        elif norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
    
    def none_norm(self, vec):
        return vec
        
    def rms_norm(self, vec):
        dist = torch.norm(vec, dim=1)
        
        if (dist == 0).all():
            return torch.zeros_like(vec)
        
        dist = dist.clamp(min=EPS)
        dist = torch.sqrt(torch.mean(dist ** 2, dim=-1) + EPS)   # 原始距离除以均方根
        return vec / dist.unsqueeze(-1).unsqueeze(-1)
    
    def max_min_norm(self, vec):
        dist = torch.norm(vec, dim=1, keepdim=True)
        
        if (dist == 0).all():
            return torch.zeros_like(vec)
        
        dist =  dist.clamp(min=EPS)
        direct = vec / dist   #方向向量，长度为1，方向不变
        
        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)
        
        return dist * direct   # 同方向，距离从0-1之间分布，原先最小距离变为0，最大距离变为1

    def forward(self, vec):
        
        if vec.shape[1] == 3:
            vec = self.norm(vec)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.shape[1] == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)
            vec1 = self.norm(vec1)
            vec2 = self.norm(vec2)
            vec = torch.cat([vec1, vec2], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        else:
            NotImplementedError()

class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, cutoff_lower, cutoff_upper, atom_feature):
        super(NeighborEmbedding, self).__init__(aggr="add")

        self.embedding = IntEmbedding(atom_feature, hidden_channels, usage="x")
        self.combine = nn.Linear(hidden_channels * 3, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.combine.weight)
        self.combine.bias.data.fill_(0)

    def forward(self, data, x, edge_weight, edge_attr):
        # remove self loops
        edge_index = data[f"edge_index"]
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = edge_attr * C.view(-1, 1)
        # print("C", C)

        # print("W", W.shape)
        # W 是根据距离（edge_weight）和edge_attr计算出来的权重

        x_neighbors = self.embedding(data)
        # print("x_neighbors", x_neighbors.shape)
        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None) # 按照W的权重，将x_neighbors的信息传递给x
        # size=None means that the size of the output tensor is automatically inferred from the input tensors.  
        #The propagate method is used to propagate messages between nodes in a graph. 
        # This is achieved by computing a message for each edge in the graph, based on the features of the nodes connected by that edge. 
        # The message is then aggregated over all incoming edges for each node, and combined with the node's own features to produce a new feature vector for that node.
        # The exact implementation of the propagate function can vary depending on the specific type of message passing being used.
        #  However, in general, the function computes messages using the message() method, aggregates the messages using the aggregate() method, and combines the aggregated messages with the node features using the update() method.
        # The propagate() function returns the new node features after message passing.
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1)) # 最后将neibours的信息和原来的x信息结合起来
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W
    
class EdgeEmbedding(MessagePassing):
    
    def __init__(self):
        super(EdgeEmbedding, self).__init__(aggr=None)
        
    def forward(self, edge_index, edge_attr, x):
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)   # 边嵌入，最终的边信息等于x_i+x_j的结果*edge_attr，注意这种写法
        return out
    
    def message(self, x_i, x_j, edge_attr):
        return (x_i + x_j) * edge_attr
    
    def aggregate(self, features, index):
        # no aggregate
        return features

class ExpNormalSmearing(nn.Module):
    '''This is a PyTorch module for implementing "exponentially modified normal smearing", which is a technique used in machine learning models for predicting molecular properties based on their 3D atomic coordinates. 
    Specifically, it is a type of radial basis function (RBF) that smears out the atomic positions to create a smooth, continuous representation of the molecular structure.

    The input to this function is a tensor of pairwise distances between atoms in a molecule (i.e. a distance matrix), and the output is a tensor of the same shape, 
    representing the smearing of these distances using an exponentially modified normal distribution. 
    The function uses a cosine cutoff to zero out distances beyond a certain threshold, and a set of learnable parameters (means and betas) to control the shape and width of the distribution.

    The key steps in the computation are as follows:

    First, the input tensor of distances is unsqueezed along the last dimension to make it compatible with broadcasting in later steps.

    The cosine cutoff function is applied to the distances using the CosineCutoff class.

    The unsmeared distances are exponentiated with a negative exponential, scaled by a factor alpha and shifted by cutoff_lower, to create a distribution with a long tail that decays to zero as the distance approaches the cutoff.

    This exponential distribution is then "modulated" by the learnable parameters means and betas using an elementwise multiplication and an exponentiation of the squared difference.

    The resulting tensor is a smoothed representation of the input distances that can be fed into downstream layers of a neural network for further processing.'''
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        # print(start_value,'start_value',math.e**(-5+0)), start_value就是e的-5次方
        means = torch.linspace(start_value, 1, self.num_rbf)  # means 是0-1之间的均匀分布
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        # print((2/64)**-2)  #这里一共64个beta，每个都一样
        # print(betas,'betas.shape')
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):  # 把距离变成径向分布函数个数的维度
        dist = dist.unsqueeze(-1)
        # print(dist.shape,'dist.shape')
        # print(self.means,'means.shape')
        # print((2/64)**-2,'2/64**-2')
        # print(dist,self.cutoff_fn(dist))
        # self.cutoff_fn(dist) 将距离归一化到0-1之间，之后再乘上e的
        # （-beta*(e的-dist次方-means)的平方）次方得到最后64维的输出
        # print((self.cutoff_fn(dist) * torch.exp( -self.betas  * (torch.exp((-dist)) - self.means) ** 2)).shape,'dist after operation')
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )

class CosineCutoff(nn.Module):
    '''The CosineCutoff class implements a cosine cutoff function for distance-dependent interactions. 
    Given a tensor of distances between particles, it returns a tensor of the same shape containing the cutoff values for each distance. 
    The cutoff values vary between 0 and 1 and are computed using a cosine function, with a lower and upper distance cutoff specified by the cutoff_lower and cutoff_upper arguments, respectively.
    The function first computes the cosine cutoff values using the following equation:

    cutoffs = 0.5 * (cos(pi * (2 * (distances - cutoff_lower) / (cutoff_upper - cutoff_lower) + 1.0)) + 1.0)

    or cutoffs = 0.5 * (cos(distances * pi / cutoff_upper) + 1.0)

    depending on whether cutoff_lower is greater than zero or not. The second equation is used if the lower cutoff is zero.



The resulting tensor of cutoff values is then multiplied elementwise with a binary tensor that is 1 where the corresponding distance value is within the specified cutoff range, 
and 0 otherwise. This removes any contributions to the interaction potential beyond the cutoff radius. Finally, the resulting tensor of cutoff values is returned.


    '''
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class Distance(nn.Module):
    def __init__(
        self,
        cutoff_lower,
        cutoff_upper,
        return_vecs=False,
    ):
        super(Distance, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.return_vecs = return_vecs

    def forward(self, data):
        
        edge_index, pos = data[f"edge_index"], data[f"pos"]

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        edge_vec = edge_vec.float()

        mask = edge_index[0] != edge_index[1]  # 非自循环的边
        # print(mask)
        edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
        # print(edge_weight.type())
        # change edge_weight to Double type
        # edge_weight = edge_weight.double()
        edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)  # weight是边的长度（欧式距离）
        # print(edge_weight,'edge_weight')
        # print(edge_vec,'edge_vec')

        lower_mask = edge_weight >= self.cutoff_lower  # 边长必须大于cutoff_lower，但是cutoff_upper是0，所以这里没有限制
        edge_index = edge_index[:, lower_mask]
        edge_weight = edge_weight[lower_mask]

        if self.return_vecs:
            edge_vec = edge_vec[lower_mask]
            return edge_index, edge_weight.clamp(1e-8), edge_vec

        return edge_index, edge_weight.clamp(1e-8), None
        
class Sphere(nn.Module):
    
    def __init__(self, l=2):
        
        super(Sphere, self).__init__()
        
        self.l = l
        
    def forward(self, edge_vec):
        
        # edge_vec = F.normalize(edge_vec, p=2, dim=-1)
        edge_sh = _spherical_harmonics(self.l, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        
        return edge_sh
        
# @torch.jit.script
def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    if lmax == 1:
        return torch.stack([
            sh_1_0, sh_1_1, sh_1_2
        ], dim=-1)

    sh_2_0 = math.sqrt(3.0) * x * z
    sh_2_1 = math.sqrt(3.0) * x * y
    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)
    sh_2_2 = y2 - 0.5 * x2z2
    sh_2_3 = math.sqrt(3.0) * y * z
    sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

    if lmax == 2:
        return torch.stack([
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4
        ], dim=-1)
        
class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        # print(x.shape,'x')
        # print(v.shape,'v')
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        # print(vec1.shape,'vec1')
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v
    
rbf_class_mapping = {"expnorm": ExpNormalSmearing}

act_class_mapping = {
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

class IntEmbedding(nn.Module):
    """
    Atom Encoder
    """
    def __init__(self, inp_dim, embed_dim, usage='atom'):
        super(IntEmbedding, self).__init__()

        self.usage = usage
        self.x_linear = nn.Linear(inp_dim, embed_dim)
        self.edge_linear = nn.Linear(inp_dim, embed_dim)

    def reset_parameters(self):
        """
        Reinitialize model parameters.
        """
        nn.init.xavier_uniform_(self.x_linear.weight)
        nn.init.constant_(self.x_linear.bias, 0)
        nn.init.xavier_uniform_(self.edge_linear.weight)
        nn.init.constant_(self.edge_linear.bias, 0)


    def forward(self, input):
        if self.usage == 'x':
            input.x = input.x.to(torch.float32)
            # print(input.x.dtype,'x,dtype')
            out_embed = self.x_linear(input.x)
        elif self.usage == 'edge':
            out_embed = self.edge_linear(input.edge_feats)
        return out_embed


