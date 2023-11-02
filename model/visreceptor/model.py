import re
from typing import Optional, Tuple
import torch
from torch import nn, Tensor
from torch_geometric.data import Data
from . import output_modules
import warnings
import numpy as np
from torch_geometric.utils import to_dense_batch
from .visnet_block import ViSNetBlock

def create_model(args):
    model_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        neighbor_embedding=args["neighbor_embedding"],
        cutoff_lower=args["cutoff_lower"],
        cutoff_upper=args["cutoff_upper"],
        attn_activation=args["attn_activation"],
        num_heads=args["num_heads"],
        distance_influence=args["distance_influence"],
        lmax=args['lmax'],
        vecnorm_type=args['vecnorm_type'],
        vecnorm_trainable=args['vecnorm_trainable'],
        x_dimension=args['x_dimension'],
        edge_dimension=args['edge_dimension'],
        dropout=args.get("dropout", 0.0),
    )
  
    if args["model"] == "ViSNetBlock":
        from .visnet_block import ViSNetBlock
        representation_model = ViSNetBlock(**model_args)
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')

    output_model = getattr(output_modules, "Equivariant" + args["output_model"])(args["embedding_dimension"], args["activation"])

    model = ViSNet(representation_model, output_model, reduce_op=args["reduce_op"])
    
    return model


def load_model(filepath, args=None, device="cpu", **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    model = create_model(args)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict, strict=False)
    return model.to(device)

def create_clip_model(student_model, teacher_model):
    
    return ViSNetCLIP(student_model=student_model, teacher_model=teacher_model)

def load_clip_model(student_filepath, teacher_filepath, args=None, device="cpu", **kwargs):
    teacher_ckpt = torch.load(teacher_filepath, map_location="cpu")
    teacher_args = teacher_ckpt["hyper_parameters"]
    
    student_ckpt = torch.load(student_filepath, map_location="cpu")
    student_args = student_ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    teacher_model = create_model(teacher_args)
    student_model = create_model(student_args)
    model = create_clip_model(teacher_model=teacher_model, student_model=student_model)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in student_ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    print("Freezing teacher model...")
    for param in teacher_model.parameters():
        param.requires_grad = False
    return model.to(device)

class ViSNet(nn.Module):
    def __init__(
        self,
        representation_model: ViSNetBlock,
        output_model: output_modules.EquivariantScalarKD,
        reduce_op="add"
    ):
        super(ViSNet, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model
        
        self.reduce_op = reduce_op
        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()

    def forward(self,
                data: Data,
                **kwargs) -> Tuple[Tensor, Optional[Tensor]]:


        # run the potentially wrapped representation model
        app_x = data.x
        x, v, z, pos, batch = self.representation_model(data)
        batch = torch.zeros_like(app_x) if data.batch is None else data.batch
        # apply the output network
        per_atom_scalar = self.output_model.pre_reduce(x, v, z, pos, batch)
        # print(per_atom_scalar.shape)
        # print(batch.shape)

        # aggregate atoms
        # out_scalar = scatter(per_atom_scalar, batch, dim=0, reduce=self.reduce_op)
        # print(out_scalar[:10,:10],'output before rotation')
        # print(data.pos[:10,:10],'position before rotation')
        # random_rotation_matrix = np.random.rand(3,3)
        # random_rotation_matrix = torch.from_numpy(random_rotation_matrix).double()

        # rotated_pos = data.pos @ random_rotation_matrix
        # per_atom_scalar = self.output_model.pre_reduce(x, v, z, rotated_pos, batch)
        # out_scalar_rotated = scatter(per_atom_scalar, batch, dim=0, reduce=self.reduce_op)
        # print(out_scalar_rotated[:10,:10],'output after rotation')
        # print(rotated_pos[:10,:10],'position after rotation')

        return per_atom_scalar, batch
    
class ViSNetCLIP(nn.Module):
    
    def __init__(
        self,
        teacher_model: ViSNet,
        student_model: ViSNet,    
    ) -> None:
        super().__init__()
        
        self.teacher_model = teacher_model
        self.student_model = student_model
        
        self.student_channels = self.student_model.representation_model.hidden_channels
        self.teacher_channels = self.teacher_model.representation_model.hidden_channels // 2
        self.mid_channels = (self.student_channels + self.teacher_channels) // 2
        self.share_head = nn.Sequential(
            nn.Linear(self.student_channels, self.student_channels),
            nn.SiLU(),
        )
        self.contrastive_output_head = nn.Sequential(
            nn.Linear(self.student_channels, self.mid_channels),
            nn.SiLU(),
            nn.Linear(self.mid_channels, self.teacher_channels),
        )
        self.energy_output_head = nn.Sequential(
            nn.Linear(self.student_channels, self.student_channels // 2),
            nn.SiLU(),
            nn.Linear(self.student_channels // 2, 1),
        )
        
        self.reset_parameters()
        self.freeze_verbose_params()
        
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.share_head[0].weight)
        self.share_head[0].bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.contrastive_output_head[0].weight)
        self.contrastive_output_head[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.contrastive_output_head[2].weight)
        self.contrastive_output_head[2].bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.energy_output_head[0].weight)
        self.energy_output_head[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.energy_output_head[2].weight)
        self.energy_output_head[2].bias.data.fill_(0)
        
    def freeze_verbose_params(self):
        
        print("Freeze the unused output head params...")
        for params in self.student_model.output_model.out_scalar_netowrk.parameters():
            params.requires_grad = False
        
    def forward(self, data: Data, stage="train", **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        
        out_rdkit, _ = self.student_model(data, use_pos_kind="rdkit")
        out_rdkit = self.share_head(out_rdkit)
        verify_eq = None
        if self.teacher_model is not None and stage == "train":
            with torch.no_grad():
                out_eq, _ = self.teacher_model(data, use_pos_kind="eq")
                verify_eq = self.teacher_model.output_model.post_reduce(out_eq)
                verify_eq = verify_eq * self.teacher_model.std + self.teacher_model.mean
                out_eq = self.teacher_model.output_model.out_scalar_netowrk[0](out_eq)
        else:
            out_eq = None
                
        pred_rdkit = self.energy_output_head(out_rdkit)
        out_rdkit = self.contrastive_output_head(out_rdkit)
        
        return out_eq, out_rdkit, pred_rdkit, verify_eq

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