import argparse
import os
import esm
import MDAnalysis as mda
import numpy as np
import pytorch_lightning as pl
import torch
import torch_geometric.loader as geom_data
from MDAnalysis.analysis import distances
from rdkit import Chem
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from src.util import LoadFromFile
from src.get_our_data import wash_molecule
from src.unimol_tools import UniMolRepr
clf = UniMolRepr(data_type='molecule')

three_to_one = {'ALA':	'A',
'ARG':	'R',
'ASN':	'N',
'ASP':	'D',
'CYS':	'C',
'CYX':	'C',
'GLN':	'Q',
'GLU':	'E',
'GLY':	'G',
'HIS':	'H',
'ILE':	'I',
'LEU':	'L',
'LYS':	'K',
'MET':	'M',
'MSE':  'M', # MSE this is almost the same AA as MET. The sulfur is just replaced by Selen
'PHE':	'F',
'PRO':	'P',
'PYL':	'O',
'SER':	'S',
'SEC':	'U',
'THR':	'T',
'TRP':	'W',
'TYR':	'Y',
'VAL':	'V',
'ASX':	'B',
'GLX':	'Z',
'XAA':	'X',
'XLE':	'J'}

parser = argparse.ArgumentParser('Test a GCN Prediction Model')
parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')
parser.add_argument('--batch-size',  help='training batch size', type=int, default=6)
parser.add_argument('--seed',type=int,  help='random seed', default=42)
parser.add_argument('--dropout_rate',  help='dropout rate', type=float, default=0.15)
parser.add_argument('--hidden_dim',  help='hidden dimension', type=int, default=256)
parser.add_argument('--residual_layers',  help='residual layers', type=int, default=10)
parser.add_argument('--num_heads_visnet',  help='number of heads', type=int, default=32)
parser.add_argument('--num_heads_Transformer',  help='number of heads', type=int, default=8)
parser.add_argument('--num_encoder_layers',  help='number of layers', type=int, default=3)
parser.add_argument('--num_decoder_layers',  help='number of layers', type=int, default=3)
parser.add_argument('--num_layers_visnet',  help='number of layers', type=int, default=9)
parser.add_argument('--num_rbf',  help='number of rbf', type=int, default=64)
parser.add_argument('--lmax',  help='lmax', type=int, default=2)
parser.add_argument('--trainable_rbf',  help='trainable rbf', action='store_true')
parser.add_argument('--vecnorm_trainable',  help='vecnorm trainable', action='store_true')
parser.add_argument('--lr',  help='learning rate', type=float, default=1e-4)
parser.add_argument('--weight_decay',  help='weight decay', type=float, default=5e-4)
parser.add_argument('--lr_factor',  help='learning rate factor', type=float, default=0.8)
parser.add_argument('--lr_patience',  help='learning rate patience', type=int, default=5)
parser.add_argument('--lr_min',  help='learning rate min', type=float, default=1e-7)
parser.add_argument('--loss_alpha',  help='loss alpha', type=float, default=0.25)
parser.add_argument('--loss_gamma',  help='loss gamma', type=float, default=2)
parser.add_argument('--proj_name',  help='project name', type=str, default='visnet')

parser.add_argument('--ckpt_path',  help='path to ckpt file for test')
parser.add_argument('--out_path',  help='path to out file for test')
parser.add_argument('--input_fn',  help='input file name for test')
parser.add_argument('--output_fn',  help='output file name for test')

args = parser.parse_args()
args = vars(args)

if len(args['input_fn'].split(',')) == 1: 
    from DeepGlycanSite import TrainSite
elif len(args['input_fn'].split(',')) == 2:
    from DeepGlycanSite_lig import TrainSite

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def obtain_resname(res):
    resname = res.resname.strip()
    return resname


def obtain_self_dist(res):
    try:
        xx = res.atoms
        dists = distances.self_distance_array(xx.positions)
        ca = xx.select_atoms("name CA")
        c  = xx.select_atoms("name C")
        n  = xx.select_atoms("name N")
        o  = xx.select_atoms("name O")
        return [dists.max()*0.1, dists.min()*0.1, distances.dist(ca,o)[-1][0]*0.1, distances.dist(o,n)[-1][0]*0.1, distances.dist(n,c)[-1][0]*0.1]
    except:
        return [0, 0, 0, 0, 0]

def obtain_dihediral_angles(res):
    try:
        if res.phi_selection() is not None:
            phi = res.phi_selection().dihedral.value()
        else:
            phi = 0
        if res.psi_selection() is not None:
            psi = res.psi_selection().dihedral.value()
        else:
            psi = 0
        if res.omega_selection() is not None:
            omega = res.omega_selection().dihedral.value()
        else:
            omega = 0
        if res.chi1_selection() is not None:
            chi1 = res.chi1_selection().dihedral.value()
        else:
            chi1 = 0
        return [phi*0.01, psi*0.01, omega*0.01, chi1*0.01]
    except:
        return [0, 0, 0, 0]

def calc_res_features(res):
    residue_type = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR',
                    'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP',
                    'GLU', 'LYS', 'ARG', 'HIS']

    residue_name = one_of_k_encoding_unk(obtain_resname(res), residue_type)
    return np.array(residue_name + obtain_self_dist(res) + obtain_dihediral_angles(res))   # 20 维resid， 5维dist， 4维dihedral

def calc_dist(res1, res2):
    dist_array = distances.distance_array(res1.atoms.positions, res2.atoms.positions)
    return dist_array

def obtain_ca_pos(res):
    try:
        pos = res.atoms.select_atoms("name CA").positions[0]
        return pos
    except:  
        return res.atoms.positions.mean(axis=0)


def obtain_sidechain_pos(res):
    try:
        pos = res.atoms.select_atoms("not name CA and not name O and not name N and not name H and not name C and not type H").positions.mean(axis=0)
        if np.isnan(pos).any():
            pos = res.atoms.positions.mean(axis=0)
        return pos
    except:  
        return res.atoms.positions.mean(axis=0)

def obtain_edge_CA(u, cutoff=8.0):
    edgeids = []
    CA_pos_array = np.array([obtain_ca_pos(res) for res in u.residues])
    SS_pos_array = np.array([obtain_sidechain_pos(res) for res in u.residues])
    dm = distance_matrix(CA_pos_array, CA_pos_array)
    sm = distance_matrix(SS_pos_array, SS_pos_array)
    array = dm <= cutoff
    row, col = np.diag_indices_from(array)
    array[row,col] = False
    edgeids = np.array(np.where(array)).T  
    dis_CA_out = dm[edgeids[:,0], edgeids[:,1]]
    dis_SS_out = sm[edgeids[:,0], edgeids[:,1]]


    return edgeids,np.array([dis_CA_out, dis_SS_out]).T


def check_connect(u, i, j):
    if abs(i-j) != 1:
       return 0
    else:
       return 1


# filter the number 1 in the end of around4_resids( we don't know how much 1 in the end of around4_resids)
def filter_1(around4_resids,around4_resnames):
    if 'LIG' in around4_resnames:
        for num,i in enumerate(around4_resnames[::-1]):
            if i != 'LIG':
                return around4_resids[:-num]
    else:
        return around4_resids

def get_atom_coordinates(residue, atom_name):
    for atom in residue.atoms:
        if atom.name == atom_name:
            return atom.position
    for atom in residue.atoms:
        if residue.resname == 'GLY' and atom.name == 'H01':
            return atom.position
    return None

def dihedral_angle(coords_A, coords_B, coords_C, coords_D):
    # Calculate the vectors
    BA = coords_A - coords_B
    BC = coords_C - coords_B
    CD = coords_D - coords_C

    # Calculate the normal vectors
    N1 = np.cross(BA, BC)
    N2 = np.cross(BC, CD)

    # Normalize the normal vectors
    N1 = N1 / np.linalg.norm(N1)
    N2 = N2 / np.linalg.norm(N2)

    # Calculate the dihedral angle using dot product and cross product
    angle = np.arctan2(np.dot(np.cross(N1, N2), BC / np.linalg.norm(BC)), np.dot(N1, N2))

    # Convert the angle to degrees
    angle = np.degrees(angle)

    return angle


# Function to convert a pdb file into an ESM embedding
def pdb_to_esm_embedding(file_path):
    # Parse the pdb file and extract the protein sequence
    sequence = get_sequences_from_pdbfile(file_path)

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data
    data = [('prot1',sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    
    return token_representations.squeeze(0)

def get_sequences_from_pdbfile(inp):
    file_path = inp
    biopython_parser = PDBParser()
    structure = biopython_parser.get_structure('random_id', file_path)
    structure = structure[0]
    sequence = None
    # print(chain,file_path)
    seq = ''
    for chain in structure:
        # print(chain,,dir(chain))
        for res_idx, residue in enumerate(structure[chain.get_id()]):
            try:
                seq += three_to_one[residue.get_resname()]
            except Exception as e:
                seq += 'X'
                print("encountered unknown AA: ", residue.get_resname(), ' in the complex. Replacing it with a dash - .')

    if sequence is None:
        sequence = seq
    else:
        sequence += ("" + seq)
    return sequence

def prot_to_pyg_graph(target_file_name, cutoff):
    """obtain the residue graphs in pyg format"""
    u = mda.Universe(target_file_name)
    avg_pos_list = []
    for residue in u.residues:
        avg_pos_list.append(residue.atoms.center_of_mass())
    avg_pos_list = np.array(avg_pos_list)
   
    res_feats = torch.tensor(np.array([calc_res_features(res) for res in u.residues])).float()

    CA_edgeids, CA_distm = obtain_edge_CA(u, cutoff) 

    CA_src_list, CA_dst_list = zip(*CA_edgeids)

    CA_pos_list = []
    CB_pos_list = []
    N_pos_list = []
    for residue in u.residues:
        # print(residue.resname,residue.atoms)
        try:
            CA_pos_list.append(residue.atoms.select_atoms("name CA").positions[0])
        except:
            CA_pos_list.append(residue.atoms.positions.mean(axis=0))
        try:
            CB_pos_list.append(residue.atoms.select_atoms("name CB").positions[0])
        except: # GLY has no CB, add nan
            try:
                CB_pos_list.append(residue.atoms.select_atoms("name H01").positions[0])
            except:
                if obtain_sidechain_pos(residue)[0] != residue.atoms.positions.mean(axis=0)[0]:
                    CB_pos_list.append(obtain_sidechain_pos(residue))
                else:
                    CB_pos_list.append(obtain_sidechain_pos(residue)+np.array([0.5,0.5,0.5]))
        try:
            N_pos_list.append(residue.atoms.select_atoms("name N").positions[0])
        except:
            N_pos_list.append(residue.atoms.positions.mean(axis=0))

    CA_pos_list = np.array(CA_pos_list)
    CB_pos_list = np.array(CB_pos_list)
    N_pos_list = np.array(N_pos_list)
    real_name = target_file_name.split('/')[-1].split('.')[0]

    CA_edge_connect = torch.tensor(np.array([check_connect(u, x, y) for x,y in zip(CA_src_list, CA_dst_list)]))
    CA_edge_feats   = torch.cat([CA_edge_connect.view(-1,1), torch.tensor(CA_distm * 0.1)], dim=1)

    esm_fea = pdb_to_esm_embedding(target_file_name)[1:-1]


    return Data(edge_feats=CA_edge_feats.float(),edge_index=torch.from_numpy(CA_edgeids).t(),x=torch.cat((res_feats,esm_fea),1),
    protein_name=np.array(real_name).repeat(res_feats.shape[0]), pos = torch.from_numpy(avg_pos_list), pos_CA = torch.from_numpy(CA_pos_list), pos_CB = torch.from_numpy(CB_pos_list), pos_N = torch.from_numpy(N_pos_list))



def calc_atom_features(atom, explicit_H=False):
    atom_symbol = [  'C',  'N',  'O',  'S',  'F',  'P', 'Cl', 'Br', 'I', 'B', 
                    'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']
    atom_degree = [ 0, 1, 2, 3, 4, 5, 6 ]
    hybrid_type = [ Chem.rdchem.HybridizationType.SP,    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,   Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2, 'other'] 

    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbol)        \
            + one_of_k_encoding(atom.GetDegree(), atom_degree)            \
            + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]     \
            + one_of_k_encoding_unk(atom.GetHybridization(), hybrid_type) \
            + [atom.GetIsAromatic()]


    total_numhs = [0, 1, 2, 3, 4]
    if not explicit_H:
       results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), total_numhs)
                                        
    return np.array(results)

def calc_bond_features(bond):
    bt = bond.GetBondType()
    bond_feats = [
           bt == Chem.rdchem.BondType.SINGLE, 
           bt == Chem.rdchem.BondType.DOUBLE,
           bt == Chem.rdchem.BondType.TRIPLE, 
           bt == Chem.rdchem.BondType.AROMATIC,
           bond.GetIsConjugated(),
           bond.IsInRing()]

    return np.array(bond_feats).astype(int)

def mol_to_graph(mol, explicit_H=False, use_chirality=False):

    num_atoms = mol.GetNumAtoms()
    x = np.zeros((num_atoms, 38))
    try:
        Chem.SanitizeMol(mol)
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            x[i] = calc_atom_features(atom, explicit_H=explicit_H)
        chiral_arr    = np.zeros([num_atoms, 3]) 
        chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True, useLegacyImplementation=False)
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] =1 
            elif rs == 'S':
                chiral_arr[i, 1] =1 
            else:
                chiral_arr[i, 2] =1 
        x = np.concatenate([x, chiral_arr], axis=1)
    except:
        try: 	
            Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except:
            pass
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            x[i] = calc_atom_features(atom, explicit_H=explicit_H)
        chiral_arr    = np.zeros([num_atoms, 3]) 
        chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True, useLegacyImplementation=False)
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] =1 
            elif rs == 'S':
                chiral_arr[i, 1] =1 
            else:
                chiral_arr[i, 2] =1 
        x = np.concatenate([x, chiral_arr], axis=1)
    x = torch.tensor(x, dtype=torch.float)

    src_list = []
    dst_list = []
    bond_feats_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()

        bond_feats = calc_bond_features(bond)
        src_list.extend([u, v])
        dst_list.extend([v, u])         
        
        bond_feats_all.append(bond_feats)
        bond_feats_all.append(bond_feats)

    bond_feats_all = np.array(bond_feats_all)
    edge_attr  = torch.tensor(bond_feats_all, dtype=torch.float)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    ligand_smiles = Chem.MolToSmiles(mol)
    reprs = clf.get_repr([ligand_smiles])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, U_pre=torch.tensor(reprs['cls_repr'][0], dtype=torch.float))

def calculate_distance_batch(points1, points2):
    return torch.norm(points1 - points2, dim=1)

def calculate_dihedral_batch(p0, p1, p2, p3):
    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 = b1 / torch.norm(b1, dim=1, keepdim=True)

    v = b0 - torch.sum(b0 * b1, dim=1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=1, keepdim=True) * b1

    x = torch.sum(v * w, dim=1)
    y = torch.sum(torch.cross(b1, v, dim=1) * w, dim=1)
    return torch.atan2(y, x) * 180 / np.pi

class TestSiteRec(TrainSite):
    def __init__(self, **model_kwargs):
        model_kwargs['batch_size'] = 16
        super(TestSiteRec, self).__init__(**model_kwargs)
        self.results_dir = args['out_path']
        # print(model_kwargs,'model_kwargs')
        self.output_fn =  args['output_fn']

    def forward(self, data, mode="train"):

        ligand, target = data
        pi = self.model(target)
        mdn = 0
        flat_arrays = np.array([arr for sublist in data[1].protein_name for arr in sublist])
        loss = mdn
        return loss,pi,flat_arrays

    def test_step(self, batch, batch_idx):
        loss,pi,name = self.forward(batch, mode="test")
        self.log("test_loss", loss, batch_size=self.batch_size)
        os.makedirs(self.results_dir, exist_ok=True)
        w = open(os.path.join(self.results_dir, self.output_fn), "w")
        w.close()
        result_file = os.path.join(self.results_dir, self.output_fn)
        # print(result_file,'result_file')
        with open(result_file, "a") as f:
            for i in range(len(name)):
                f.write(f"{name[i]} {pi[i]}\n")

class TestSiteLig(TrainSite):
    def __init__(self, **model_kwargs):
        model_kwargs['batch_size'] = 16
        super(TestSiteLig, self).__init__(**model_kwargs)
        self.results_dir = args['out_path']
        self.output_fn =  args['output_fn']
        # print(self.results_dir,'self.results_dir',model_kwargs)

    def forward(self, data, mode="train"):

        ligand, target = data
        pi = self.model(ligand,target)
        mdn = 0
        flat_arrays = np.array([arr for sublist in data[1].protein_name for arr in sublist])
        loss = mdn
        return loss,pi,flat_arrays

    def test_step(self, batch, batch_idx):
        loss,pi,name = self.forward(batch, mode="test")
        self.log("test_loss", loss, batch_size=self.batch_size)
        os.makedirs(self.results_dir, exist_ok=True)
        w = open(os.path.join(self.results_dir, self.output_fn), "w")
        w.close()
        result_file = os.path.join(self.results_dir, self.output_fn)
        print(result_file,'result_file')
        with open(result_file, "a") as f:
            for i in range(len(name)):
                f.write(f"{name[i]} {pi[i]}\n")

def get_pl_trainer(out_path):
    """ Create a PyTorch Lightning trainer with the generation callback """

    root_dir = os.path.join(out_path, "results")
    os.makedirs(root_dir, exist_ok=True)

    trainer = pl.Trainer(
        default_root_dir=root_dir,
        accelerator = 'cpu',
        devices = 1,
        
    )

    trainer.logger._default_hp_metric = None
    return trainer

def main():

    input_file_name = args['input_fn'].split(',')

    if len(input_file_name) == 1:   # pure rec mode, only a file
        input_file_name = input_file_name[0]
        target = prot_to_pyg_graph(input_file_name, 8.0)
        ligand = torch.randn(1,1)
        Data_list = [[ligand, target]]

    elif len(input_file_name) == 2:   # rec+lig mode, only a file
        rec_file_name = input_file_name[0]
        ligand_file_name = wash_molecule(input_file_name[1])
        target = prot_to_pyg_graph(rec_file_name, 8.0)
        ligand = mol_to_graph(Chem.SDMolSupplier(ligand_file_name)[0])
        Data_list = [[ligand, target]]

    for data in Data_list:
        edge_index = data[1].edge_index
        pos_CB = data[1].pos_CB
        pos_CA = data[1].pos_CA
        pos_N = data[1].pos_N
        num_edges = edge_index.shape[1]
        
        existing_edge_feats = data[1].edge_feats
        new_edge_feats = torch.zeros((num_edges, 6), device=existing_edge_feats.device) # ensure same device

        node1 = edge_index[0, :].long()
        node2 = edge_index[1, :].long()

        dist = calculate_distance_batch(pos_CB[node1], pos_CB[node2])
        new_edge_feats[:, 0] = dist / 10

        dihedral = calculate_dihedral_batch(pos_CA[node1], pos_CB[node1], pos_CB[node2], pos_CA[node2])
        dihedral1 = calculate_dihedral_batch(pos_N[node1], pos_CA[node1], pos_CB[node1], pos_CB[node2])
        dihedral2 = calculate_dihedral_batch(pos_CB[node1], pos_CB[node2], pos_CA[node2], pos_N[node2])
        dihedral3 = calculate_dihedral_batch(pos_N[node2], pos_CA[node2], pos_CB[node2], pos_CB[node1])
        dihedral4 = calculate_dihedral_batch(pos_CB[node2], pos_CB[node1], pos_CA[node1], pos_N[node1])

        new_edge_feats[:, 1] = dihedral / 100
        new_edge_feats[:, 2] = dihedral1 / 100
        new_edge_feats[:, 3] = dihedral2 / 100
        new_edge_feats[:, 4] = dihedral3 / 100
        new_edge_feats[:, 5] = dihedral4 / 100
        data[1].edge_feats = torch.cat((existing_edge_feats, new_edge_feats), dim=1)
    
    test_list  = Data_list

    torch.save(test_list, args['out_path'] + f'/{args["output_fn"]}.pt')

    test_loader  = geom_data.DataLoader(test_list,  batch_size=1, num_workers=4)

    if len(args['input_fn'].split(',')) == 1: 
        # print(args['output_fn'])

        model = TestSiteRec(seed = args['seed'], dropout_rate = args['dropout_rate'], batch_size=16, results_dir = args['out_path'], output_fn = args['output_fn'],
        hidden_dim = args['hidden_dim'], residual_layers = args['residual_layers'],
        num_heads_visnet = args['num_heads_visnet'], num_heads_Transformer = args['num_heads_Transformer'],
        num_encoder_layers = args['num_encoder_layers'], num_decoder_layers = args['num_decoder_layers'],
        num_layers_visnet = args['num_layers_visnet'], num_rbf = args['num_rbf'], lmax = args['lmax'],
        trainable_rbf = args['trainable_rbf'], vecnorm_trainable = args['vecnorm_trainable'],
        lr = args['lr'], weight_decay = args['weight_decay'], lr_factor = args['lr_factor'],
        lr_patience = args['lr_patience'], lr_min = args['lr_min'], loss_alpha = args['loss_alpha'],
        loss_gamma = args['loss_gamma']).load_from_checkpoint(args['ckpt_path'],map_location=torch.device('cpu'))

    elif len(args['input_fn'].split(',')) == 2: 
        model = TestSiteLig(seed = args['seed'], dropout_rate = args['dropout_rate'], batch_size=16, results_dir = args['out_path'], output_fn = args['output_fn'],
        hidden_dim = args['hidden_dim'], residual_layers = args['residual_layers'],
        num_heads_visnet = args['num_heads_visnet'], num_heads_Transformer = args['num_heads_Transformer'],
        num_encoder_layers = args['num_encoder_layers'], num_decoder_layers = args['num_decoder_layers'],
        num_layers_visnet = args['num_layers_visnet'], num_rbf = args['num_rbf'], lmax = args['lmax'],
        trainable_rbf = args['trainable_rbf'], vecnorm_trainable = args['vecnorm_trainable'],
        lr = args['lr'], weight_decay = args['weight_decay'], lr_factor = args['lr_factor'],
        lr_patience = args['lr_patience'], lr_min = args['lr_min'], loss_alpha = args['loss_alpha'],
        loss_gamma = args['loss_gamma']).load_from_checkpoint(args['ckpt_path'],map_location=torch.device('cpu'))

    trainer = get_pl_trainer(args['out_path'])

    trainer.test(model, dataloaders=test_loader)

if __name__ == '__main__':
    main()