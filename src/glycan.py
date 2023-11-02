# Written by Xi Cheng (xicheng@simm.ac.cn)
import sys
import numpy as np
from   rdkit import Chem
import MDAnalysis as mda
from   MDAnalysis.analysis import distances

### basic anaysis ###
def calc_dist(pos1, pos2):
    '''calculate distances between two position arrays'''
    dist_array = distances.distance_array(pos1, pos2)
    return dist_array

### pdb ###
def select_byres_cmd(iresidue):
    '''MDAnalysis selection cmd for a residue'''
    return ("resnum %d " %(iresidue.resnum)) + " and resname " + iresidue.resname + " and segid " + iresidue.segid

def get_bound_ligands(ligands, pdb_file_name, nresidue_cutoff=3, dist_cutoff=4):
    '''get ligands binding to protein'''
    u = mda.Universe(pdb_file_name)
    bound_ligands = ligands.select_atoms("resid 9999")
    for i in range(len(ligands.residues)):
        ligand_atoms = ligands.select_atoms(select_byres_cmd(ligands.residues[i]))
        bound_residues = u.select_atoms(("around %f group ligand" %dist_cutoff), ligand=ligand_atoms)
        bound_residue_num = len((bound_residues.select_atoms("protein")).residues)
        if bound_residue_num >= nresidue_cutoff:
           bound_ligands = bound_ligands + ligand_atoms
    return bound_ligands

### for glycan ###
def is_oxygen_in_hexa_ring(o_atom, mol):
    '''check if oxygen is in a hexa ring'''
    c1_c5_atoms = mol.select_atoms("name C* and bonded index %d" %o_atom.index)
    if len(c1_c5_atoms) < 2:
       return 0
    c1_atom = mol.select_atoms("index %d" %c1_c5_atoms.indices[0])
    c5_atom = mol.select_atoms("index %d" %c1_c5_atoms.indices[1])
    c2_atoms = mol.select_atoms("name C* and bonded index %d" %c1_atom.indices[0])
    if len(c2_atoms) < 1:
       return 0
    c4_atoms = mol.select_atoms("name C* and bonded index %d" %c5_atom.indices[0])
    if len(c4_atoms) < 1:
       return 0
    c3_atoms = mol.select_atoms("name C*") - c1_c5_atoms - c2_atoms - c4_atoms
    if len(c3_atoms) < 1:
       return 0
    c2_pos = c2_atoms.positions
    c4_pos = c4_atoms.positions
    for i in range(len(c3_atoms)):
        c3_pos = c3_atoms[i].position
        dist2 = calc_dist(c3_pos, c2_pos)
        dist4 = calc_dist(c3_pos, c4_pos)
        if (dist2.min() < 2) and (dist4.min() < 2):
           return 1
    return 0

def is_oxygen_in_penta_ring(o_atom, mol):
    '''check if oxygen is in a penta ring'''
    c1_c4_atoms = mol.select_atoms("name C* and bonded index %d" %o_atom.index)
    if len(c1_c4_atoms) < 2:
       return 0
    c1_atom  = mol.select_atoms("index %d" %c1_c4_atoms.indices[0])
    c4_atom  = mol.select_atoms("index %d" %c1_c4_atoms.indices[1])
    c2_atoms = mol.select_atoms("name C* and bonded index %d" %c1_atom.indices[0])
    if len(c2_atoms) < 1:
       return 0
    c3_atoms = mol.select_atoms("name C* and bonded index %d" %c4_atom.indices[0]) 
    if len(c3_atoms) < 1:
       return 0
    c2_pos = c2_atoms.positions
    c3_pos = c3_atoms.positions
    dist = calc_dist(c2_pos, c3_pos)
    if dist.min() < 2:
       return 1
    return 0

def contain_hexa_ring(mol):
    '''check if a molecule contains hexa ring structure'''
    o_atoms = mol.select_atoms("name O* and bonded name C*")
    for i in range(len(o_atoms)):
        oxygen = o_atoms[i]
        if is_oxygen_in_hexa_ring(oxygen, mol) or is_oxygen_in_penta_ring(oxygen, mol):
           return 1
    return 0

def get_glycans(heta):
    '''get glycans in pdb file'''
    # select compounds
    compounds = heta.select_atoms("resid 9999")
    for i in range(len(heta.residues)):
        heta_atoms = heta.select_atoms(select_byres_cmd(heta.residues[i]))
        c_atoms = heta_atoms.select_atoms("name C*")
        o_atoms = heta_atoms.select_atoms("name O*")
        if len(compounds):
           # avoid clash
           compound_dist = calc_dist(compounds.atoms.positions, heta_atoms.atoms.positions)
           if compound_dist.min() < 1.0:
              continue
        if len(c_atoms) and len(o_atoms):
           compounds = compounds + heta_atoms
    # select hexa_ring compounds as glycans
    glycans = compounds.select_atoms("resid 9999")
    for i in range(len(compounds.residues)):
        molecule = compounds.select_atoms(select_byres_cmd(compounds.residues[i]))
        if contain_hexa_ring(molecule):
           glycans = glycans + molecule
    return glycans

def get_full_glycan_mol_list(glycans, heta, max_round=5):
    '''get full length glycan molecule arrays'''
    glycan_mol = []
    assigned_glycans = heta.select_atoms("resid 9999")
    for i in range(len(glycans.residues)):
        glycan_res = heta.select_atoms(select_byres_cmd(glycans.residues[i]))
        if len(assigned_glycans.select_atoms("group glycan", glycan=glycan_res)):
           continue
        for j in range(max_round):
            connected_heta = heta.select_atoms("byres bonded group glycan", glycan=glycan_res)
            if len(connected_heta):
               glycan_res = glycan_res | connected_heta
            else:
               break
        glycan_mol.append(glycan_res)
        for k in range(len(glycan_res.residues)):
            assigned_glycans = assigned_glycans | glycan_res
    return glycan_mol

def get_glycan_binding_chain_list(mol_list, pdb_file_name, dist_cutoff=4):
    '''get glycan-binding protein chain list'''
    u = mda.Universe(pdb_file_name)
    chain_list = []
    for i in range(len(mol_list)):
        bind_atoms    = u.select_atoms(("around %f group mol" %dist_cutoff), mol=mol_list[i])
        bind_residues = bind_atoms.select_atoms("protein")
        bind_chains   = u.select_atoms("same chainID as group br", br=bind_residues)
        bind_protein_chains = bind_chains.select_atoms("protein")
        chain_list.append(bind_protein_chains)
    return chain_list
