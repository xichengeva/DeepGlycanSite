# Written by Xi Cheng (xicheng@simm.ac.cn)
import os
import sys
import numpy as np
import MDAnalysis as mda
from   MDAnalysis.analysis import distances
import pymol2
from   pymol import cmd
from   src.glycan import get_glycans
from   src.glycan import get_bound_ligands
from   src.glycan import get_full_glycan_mol_list
from   src.glycan import get_glycan_binding_chain_list
from   src.file   import get_list_from_file

### MDAnalysis ###
def get_heta_components(pdb_file_name):
    '''get hetatm atom groups from pdb'''
    u = mda.Universe(pdb_file_name)
    return u.select_atoms("record_type HETATM")

### basic ###
def get_pdb_id(pdb_file_name):
    '''get pdb id from pdbxxxx.ent.gz file'''
    index = len(pdb_file_name.split("pdb")) - 1
    name = pdb_file_name.split("pdb")[index]
    pdb_id = name.split(".")[0]
    return pdb_id

def get_protein_chain_name(chain_list):
    sort_chain_list = np.sort(np.unique(chain_list))
    name=""
    for i in range(len(sort_chain_list)):
        name = name + sort_chain_list[i]
    return name.upper()

def assign_mol_resid(mol, rid=0):
    '''assign residue id from rid'''
    for i in range(len(mol.residues)):
        mol.residues[i].resid = i+rid
    return mol

def prepare_protein_glycan_files(compressed_pdb_file_name, output_path):
    '''get protein.pdb and glycan.mol2 from compressed pdb*ent.gz'''
    job_id_list = []
    pdb_id = get_pdb_id(compressed_pdb_file_name)
    pdb_file_name = output_path + "/" + pdb_id + ".pdb"
 
    print("Processing pdb...")
    cmd.load(compressed_pdb_file_name, pdb_id)
    cmd.h_add()
    cmd.save(pdb_file_name)
    cmd.remove("all")
    print(pdb_file_name)

    print("Processing glycan...")
    heta = get_heta_components(pdb_file_name)
    glycans = get_glycans(heta)
    bound_glycans   = get_bound_ligands(glycans, pdb_file_name)
    glycan_mol_list = get_full_glycan_mol_list(bound_glycans, heta)
    num_glycan_mol  = len(glycan_mol_list)
    print("Fetched %d glycan molecules." %(num_glycan_mol))
    if num_glycan_mol == 0:
       return job_id_list

    print("Processing protein...")
    protein_chain_list = get_glycan_binding_chain_list(glycan_mol_list, pdb_file_name)

    print("Writing...")
    for i in range(len(glycan_mol_list)):
        glycan_mol_name    = ("carb%d" %(i+1))
        protein_chain_name = get_protein_chain_name(protein_chain_list[i].chainIDs)
        complex_name = pdb_id + protein_chain_name + "_" + glycan_mol_name 
        protein_file_name  = output_path + "/" + complex_name + "_protein.pdb"
        glycan_file_name   = output_path + "/" + complex_name + "_ligand.pdb"
#        glycan_mol_list[i].write(glycan_file_name)
        glycan_mol = assign_mol_resid(glycan_mol_list[i], 1)
        glycan_mol.write(glycan_file_name)
#        print(glycan_file_name)
        protein_chain = assign_mol_resid(protein_chain_list[i])
        protein_chain.write(protein_file_name)
#        protein_chain_list[i].write(protein_file_name)
#        print(protein_file_name)
        job_id_list.append(complex_name)
        glycan_mol2_file_name = glycan_file_name.replace(".pdb", ".mol2")
        obabel_cmd = "obabel -ipdb " + glycan_file_name + " -omol2 -O" + glycan_mol2_file_name + ("\n")
        os.system(obabel_cmd)
#        print(glycan_mol2_file_name)
    return job_id_list
