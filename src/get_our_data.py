import os,shutil,subprocess,sys
from tqdm import tqdm
from rdkit import Chem
from rdkit import rdBase
import rdkit.RDLogger as rdl

# Disable RDKit warnings and error messages
rdBase.DisableLog('rdApp.warning')
rdBase.DisableLog('rdApp.error')
import os,shutil,subprocess

def find_bad_C(mol):
    bad_atom_list = []
    for atom in mol.GetAtoms():
        try:
            valence = atom.GetExplicitValence()
            # print(valence, atom.GetAtomicNum())
            if valence > 4 and atom.GetAtomicNum() == 6:
                bad_atom_list.append(atom.GetIdx())
        except ValueError:
            continue
    return bad_atom_list


def find_bad_N(mol):
    bad_atom_list = []
    for atom in mol.GetAtoms():
        try:
            valence = atom.GetExplicitValence()
            # print(valence, atom.GetAtomicNum())
            if valence > 3 and atom.GetAtomicNum() == 7:
                bad_atom_list.append(atom.GetIdx())
        except ValueError:
            continue
    return bad_atom_list


def find_bad_S(mol):
    bad_atom_list = []
    for atom in mol.GetAtoms():
        try:
            valence = atom.GetExplicitValence()
            # print(valence, atom.GetAtomicNum())
            if valence > 6 and atom.GetAtomicNum() == 16:
                bad_atom_list.append(atom.GetIdx())
        except ValueError:
            continue
    return bad_atom_list


def find_bad_F(mol):
    bad_atom_list = []
    for atom in mol.GetAtoms():
        try:
            valence = atom.GetExplicitValence()
            # print(valence, atom.GetAtomicNum())
            if valence > 1 and atom.GetAtomicNum() == 9:
                bad_atom_list.append(atom.GetIdx())
        except ValueError:
            continue
    return bad_atom_list


def find_which_to_delete(mol, atom_id):
    atom = mol.GetAtomWithIdx(atom_id)
    neighbors = atom.GetNeighbors()

    min_count = float('inf')
    min_neighbor = None
    for neighbor in neighbors:
        neighbor_id = neighbor.GetIdx()
        neighbor_neighbors = neighbor.GetNeighbors()
        count = len(neighbor_neighbors)
        if count < min_count:
            min_count = count
            min_neighbor = neighbor_id
    for neighbor in neighbors:
        # if the atom is Co, set it as the min neighbor
        if neighbor.GetAtomicNum() == 27:
            min_neighbor = neighbor.GetIdx()

    return atom_id, min_neighbor


def find_which_to_delete_D_to_S(mol, atom_id):
    atom = mol.GetAtomWithIdx(atom_id)
    neighbors = atom.GetNeighbors()

    min_count = float('inf')
    min_neighbor = None
    # if neibor has DOUBLE bond with atom, set it as the min neighbor
    for neighbor in neighbors:
        bond = mol.GetBondBetweenAtoms(atom_id, neighbor.GetIdx())
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            min_neighbor = neighbor.GetIdx()

    return atom_id, min_neighbor


def find_which_to_delete_S(mol, atom_id):
    atom = mol.GetAtomWithIdx(atom_id)
    neighbors = atom.GetNeighbors()

    min_count = float('inf')
    min_neighbor = None
    # if neibor has DOUBLE bond with N atom, set it as the min neighbor
    for neighbor in neighbors:
        bond = mol.GetBondBetweenAtoms(atom_id, neighbor.GetIdx())
        if bond.GetBondType() == Chem.BondType.DOUBLE and neighbor.GetAtomicNum() == 7:
            min_neighbor = neighbor.GetIdx()

    return atom_id, min_neighbor


def find_which_to_delete_F(mol, atom_id):
    atom = mol.GetAtomWithIdx(atom_id)
    neighbors = atom.GetNeighbors()
    conformer = mol.GetConformer()

    max_distance = float('-inf')
    max_neighbor = None

    # Find the farthest neighbor and set it as the max_neighbor
    for neighbor in neighbors:
        atom_pos = conformer.GetAtomPosition(atom_id)
        neighbor_pos = conformer.GetAtomPosition(neighbor.GetIdx())

        distance = atom_pos.Distance(neighbor_pos)

        if distance > max_distance:
            max_distance = distance
            max_neighbor = neighbor.GetIdx()

    return atom_id, max_neighbor


def find_which_to_delete_H(mol, atom_id):
    atom = mol.GetAtomWithIdx(atom_id)
    neighbors = atom.GetNeighbors()
    conformer = mol.GetConformer()

    max_distance = float('-inf')
    max_neighbor = None

    # Find the farthest neighbor and set it as the max_neighbor
    for neighbor in neighbors:
        print(neighbor)
        atom_pos = conformer.GetAtomPosition(atom_id)
        neighbor_pos = conformer.GetAtomPosition(neighbor.GetIdx())

        distance = atom_pos.Distance(neighbor_pos)
        print(neighbor, distance, max_distance, neighbor.GetAtomicNum())

        if distance > max_distance and neighbor.GetAtomicNum() == 1:
            max_distance = distance
            max_neighbor = neighbor.GetIdx()

    return atom_id, max_neighbor


def remove_atoms_and_update_conformer(mol, atom_ids_to_remove, conformer_id=0):
    mw = Chem.RWMol(mol)
    conf = mol.GetConformer(conformer_id)

    new_conf = Chem.Conformer(mol.GetNumAtoms() - 1)
    for idx in range(mol.GetNumAtoms()):
        if idx != atom_ids_to_remove:
            pos = conf.GetAtomPosition(idx)
            new_conf.SetAtomPosition(
                idx if idx < atom_ids_to_remove else idx - 1, pos)

    mw.RemoveAtom(atom_ids_to_remove)
    # Update the conformer of the modified molecule
    mw.RemoveAllConformers()
    mw.AddConformer(new_conf, assignId=True)

    return mw


def wash_molecule(fn):
    subprocess.run(f'antechamber -i {fn} -fi sdf -o {fn.replace(".sdf","_1.sdf")} -fo sdf -rn UNK -at gaff2 -an yes -dr no -pf yes', stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    print(fn)
    try:
        mol = Chem.MolFromMolFile(
            fr'{fn}', removeHs=False)
        w = Chem.SDWriter(
            fr'{fn.replace(".sdf","_2.sdf")}')
        w.write(mol)

    except:
        # print(i)
        mol = Chem.MolFromMolFile(
            fr'{fn.replace(".sdf","_1.sdf")}', sanitize=False)
        atom_id = find_bad_C(mol)
        print(mol)

        if len(atom_id) > 0:
            mw = Chem.RWMol(mol)
            for atom_ids in atom_id:
                to_delete_1, to_delete_2 = find_which_to_delete(
                    mol, atom_ids)
                mw.RemoveBond(to_delete_1, to_delete_2)
                mw.AddBond(to_delete_1, to_delete_2, Chem.BondType.SINGLE)
                mol = mw.GetMol()
                to_delete_1, to_delete_2 = find_which_to_delete_H(
                    mol, atom_ids)
                print(to_delete_1, to_delete_2)
                mw = remove_atoms_and_update_conformer(mol, to_delete_2)
            mol = mw.GetMol()

        atom_id = find_bad_N(mol)
        if len(atom_id) > 0:
            mw = Chem.RWMol(mol)
            # if no hydrogen on it to be delete, set the ExplicitValence of N to 4 
            for atom_ids in atom_id:
                to_delete_1, to_delete_2 = find_which_to_delete(mol, atom_ids)
                if mol.GetAtomWithIdx(to_delete_2).GetAtomicNum() == 1:
                    mw = remove_atoms_and_update_conformer(mol, to_delete_2)
                    mol = mw.GetMol()
                else:
                    try:
                        mol.GetAtomWithIdx(to_delete_1).SetFormalCharge(1)
                        Chem.SanitizeMol(mol)
                    except:
                        # print(i,'enter this')
                        for atom in mol.GetAtoms():
                            if atom.GetSymbol() == 'N' and atom.GetExplicitValence() > 3:
                                atom.SetFormalCharge(1)

        try:
            Chem.SanitizeMol(mol)
        except:
            atom_id = find_bad_N(mol)
            if len(atom_id) > 0:
                mw = Chem.RWMol(mol)
                for atom_ids in atom_id:
                    try:
                        to_delete_1, to_delete_2 = find_which_to_delete_D_to_S(
                            mol, atom_ids)
                        mw.RemoveBond(to_delete_1, to_delete_2)
                        mw.AddBond(to_delete_1, to_delete_2,
                                   Chem.BondType.SINGLE)
                    except:
                        # print(i)
                        to_delete_1, to_delete_2 = find_which_to_delete_H(
                            mol, atom_ids)
                        mw = remove_atoms_and_update_conformer(
                            mol, to_delete_2)
                mol = mw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            atom_id = find_bad_S(mol)
            if len(atom_id) > 0:
                mw = Chem.RWMol(mol)
                for atom_ids in atom_id:
                    to_delete_1, to_delete_2 = find_which_to_delete_S(
                        mol, atom_ids)
                    mw.RemoveBond(to_delete_1, to_delete_2)
                    mw.AddBond(to_delete_1, to_delete_2, Chem.BondType.SINGLE)
                mol = mw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            atom_id = find_bad_F(mol)
            if len(atom_id) > 0:
                mw = Chem.RWMol(mol)
                for atom_ids in atom_id:
                    to_delete_1, to_delete_2 = find_which_to_delete_F(
                        mol, atom_ids)
                    mw.RemoveBond(to_delete_1, to_delete_2)
                mol = mw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            print(i)
            # break
        w = Chem.SDWriter(
        fr'{fn.replace(".sdf","_2.sdf")}', removeHs=False)
        w.write(mol)
    return fn.replace(".sdf","_2.sdf")