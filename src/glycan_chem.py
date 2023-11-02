from rdkit import Chem

def has_one_oxygen_and_sp3_carbons_in_ring(m, ring_elements):
    '''has one oxygen and sp3 carbons in ring'''
    cnt   = 0
    cflag = 1
    hflag = 1
    for i in range(len(ring_elements)):
        idx = ring_elements[i]
        atom = m.GetAtomWithIdx(idx)
        atom_type = atom.GetSymbol()
        hybrid    = str(atom.GetHybridization())
        # one oxygen
        if (atom_type == 'O') or (atom_type == "o"):
           cnt = cnt + 1
        elif (atom_type == 'C') or (atom_type == "c"):
           # sp3 carbons
           if hybrid == "UNSPECIFIED":
              neighbors = [x.GetAtomicNum() for x in atom.GetNeighbors()]
              neighbor_type = [x.GetSymbol() for x in atom.GetNeighbors()]
              if len(neighbors) == 4:
                  hybrid = "SP3"
           if hybrid != "SP3":
              hflag = 0
              break
        else:
           cflag = 0
           break
    if (cflag == 0) or (hflag == 0):
       return False
    if cnt == 1:
       return True
    return False

def is_fused_ring(m, ring_elements):
    '''check if this ring is fused '''
    flag = 0
    ssr = Chem.GetSymmSSSR(m)
    for i in range(len(ssr)):
        ssr_list = list(ssr[i])
        if ssr_list == ring_elements:
           continue
        intersection = set(ssr_list) & set(ring_elements)
        if len(intersection):
           flag = 1
           break
    if flag:
       return True
    return False

def get_hydroxyl_branch(m, ring_elements):
    '''get hydroxyl oxygen branch of ring'''
    branch_atoms = set()
    # identify ring oxygen
    ring_oxygen = []
    for i in range(len(ring_elements)):
        atom = m.GetAtomWithIdx(ring_elements[i])
        if (atom.GetSymbol() == 'O') or (atom.GetSymbol() == 'o'):
           ring_oxygen.append(atom.GetIdx())
    if len(ring_oxygen) < 1:
        return ring_oxygen
    # identify branch oxygen
    branch_oxygen = []
    for i in range(len(ring_elements)):
        atom = m.GetAtomWithIdx(ring_elements[i])
        nbr  = atom.GetNeighbors()
        for j in range(len(nbr)):
            nbr_idx = nbr[j].GetIdx()
            if nbr_idx in ring_oxygen:
               continue
            if (nbr[j].GetSymbol() == 'O') or (nbr[j].GetSymbol() == 'o'):
               branch_oxygen.append(nbr_idx)
    if len(branch_oxygen) < 1:
        return branch_oxygen
    # identify branch hydroxyl oxygen
    branch_hydroxyl = []
    for i in range(len(branch_oxygen)):
        atom = m.GetAtomWithIdx(branch_oxygen[i])
        nbr  = atom.GetNeighbors()
        for j in range(len(nbr)):
            nbr_idx = nbr[j].GetIdx()
            if (nbr[j].GetSymbol() == 'H') or (nbr[j].GetSymbol() == 'h'):
               branch_hydroxyl.append(nbr_idx)
               branch_atoms.add(branch_oxygen[i])
    if len(branch_hydroxyl) < 1:
        return branch_hydroxyl
    return list(branch_atoms)

def get_glycoside_group_idx(m):
    '''identify glycoside group atom index'''
    glycan_ring_idx = get_glycan_ring_idx(m)
    # identify carbon from CH2OH branch
    glycan_base_idx = []
    CH2O_carbon = [ -1 ] * len(glycan_ring_idx)
    cflag  = 0   # if find the carbon from CH2OH branch 
    for i in range(len(glycan_ring_idx)):
        for j in range(len(glycan_ring_idx[i])):
            idx = glycan_ring_idx[i][j]
            atom = m.GetAtomWithIdx(idx)
            nbr  = atom.GetNeighbors()
            if len(nbr) < 1:
                continue
            for k in range(len(nbr)):
                nbr_idx = nbr[k].GetIdx()
                # skip atom in glycan ring
                if nbr_idx in glycan_ring_idx[i]:
                    continue
                if (nbr[k].GetSymbol() == 'C') or (nbr[k].GetSymbol() == 'c'):
                    CH2O_carbon[i] = nbr_idx
                    cflag = 1
                    break
        idx_list = list(set(glycan_ring_idx[i]))
        if not (CH2O_carbon[i] == -1):
            idx_list.append(CH2O_carbon[i])
        glycan_base_idx.append(idx_list)

    # identify neigbors of glycan base
    glycoside_group_idx = []
    for i in range(len(glycan_base_idx)):
        neigbors_idx = set()
        for j in range(len(glycan_base_idx[i])):
            idx = glycan_base_idx[i][j]
            atom = m.GetAtomWithIdx(idx)
            nbr  = atom.GetNeighbors()
            if len(nbr) < 1:
                continue
            for k in range(len(nbr)):
                nbr_idx = nbr[k].GetIdx()
            neigbors_idx.add(nbr_idx)
        idx_list = list(neigbors_idx.union(set(glycan_base_idx[i])))
        glycoside_group_idx.append(idx_list)
    return glycoside_group_idx

def get_glycan_ring_idx(m):
    '''identify glycan ring from molecule'''
    glycan_ring = []
    # get rings
    ssr = Chem.GetSymmSSSR(m)
    if len(ssr) < 1:   # no rings
       return glycan_ring
    for i in range(len(ssr)):
        ssr_list = list(ssr[i])
        # penta or hexa ring
        if (len(ssr_list) == 5) or (len(ssr_list) == 6):
           # must has one oxygen and carbons in ring
           if not has_one_oxygen_and_sp3_carbons_in_ring(m, ssr_list):
              continue
           # exclude fused ring
           if (len(ssr) > 1) and is_fused_ring(m, ssr_list):
              continue
           # must has hydroxyl branch
           if not get_hydroxyl_branch(m, ssr_list):
              continue 
           glycan_ring.append(ssr_list)
    return glycan_ring

def get_glycan_ring(sdf_file_name):
    '''identify glycan ring'''
    glycan_ring = []
    m = Chem.MolFromMolFile(sdf_file_name, removeHs=False)
    if m is None:
        pdb_file_name = sdf_file_name.replace("_ligand.sdf", "_ligand.pdb")
        m1 = Chem.MolFromPDBFile(pdb_file_name, removeHs=False) 
        if m1 is None:
           return glycan_ring 
        else:
           m = m1
    # get rings
    ssr = Chem.GetSymmSSSR(m)
    if len(ssr) < 1:   # no rings
        return glycan_ring
    for i in range(len(ssr)):
        ssr_list = list(ssr[i])
        # penta or hexa ring
        if (len(ssr_list) == 5) or (len(ssr_list) == 6):
           # must has one oxygen and carbons in ring
           if not has_one_oxygen_and_sp3_carbons_in_ring(m, ssr_list):
              continue
           # exclude fused ring
           if (len(ssr) > 1) and is_fused_ring(m, ssr_list):
              continue
           # must has hydroxyl branch
           if not get_hydroxyl_branch(m, ssr_list):
              continue 
           glycan_ring.append(ssr_list)
    return glycan_ring
