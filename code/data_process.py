# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.data import Data
from utils_ import parameters_set
args = parameters_set()

def data_lode():

    'Read the initial features files'
    mirna_feat = np.loadtxt(args.miRNA_feature) # miRNA kmer feature
    drug_feat = np.loadtxt(args.drug_feature)  # drug maccs feature
    disease_feat = np.loadtxt(args.disease_feature) # disease sem similarity

    mirna_features = torch.FloatTensor(mirna_feat)
    drug_features = torch.FloatTensor(drug_feat)
    disease_features = torch.FloatTensor(disease_feat)


    features = {'mi': mirna_features, 'dr': drug_features, 'di': disease_features}

    in_size = {'mi': mirna_features.shape[1], 'dr': drug_features.shape[1],
               'di': disease_features.shape[1]}
    # print('in_size:', mirna_features.shape[1], drug_features.shape[1], disease_features.shape[1])
    return features, in_size


# miRNA sequence encoding
def encode_sequence(sequence, max_len):

    mapping = {'A': 1, 'U': 2, 'C': 3, 'G': 4}
    encoded = [mapping.get(base.upper(), 0) for base in sequence]

    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded.extend([0] * (max_len - len(encoded)))
    return torch.tensor(encoded, dtype=torch.long)

def load_mirna_data(max_len):
    df = pd.read_excel(args.miRNA_sequence)
    encoded = []
    for idx, row in df.iterrows():
        seq = row['Sequence']
        enc = encode_sequence(seq, max_len=max_len)
        encoded.append(enc)
    seq_tensor = torch.stack(encoded, dim=0)
    return seq_tensor


# drug smiles encoding
def one_of_k_encoding_unk(value, choices):
    if value not in choices:
        value = choices[-1]
    return [int(value == item) for item in choices]

def smiles_to_graph(smiles, max_num_nodes):
    """
    Convert SMILES to a torch_geometric.data.Data object.
    If conversion fails, return None.
    """
    atom_types = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
        'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
        'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
        'Pt', 'Hg', 'Pb', 'Unknown'
    ]
    hybrid_types = [
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
        'other'
    ]
    degrees = [0, 1, 2, 3, 4, 5]
    formal_charges = [-2, -1, 0, 1, 2]
    num_hs = [0, 1, 2, 3, 4]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atom_type_enc = one_of_k_encoding_unk(atom.GetSymbol(), atom_types)
        hyb = atom.GetHybridization()
        hyb_val = hyb if hyb in hybrid_types[:-1] else 'other'
        hybrid_enc = one_of_k_encoding_unk(hyb_val, hybrid_types)
        degree_enc = one_of_k_encoding_unk(atom.GetDegree(), degrees)
        charge_enc = one_of_k_encoding_unk(atom.GetFormalCharge(), formal_charges)
        num_h_enc = one_of_k_encoding_unk(atom.GetTotalNumHs(), num_hs)
        aromatic = [int(atom.GetIsAromatic())]
        radical = [atom.GetNumRadicalElectrons()]
        in_ring = [int(atom.IsInRing())]
        feats = atom_type_enc + hybrid_enc + degree_enc + charge_enc + num_h_enc + aromatic + radical + in_ring
        atom_features.append(feats)

    num_atoms = len(atom_features)
    if num_atoms == 0:
        return None

    feat_size = len(atom_features[0])
    if num_atoms < max_num_nodes:
        padding = [[0] * feat_size for _ in range(max_num_nodes - num_atoms)]
        atom_features.extend(padding)
    else:
        atom_features = atom_features[:max_num_nodes]

    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])

    x = torch.tensor(np.array(atom_features), dtype=torch.float)
    if len(edge_indices) == 0:
        # create dummy self-loop if there are no bonds
        edge_index = torch.tensor([list(range(x.size(0))), list(range(x.size(0)))], dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data

def load_drug_data(max_num_nodes):
    df = pd.read_excel(args.drug_smiles)
    graphs = []
    for idx, row in df.iterrows():
        smiles = str(row['SMILES']).strip() if 'SMILES' in row.index else ''
        g = smiles_to_graph(smiles, max_num_nodes)
        graphs.append(g)
    return graphs

# disease similarity encoding
def load_disease_data():
    """Load disease semantic vectors - return tensor and valid indices"""
    arr = np.loadtxt(args.disease_similarity)
    tensor = torch.FloatTensor(arr)
    return tensor

def load_all(max_seq_len=24, max_atom_len=100):
    features, in_size = data_lode()
    mi_seq_tensor = load_mirna_data(max_len=max_seq_len)
    dr_graphs = load_drug_data(max_num_nodes=max_atom_len)
    di_attr_tensor = load_disease_data() #

    return {
        'features': features,
        'in_size': in_size,
        'mi_seq_tensor': mi_seq_tensor,
        'dr_graphs': dr_graphs,
        'di_attr_tensor': di_attr_tensor,
    }
