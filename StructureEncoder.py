import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from Bio.PDB import PDBParser
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from config import Config


AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


class StructureEncoder(nn.Module):
    """
    PDB → Residue Graph → GCN → Structure Embedding

    Input:
        pdb_paths: List[str]

    Output:
        {
            'token_embeddings': node_embeddings,
            'projected_tokens': projected_tokens,
            'structure_embedding': graph_embedding,
            'attention_mask': attention_mask
        }
    """

    def __init__(self):
        super().__init__()

        self.parser = PDBParser(QUIET=True)

        # Node feature dimension:
        # 20 AA one-hot + 3 coordinates
        input_dim = 23

        self.gcn1 = GCNConv(input_dim, 128)
        self.gcn2 = GCNConv(128, 256)
        self.gcn3 = GCNConv(256, Config.SHARED_DIM)

        self.dropout = nn.Dropout(Config.DROPOUT)

    def _residue_to_feature(self, residue):
        aa = residue.get_resname()
        aa_onehot = torch.zeros(20, dtype=torch.float)

        if aa in AA_TO_IDX:
            aa_onehot[AA_TO_IDX[aa]] = 1.0

        if 'CA' in residue:
            coord = torch.tensor(
                residue['CA'].get_coord(),
                dtype=torch.float
            )
        else:
            coord = torch.zeros(3, dtype=torch.float)

        feature = torch.cat([aa_onehot, coord], dim=0)
        return feature

    def _build_graph(self, pdb_path: str) -> Data:
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f'PDB file not found: {pdb_path}')

        structure = self.parser.get_structure('protein', pdb_path)

        residues = []
        coords = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in AA_TO_IDX and 'CA' in residue:
                        residues.append(residue)
                        coords.append(torch.tensor(
                            residue['CA'].get_coord(),
                            dtype=torch.float
                        ))

        if len(residues) == 0:
            # Create a dummy node to avoid crashes
            x = torch.zeros((1, 23), dtype=torch.float)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            return Data(x=x, edge_index=edge_index)

        x = torch.stack(
            [self._residue_to_feature(r) for r in residues]
        )

        coords = torch.stack(coords)

        edge_list = []
        num_nodes = len(coords)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distance = torch.norm(coords[i] - coords[j]).item()
                if distance < 8.0:
                    edge_list.append([i, j])
                    edge_list.append([j, i])

        if len(edge_list) == 0:
            edge_list = [[0, 0]]

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)

    def forward(self, pdb_paths: List[str]):

        device = next(self.parameters()).device

        ####################################################
        # Build Graphs
        ####################################################

        graphs = [self._build_graph(p) for p in pdb_paths]

        batch_graph = Batch.from_data_list(graphs).to(device)

        x = batch_graph.x
        edge_index = batch_graph.edge_index
        batch = batch_graph.batch

        ####################################################
        # GCN
        ####################################################

        x = F.relu(self.gcn1(x, edge_index))
        x = self.dropout(x)

        x = F.relu(self.gcn2(x, edge_index))
        x = self.dropout(x)

        node_embeddings = self.gcn3(x, edge_index)

        ####################################################
        # Graph Embedding
        ####################################################

        graph_embedding = global_mean_pool(
            node_embeddings,
            batch
        )

        ####################################################
        # Padding Tokens
        ####################################################

        batch_size = len(graphs)

        node_counts = []

        max_nodes = 0

        for graph in graphs:

            n = graph.x.size(0)

            node_counts.append(n)

            if n > max_nodes:
                max_nodes = n

        hidden_dim = node_embeddings.size(-1)

        projected_tokens = torch.zeros(
            batch_size,
            max_nodes,
            hidden_dim,
            device=device
        )

        attention_mask = torch.zeros(
            batch_size,
            max_nodes,
            dtype=torch.long,
            device=device
        )

        start = 0

        for i, n in enumerate(node_counts):

            projected_tokens[i, :n] = node_embeddings[start:start+n]

            attention_mask[i, :n] = 1

            start += n

        ####################################################
        # Output
        ####################################################

        return {

            "token_embeddings": projected_tokens,

            "projected_tokens": projected_tokens,

            "structure_embedding": graph_embedding,

            "attention_mask": attention_mask

        }