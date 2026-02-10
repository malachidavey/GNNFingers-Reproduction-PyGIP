import torch
import dgl
from torch_geometric.datasets import TUDataset

def load_tu_as_dgl(name, root="./data"):
    """
    Loads TU dataset using PyG and converts to DGL graphs.
    Stores graph-level labels at g.graph_data['label'].
    """
    pyg_ds = TUDataset(root=root, name=name)
    dgl_graphs = []

    for data in pyg_ds:
        # Create graph
        g = dgl.graph(
            (data.edge_index[0], data.edge_index[1]),
            num_nodes=data.num_nodes
        )
        g = dgl.add_self_loop(g)

        # Node features
        if data.x is not None:
            g.ndata["feat"] = data.x.float()
        else:
            g.ndata["feat"] = torch.ones((data.num_nodes, 1))

        # GRAPH-LEVEL label
        graph_label = data.y.squeeze()          # e.g., tensor(3)
        g.graph_data = {"label": graph_label}   # store at graph level

        dgl_graphs.append(g)

    return dgl_graphs, pyg_ds
