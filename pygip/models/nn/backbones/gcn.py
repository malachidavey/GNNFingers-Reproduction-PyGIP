import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn.functional as F

def drop_edge(g, drop_rate=0.2):
    import dgl
    num_edges = g.num_edges()
    mask = torch.rand(num_edges) > drop_rate
    src, dst = g.edges()
    src = src[mask]
    dst = dst[mask]
    new_g = dgl.graph((src, dst), num_nodes=g.num_nodes())
    new_g = dgl.add_self_loop(new_g)
    return new_g

class GCN(nn.Module):
    def __init__(self, feature_number, label_number, hidden_dim=64):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList([
        dglnn.GraphConv(feature_number, hidden_dim, activation=F.relu, allow_zero_in_degree=True),
        dglnn.GraphConv(hidden_dim, label_number, allow_zero_in_degree=True)
        ])
        


    def forward(self, g, x):
        """Standard forward pass."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
        return h

    def get_embeddings(self, g, x):
        """Return penultimate embeddings for GraphMI's GAE stage."""
        import dgl
        import torch

        # Handle adjacency input
        if isinstance(g, torch.Tensor):
            # If g is an adjacency matrix [N, N], build a DGLGraph
            if g.ndim == 2:
                src, dst = torch.nonzero(g > 0, as_tuple=True)
                g = dgl.graph((src, dst))
            g = dgl.add_self_loop(g)

        # If x is [N, F], ensure F matches input layer
        if x is None or x.ndim != 2:
            raise ValueError("x must be a 2D feature matrix [N, F].")

        h = x
        # Forward through all but last layer to get penultimate embeddings
        for layer in self.layers[:-1]:
            h = F.relu(layer(g, h))
        return h

