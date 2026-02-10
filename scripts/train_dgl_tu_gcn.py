import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from pygip.datasets.dgl_tu_loader import load_tu_as_dgl
import dgl
import dgl.nn.pytorch as dglnn

# -------------------------
# GCN Model
# -------------------------
class DGL_GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, activation=F.relu)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, activation=F.relu)
        self.lin   = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, x, dropedge_rate=0.2):
    # ---- DROPPEDGES DEFENSE ----
        if dropedge_rate > 0:
            # randomly drop edges
            num_edges = g.num_edges()
            keep = int(num_edges * (1 - dropedge_rate))

            # random permutation of all edges
            perm = torch.randperm(num_edges, device=g.device)
            keep_idx = perm[:keep]

            # build masked graph
            src, dst = g.edges()
            g = dgl.graph((src[keep_idx], dst[keep_idx]), num_nodes=g.num_nodes())
            g = dgl.add_self_loop(g)
    # ----------------------------------

        h = self.conv1(g, x)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.lin(hg)
    def penultimate(self, g, x):
        h = self.conv1(g, x)
        h = self.conv2(g, h)
        return h

# -------------------------
# Training
# -------------------------
def train_one(dataset_name, epochs, lr, hidden_dim, out_dir, dropedge_rate, featmask, dp_sigma):
    graphs, pyg_ds = load_tu_as_dgl(dataset_name)

    num_classes = pyg_ds.num_classes
    in_dim = graphs[0].ndata['feat'].shape[1]

    model = DGL_GCN(in_dim, hidden_dim, num_classes)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0

        for g in graphs:
            x = g.ndata['feat']
            # ---- FEATURE MASKING DEFENSE ----
            if featmask > 0:
                mask = (torch.rand_like(x) > featmask).float()
                x = x * mask
            # ---------------------------------
            # ----- DP-NOISE ----------------
            if dp_sigma > 0:
                x = x + torch.randn_like(x) * dp_sigma
            #------------------------------------
            logits = model(g, g.ndata['feat'], dropedge_rate=dropedge_rate)
            label = g.graph_data["label"].unsqueeze(0)

            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} Loss: {total_loss:.4f}")

    suffix = ""
    if dropedge_rate > 0:
        suffix += f"_dropedge{dropedge_rate}"
    if featmask > 0:
        suffix += f"_featmask{featmask}"
    if dp_sigma > 0:
        suffix += f"_dpsigma{dp_sigma}"
    ckpt_path = os.path.join(out_dir, f"{dataset_name.lower()}_gcn{suffix}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print("Saved:", ckpt_path)



# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--out_dir", default="dgl_checkpoints")
    parser.add_argument("--dropedge", type=float, default=0.0, help="DropEdge probability")
    parser.add_argument("--featmask", type=float, default=0.0, help="Random node feature masking rate")
    parser.add_argument("--dp_sigma", type=float, default=0.0, help="Gaussian noise scale for DP-style feature noise")

    args = parser.parse_args()

    train_one(args.dataset, args.epochs, args.lr, args.hidden_dim, args.out_dir, args.dropedge, args.featmask, args.dp_sigma)
