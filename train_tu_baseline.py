#!/usr/bin/env python3
import os, argparse, random, numpy as np, torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, global_mean_pool
from pygip.datasets.datasets import ENZYMES, PROTEINS, AIDS

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=6, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin   = nn.Linear(hidden, out_dim)
        self.do = nn.Dropout(dropout); self.act = nn.ReLU()
    def forward(self, x, ei, batch):
        x = self.act(self.conv1(x, ei))
        x = self.do(x)
        x = self.conv2(x, ei)
        x = global_mean_pool(x, batch)
        return self.lin(x)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=6, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.lin   = nn.Linear(hidden, out_dim)
        self.do = nn.Dropout(dropout); self.act = nn.ReLU()
    def forward(self, x, ei, batch):
        x = self.act(self.conv1(x, ei))
        x = self.do(x)
        x = self.conv2(x, ei)
        x = global_mean_pool(x, batch)
        return self.lin(x)

class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=6, dropout=0.5):
        super().__init__()
        mlp1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        mlp2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)
        self.lin = nn.Linear(hidden, out_dim)
        self.do = nn.Dropout(dropout)
        self.act = nn.ReLU()
    def forward(self, x, ei, batch):
        x = self.act(self.conv1(x, ei))
        x = self.do(x)
        x = self.conv2(x, ei)
        x = global_mean_pool(x, batch)
        return self.lin(x)

def get_dataset(name):
    name = name.upper()
    if name == "ENZYMES":   return ENZYMES(api_type="pyg", path="./data")
    if name == "PROTEINS":  return PROTEINS(api_type="pyg", path="./data")
    if name == "AIDS":      return AIDS(api_type="pyg", path="./data")
    raise ValueError(f"Unknown TU dataset: {name}")

def build_model(arch, in_dim, out_dim):
    arch = arch.upper()
    if arch == "GCN":       return GCN(in_dim, 64, out_dim)
    if arch == "GIN":       return GIN(in_dim, 64, out_dim)
    if arch == "GRAPHSAGE": return GraphSAGE(in_dim, 64, out_dim)
    raise ValueError(f"Unknown arch: {arch}")

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval(); correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x.float(), batch.edge_index, batch.batch)
        pred = logits.argmax(dim=-1)
        correct += int((pred == batch.y).sum().item())
        total += batch.y.size(0)
    return correct / max(total, 1)

def train_one(ds_name, arch, epochs, bs, lr, seed, out_dir):
    set_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = get_dataset(ds_name)
    train_loader = DataLoader(ds.train_data, batch_size=bs, shuffle=True)
    test_loader  = DataLoader(ds.test_data,  batch_size=bs, shuffle=False)
    sample = ds.train_data[0]
    in_dim = int(sample.x.size(-1)) if getattr(sample, "x", None) is not None else 1
    out_dim = int(ds.graph_dataset.num_classes)
    model = build_model(arch, in_dim, out_dim).to(dev)
    opt = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, epochs+1):
        model.train()
        for batch in train_loader:
            batch = batch.to(dev)
            opt.zero_grad()
            logits = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y.long())
            loss.backward(); opt.step()
        acc = eval_model(model, test_loader, dev)
        if acc > best:
            best = acc
            os.makedirs(out_dir, exist_ok=True)
            ck = os.path.join(out_dir, f"{ds_name.lower()}_{arch.lower()}_s{seed}.pt")
            torch.save(model.state_dict(), ck)
        if ep % 10 == 0 or ep == 1:
            print(f"[{ds_name}-{arch}] epoch {ep}/{epochs} acc={acc:.4f} best={best:.4f}")

    with open(os.path.join(out_dir, f"{ds_name.lower()}_{arch.lower()}_s{seed}.metrics.txt"), "w") as f:
        f.write(f"best_acc={best:.6f}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ENZYMES","PROTEINS","AIDS"])
    ap.add_argument("--arch", default="GCN", choices=["GCN","GIN","GraphSAGE"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results_tu")
    args = ap.parse_args()
    train_one(args.dataset, args.arch, args.epochs, args.batch_size, args.lr, args.seed, args.out_dir)
