import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from sklearn.metrics import roc_auc_score, average_precision_score

from pygip.datasets.dgl_tu_loader import load_tu_as_dgl


# -------------------------------------------------
# DGL GCN (must match train_dgl_tu_gcn.py)
# -------------------------------------------------
class DGL_GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, activation=F.relu, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, activation=F.relu, allow_zero_in_degree=True)
        self.lin = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, g, x):
        h = self.conv1(g, x)
        h = self.conv2(g, h)
        g.ndata["h"] = h
        hg = dgl.mean_nodes(g, "h")
        return self.lin(hg)

    def penultimate(self, g, x):
        h = self.conv1(g, x)
        h = self.conv2(g, h)
        return h


# -------------------------------------------------
# GraphMI Attack (uses your class)
# -------------------------------------------------
from pygip.attacks.graphmi_attack import GraphMIAttack


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
def eval_reconstruction(A_pred, A_true):
    N = A_true.shape[0]
    y_true, y_score = [], []

    for i in range(N):
        for j in range(i+1, N):
            y_true.append(int(A_true[i, j]))
            y_score.append(float(A_pred[i, j]))

    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.0001)
    parser.add_argument("--K", type=int, default=20)
    args = parser.parse_args()

    # -----------------------------
    # Load DGL TU Dataset
    # -----------------------------
    print(f"\nLoading dataset '{args.dataset}'...")
    graphs, ds_info = load_tu_as_dgl(args.dataset)
    g = graphs[0]                    # use first graph to match paper setting
    g = dgl.add_self_loop(g)

    X = g.ndata["feat"].float()
    Y = torch.tensor(int(g.graph_data["label"]), dtype=torch.long)
    Y = Y.unsqueeze(0)               # make shape [1]

    # Safe fallback: build adjacency manually (works on macOS)
    src, dst = g.edges()
    N = g.num_nodes()
    A_true = torch.zeros((N, N), dtype=torch.float32)
    A_true[src, dst] = 1
    A_true[dst, src] = 1

    print(f"Loaded {args.dataset}: Nodes={X.shape[0]}, Features={X.shape[1]}, Label={Y.item()}")

    # -----------------------------
    # Prepare Model
    # -----------------------------
    in_dim = X.shape[1]
    out_dim = ds_info.num_classes

    model = DGL_GCN(in_dim, 64, out_dim)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()

    # -----------------------------
    # Density estimate (correct formula)
    # -----------------------------
    E = float(A_true.sum().item() / 2)
    possible = (A_true.shape[0] * (A_true.shape[0] - 1)) / 2
    density = E / possible

    print(f"Estimated graph density: {density:.4f}")

    # -----------------------------
    # Run GraphMI Attack
    # -----------------------------
    print("\nRunning GraphMI...")
    attack = GraphMIAttack(
        model=model,
        X=X,
        Y=Y,
        device="cpu",
        alpha=args.alpha,
        beta=args.beta,
        lr=args.lr,
        iters=args.iters,
        K=args.K,
        est_density=density,
    )

    t0 = time.time()
    A_rec = attack.run()
    t1 = time.time()

    # -----------------------------
    # Evaluate
    # -----------------------------
    auc, ap = eval_reconstruction(A_rec, A_true.numpy())
    print(f"\nDone. AUC={auc:.4f}  AP={ap:.4f}  Time={t1 - t0:.2f}s")

    # -----------------------------
    # Save Results
    # -----------------------------
    os.makedirs("results_graphmi", exist_ok=True)
    out = {
        "dataset": args.dataset,
        "ckpt": args.ckpt,
        "AUC": auc,
        "AP": ap,
        "time": t1 - t0,
        "iters": args.iters,
        "K": args.K,
        "density": density,
    }
    fname = f"{args.dataset}_graphmi_results.json"
    with open(os.path.join("results_graphmi", fname), "w") as f:
        json.dump(out, f, indent=2)

    print("Saved:", os.path.join("results_graphmi", fname))


if __name__ == "__main__":
    main()
