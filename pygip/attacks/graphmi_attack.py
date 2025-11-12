"""
GraphMI white-box attack for PyGIP.

Place in: pygip/attacks/graphmi_attack.py

Notes:
- This attempts to be plug-and-play with common PyGIP conventions:
  - Model is expected to be callable as model(X, A_prob) -> logits
  - Model should expose embeddings via model.penultimate(X, A) or model.get_embeddings(X, A)
- If your model API differs, search for the TODO markers in this file and adapt.
"""
import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

# -------------------------
# Utilities
# -------------------------
def upper_tri_indices(N, device=None):
    rows = []
    cols = []
    for i in range(N):
        for j in range(i+1, N):
            rows.append(i)
            cols.append(j)
    rows = torch.tensor(rows, dtype=torch.long, device=device)
    cols = torch.tensor(cols, dtype=torch.long, device=device)
    return rows, cols

def vec_to_sym_adj(vec, N, device=None):
    """Convert vector of length n = N*(N-1)/2 to NxN symmetric matrix."""
    if device is None:
        device = vec.device
    A = torch.zeros((N, N), device=device)
    rows, cols = upper_tri_indices(N, device=device)
    A[rows, cols] = vec
    A[cols, rows] = vec
    return A

def sym_adj_to_vec(A):
    N = A.shape[0]
    parts = []
    for i in range(N):
        for j in range(i+1, N):
            parts.append(A[i, j])
    return torch.stack(parts)

def feature_smoothness_loss(A, X, eps=1e-9):
    """tr(X^T Lhat X) where Lhat = I - D^{-1/2} A D^{-1/2}"""
    deg = A.sum(dim=1).clamp(min=eps)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))
    Lhat = torch.eye(A.size(0), device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
    XT = X.transpose(0, 1)
    t = torch.trace(XT @ Lhat @ X)
    return t

# -------------------------
# GraphMI attack class
# -------------------------
class GraphMIAttack:
    def __init__(self, model, X, Y, A_true=None, device='cpu',
                 alpha=0.001, beta=0.0001, lr=0.1, iters=100, K=20, est_density=None):
        self.model = model.to(device).eval()
        self.device = device
        self.X = X.to(device)
        self.Y = Y.to(device)
        self.A_true = A_true  # optional, for density estimation / eval
        self.N = X.shape[0]
        self.n_vars = self.N * (self.N - 1) // 2
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.iters = iters
        self.K = K
        # density estimate: fraction of edges among possible undirected pairs
        if est_density is not None:
            self.r = est_density
        elif A_true is not None:
            # A_true expected binary NxN numpy or tensor
            total_pairs = float(self.N * (self.N - 1) / 2)
            if isinstance(A_true, torch.Tensor):
                edge_count = A_true.sum().item() / 2.0 if A_true.dim() == 2 else float(A_true.sum())
            else:
                edge_count = float(A_true.sum()) / 2.0 if hasattr(A_true, 'shape') and A_true.ndim == 2 else float(A_true.sum())
            self.r = max(1e-6, edge_count / total_pairs)
        else:
            self.r = 0.1
        # precompute upper-tri indices
        self.rows, self.cols = upper_tri_indices(self.N, device=self.device)

    def init_vector(self, init='zeros'):
        if init == 'zeros':
            v = torch.zeros(self.n_vars, device=self.device, requires_grad=True)
        elif init == 'uniform':
            v = torch.rand(self.n_vars, device=self.device, requires_grad=True)
        else:
            raise ValueError("init must be 'zeros' or 'uniform'")
        return v

    def attack_loss(self, a_vec):
        # build symmetric adjacency probability matrix
        A_prob = vec_to_sym_adj(a_vec, self.N, device=a_vec.device)
        A_prob = torch.clamp(A_prob, 0.0, 1.0)

        # Convert adjacency probabilities -> DGLGraph expected by model
        # (threshold at 0.5 for constructing graph edges; model forward uses graph)
        with torch.no_grad():
            A_bin = (A_prob > 0.5).float()
        src, dst = torch.nonzero(A_bin, as_tuple=True)
        try:
            import dgl
        except Exception:
            raise RuntimeError("dgl not available in environment; install dgl to run this attack with your model.")

        # build graph and add self loops (common for GCN implementations)
        g = dgl.graph((src, dst), num_nodes=A_bin.shape[0])
        g = dgl.add_self_loop(g)

        # forward through model: model expects (graph, features)
        logits = self.model(g, self.X)

        # cross-entropy on logits vs true labels
        ce = F.cross_entropy(logits, self.Y, reduction='mean')

        # smoothness on A_prob (feature smoothness regularizer)
        Ls = feature_smoothness_loss(A_prob, self.X)

        # l2 regularization on the vector of probabilities
        reg = torch.norm(a_vec, p=2)**2

        loss = ce + self.alpha * Ls + self.beta * reg
        return loss, A_prob, logits


    def projected_gradient_descent(self, v):
        optimizer = torch.optim.SGD([v], lr=self.lr)
        for t in range(self.iters):
            optimizer.zero_grad()
            loss, A_prob, _ = self.attack_loss(v)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                v.clamp_(0.0, 1.0)
            if (t+1) % max(1, self.iters//10) == 0:
                with torch.no_grad():
                    deg_mean = A_prob.sum().item() / self.N
                print(f"[PGD] iter {t+1}/{self.iters} loss={loss.item():.6f} deg_mean={deg_mean:.3f}")
        return v.detach()

    def graph_autoencoder_postprocess(self, a_vec):
        """Take current vector -> form A_prob -> get embeddings Z from model -> compute sigmoid(ZZ^T)"""
        A_prob = vec_to_sym_adj(a_vec, self.N, device=a_vec.device)
        # Model must provide embeddings — try common method names
        if hasattr(self.model, "penultimate"):
            Z = self.model.penultimate(self.X, A_prob)
        elif hasattr(self.model, "get_embeddings"):
            Z = self.model.get_embeddings(self.X, A_prob)
        else:
            # As fallback, try forward returning embeddings
            try:
                logits, Z = self.model(self.X, A_prob, return_embeddings=True)
            except TypeError:
                raise RuntimeError("Model does not expose embeddings. Add .penultimate(X,A) or .get_embeddings(X,A) or forward(..., return_embeddings=True).")
        with torch.no_grad():
            A_hat = torch.sigmoid(Z @ Z.t())
        return sym_adj_to_vec(A_hat)

    def random_sampling(self, a_vec, K=None, density=None):
        if K is None: K = self.K
        if density is None: density = self.r
        n_possible = self.n_vars
        density_val = density.item() if torch.is_tensor(density) else float(density)
        k_edges = max(1, int(round(density_val * (self.N*(self.N-1)/2))))
        a_np = (a_vec.detach().cpu().numpy()).astype(float)
        # avoid zeros
        a_np = np.clip(a_np, 1e-12, None)
        a_norm = a_np / a_np.sum()
        best_loss = float('inf')
        best_b = None
        for _ in range(K):
            chosen = np.random.choice(np.arange(n_possible), size=k_edges, replace=False, p=a_norm)
            b = torch.zeros(self.n_vars, device=self.device)
            b[chosen] = 1.0
            loss, _, _ = self.attack_loss(b)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_b = b.clone()
        return best_b

    def run(self, init='zeros', use_gae=True):
        v = self.init_vector(init)
        v_opt = self.projected_gradient_descent(v)
        if use_gae:
            try:
                v_gae = self.graph_autoencoder_postprocess(v_opt)
            except Exception as e:
                print("GAE postprocess failed:", e)
                v_gae = v_opt.detach()
        else:
            v_gae = v_opt.detach()
        v_bin = self.random_sampling(v_gae, K=self.K, density=self.r)
        if v_bin is None:
            # fallback: threshold
            v_bin = (v_gae > 0.5).float()
        A_rec = vec_to_sym_adj(v_bin, self.N, device=v_bin.device)
        return A_rec.cpu().numpy()

# -------------------------
# Evaluation helpers
# -------------------------
def evaluate_reconstruction_scores(A_score, A_true):
    # A_score may be probabilities or binary; flatten upper tri
    N = A_true.shape[0]
    y_true = []
    y_score = []
    for i in range(N):
        for j in range(i+1, N):
            y_true.append(int(A_true[i, j]))
            y_score.append(float(A_score[i, j]))
    # handle edge case: all zeros or all ones
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float('nan')
    try:
        ap = average_precision_score(y_true, y_score)
    except Exception:
        ap = float('nan')
    return auc, ap

# -------------------------
# Small loader adaptors
# -------------------------
def load_model_and_data_ckpt(ckpt_path, dataset_name, device='cpu'):
    """
    Tries to load a checkpoint saved with:
    torch.save({'model_state_dict': ..., 'model_class': 'GCN', 'args': {...}}, path)
    FALLBACK: user must adapt this to their checkpoint format.
    """
    ck = torch.load(ckpt_path, map_location=device)
    # If train script saved entire model object:
    if isinstance(ck, dict) and 'model' in ck:
        model = ck['model']
    elif isinstance(ck, dict) and 'model_state_dict' in ck:
        # we need user's model class; try common ones
        from pygip.models import GCN  # adjust if names differ
        arch = ck.get('arch', 'GCN').upper()
        if arch == 'GCN':
            model = GCN(feature_number=21, label_number=6)
        elif arch == 'GIN':
            model = GIN()
        else:
            model = GraphSAGE()
        model.load_state_dict(ck['model_state_dict'])
    else:
        raise RuntimeError("Unknown checkpoint format. Edit load_model_and_data_ckpt to match your checkpoint.")
    model.to(device).eval()

    # Load TU dataset directly (using each class's load_pyg_data)
    print(f"Loading TU dataset '{dataset_name}' using built-in PyG loader...")
    from pygip.datasets.datasets import ENZYMES, PROTEINS, AIDS
    from torch_geometric.utils import to_dense_adj

    if dataset_name.upper() == 'ENZYMES':
        ds = ENZYMES.__new__(ENZYMES)
    elif dataset_name.upper() == 'PROTEINS':
        ds = PROTEINS.__new__(PROTEINS)
    elif dataset_name.upper() == 'AIDS':
        ds = AIDS.__new__(AIDS)
    else:
        raise RuntimeError(f"Unsupported dataset: {dataset_name}")
    
        # set dataset path manually so the PyG loader knows where to store data
    ds.path = './data'

    # trigger the PyG-based TU loader (this function usually populates ds.dataset)
    print("Calling load_pyg_data() ...")
    _ = ds.load_pyg_data()

    # try to locate the loaded dataset object (some subclasses store it under .dataset or .data)
    dataset_obj = None
    if hasattr(ds, "dataset"):
        dataset_obj = ds.dataset
        print("Found dataset in ds.dataset")
    elif hasattr(ds, "data"):
        dataset_obj = ds.data
        print("Found dataset in ds.data")
    elif hasattr(ds, "pyg_dataset"):
        dataset_obj = ds.pyg_dataset
        print("Found dataset in ds.pyg_dataset")
    else:
        # fallback: scan ds.__dict__ for a TUDataset-like object
        for k, v in ds.__dict__.items():
            if (hasattr(v, "__getitem__") and hasattr(v, "__len__") and not isinstance(v, (str, bytes, dict))):
                dataset_obj = v
                print(f"Found dataset in ds.__dict__['{k}']")
                break

    if dataset_obj is None:
        raise RuntimeError(
            "Could not find loaded PyG dataset inside the ENZYMES object. "
            "Run print(ds.__dict__.keys()) to inspect available attributes."
        )

    print("Dataset object located successfully.")

    # Select first graph (GraphMI assumes a single graph)
    data = dataset_obj[0] if isinstance(dataset_obj, list) or hasattr(dataset_obj, '__getitem__') else dataset_obj

    from torch_geometric.utils import to_dense_adj
    X = torch.tensor(data.x, dtype=torch.float32)
    edge_index = data.edge_index
    A_true = to_dense_adj(edge_index)[0]

    # --- Fix: handle graph-level vs node-level labels ---
    if data.y is None or len(data.y.shape) == 0 or data.y.numel() == 1:
        # Single label for the whole graph — replicate across all nodes
        Y = torch.zeros(X.shape[0], dtype=torch.long)
    else:
        Y = torch.tensor(data.y, dtype=torch.long).squeeze()

    print(f"Loaded {dataset_name} | Nodes: {X.shape[0]} | Features: {X.shape[1]} | Labels: {len(Y)}")

    return model, X, Y, A_true

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint')
    parser.add_argument('--dataset', default='ENZYMES', help='Dataset name (ENZYMES/PROTEINS/AIDS)')
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.0001)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--init', choices=['zeros', 'uniform'], default='zeros')
    parser.add_argument('--no-gae', action='store_true', help='Disable GAE postprocessing')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    print("Loading checkpoint and data...")
    model, X, Y, A_true = load_model_and_data_ckpt(args.ckpt, args.dataset, device=device)
    print("Starting GraphMI attack (white-box)...")
    attack = GraphMIAttack(model, X, Y, A_true=torch.tensor(A_true), device=device,
                           alpha=args.alpha, beta=args.beta, lr=args.lr, iters=args.iters, K=args.K,
                           est_density=(A_true.sum()/(A_true.shape[0]*(A_true.shape[0]-1))))
    t0 = time.time()
    A_rec = attack.run(init=args.init, use_gae=(not args.no_gae))
    t1 = time.time()
    auc, ap = evaluate_reconstruction_scores(A_rec, A_true)
    print(f"Done. Time: {t1-t0:.2f}s  AUC={auc:.4f}  AP={ap:.4f}")
    # save results
    os.makedirs("results_graphmi", exist_ok=True)
    out = {
        'dataset': args.dataset,
        'ckpt': args.ckpt,
        'iters': args.iters,
        'alpha': args.alpha,
        'beta': args.beta,
        'K': args.K,
        'auc': float(auc) if not np.isnan(auc) else None,
        'ap': float(ap) if not np.isnan(ap) else None,
        'time_s': float(t1-t0)
    }
    import json
    base = os.path.splitext(os.path.basename(args.ckpt))[0]
    outpath = os.path.join("results_graphmi", f"{args.dataset}_{base}_graphmi.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print("Results saved to", outpath)

if __name__ == '__main__':
    main()

