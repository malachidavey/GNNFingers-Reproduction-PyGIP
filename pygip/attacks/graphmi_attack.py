import torch
import torch.nn.functional as F
import dgl
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def upper_tri_indices(N, device=None):
    rows = []
    cols = []
    for i in range(N):
        for j in range(i+1, N):
            rows.append(i)
            cols.append(j)
    return torch.tensor(rows, device=device), torch.tensor(cols, device=device)


def vec_to_sym_adj(vec, N, device=None):
    if device is None:
        device = vec.device
    A = torch.zeros((N, N), device=device)
    r, c = upper_tri_indices(N, device)
    A[r, c] = vec
    A[c, r] = vec
    return A


def sym_adj_to_vec(A):
    N = A.shape[0]
    vals = []
    for i in range(N):
        for j in range(i+1, N):
            vals.append(A[i, j])
    return torch.stack(vals)


def feature_smoothness_loss(A, X, eps=1e-9):
    deg = A.sum(1).clamp(min=eps)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))
    Lhat = torch.eye(A.size(0), device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
    return torch.trace(X.T @ Lhat @ X)


class GraphMIAttack:
    def __init__(
        self, model, X, Y, A_true=None, device="cpu",
        alpha=0.001, beta=0.0001, lr=0.1, iters=100, K=20, est_density=None
    ):
        self.model = model.to(device).eval()
        self.X = X.to(device)
        self.Y = Y.to(device)
        self.device = device
        self.iters = iters
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.K = K

        self.N = X.shape[0]
        self.n_vars = self.N * (self.N - 1) // 2

        if est_density is not None:
            self.r = est_density
        elif A_true is not None:
            total = self.N * (self.N - 1) / 2
            edge_count = A_true.sum() / 2
            self.r = max(1e-6, edge_count / total)
        else:
            self.r = 0.1

        self.r_idx, self.c_idx = upper_tri_indices(self.N, device)

    def init_vector(self):
        return torch.zeros(self.n_vars, device=self.device, requires_grad=True)

    def attack_loss(self, a_vec):
        A_prob = vec_to_sym_adj(a_vec, self.N, device=self.device).clamp(0,1)
        A_bin = (A_prob > 0.5).float()

        src, dst = torch.nonzero(A_bin, as_tuple=True)
        g = dgl.graph((src, dst), num_nodes=self.N)
        g = dgl.add_self_loop(g)

        logits = self.model(g, self.X)
        ce = F.cross_entropy(logits, self.Y)

        Ls = feature_smoothness_loss(A_prob, self.X)
        reg = torch.norm(a_vec, 2)**2

        return ce + self.alpha * Ls + self.beta * reg

    def run_pgd(self, v):
        opt = torch.optim.SGD([v], lr=self.lr)
        for t in range(self.iters):
            opt.zero_grad()
            loss = self.attack_loss(v)
            loss.backward()
            opt.step()
            with torch.no_grad():
                v.clamp_(0, 1)
        return v.detach()

    def gae_postprocess(self, v_opt):
        # Convert PGD vector -> initial adjacency probabilities
        A_prob = vec_to_sym_adj(v_opt, self.N, self.device)

        # Threshold to build binary adjacency
        A_bin = (A_prob > 0.5).float()

        # Build DGL graph from binary adjacency
        src, dst = torch.nonzero(A_bin, as_tuple=True)
        g = dgl.graph((src, dst), num_nodes=self.N)
        g = dgl.add_self_loop(g)

        # ---- FIX: Use local_scope so DGL does not throw errors ----
        with g.local_scope():
            g.ndata['feat'] = self.X
            Z = self.model.penultimate(g, self.X)   # NxH node embeddings

        # Decode adjacency with sigmoid(ZZ^T)
        A_hat = torch.sigmoid(Z @ Z.t())

        # Convert matrix back → vector
        return sym_adj_to_vec(A_hat)


    def random_sampling(self, a_vec):
        a_np = (a_vec.detach().cpu().numpy().astype(float))
        a_np = np.clip(a_np, 1e-9, None)
        a_np /= a_np.sum()

        k = int(round(self.r * self.n_vars))
        best_loss, best_b = 1e9, None

        for _ in range(self.K):
            idx = np.random.choice(self.n_vars, size=k, replace=False, p=a_np)
            b = torch.zeros(self.n_vars, device=self.device)
            b[idx] = 1.0
            loss = self.attack_loss(b)
            if loss.item() < best_loss:
                best_loss, best_b = loss.item(), b.clone()

        return best_b

    def run(self):
        v = self.init_vector()
        v = self.run_pgd(v)
        v = self.gae_postprocess(v)
        v_bin = self.random_sampling(v)

        A_rec = vec_to_sym_adj(v_bin, self.N, self.device)
        return A_rec.cpu().numpy()
