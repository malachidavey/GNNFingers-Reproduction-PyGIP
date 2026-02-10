import torch
import dgl
from pygip.attacks.graphmi_attack import GraphMIAttack
from scripts.train_dgl_tu_gcn import DGL_GCN  # adjust if needed

def main():
    N = 10                     # small toy graph
    Fdim = 5                   # feature size
    Hdim = 16                  # hidden dim
    C = 3                      # num classes

    # create random adjacency (simple chain)
    src = torch.arange(N-1)
    dst = torch.arange(1, N)
    g = dgl.graph((src, dst), num_nodes=N)
    g = dgl.add_self_loop(g)

    # random features + labels
    X = torch.randn(N, Fdim)
    Y = torch.randint(0, C, (1,))

    model = DGL_GCN(Fdim, Hdim, C)
    model.eval()

    # init attack object
    attack = GraphMIAttack(model, X, Y, device="cpu", iters=5)

    # make a fake PGD vector
    n_vars = attack.n_vars
    v_opt = torch.rand(n_vars)

    print("Running gae_postprocess test...")
    vec = attack.gae_postprocess(v_opt)
    print("Returned vector shape:", vec.shape)

    expected = n_vars
    print("Expected shape:", expected)

    if vec.shape[0] == expected:
        print("✅ gae_postprocess() works correctly!")
    else:
        print("❌ incorrect output shape!")

if __name__ == "__main__":
    main()
