import os, argparse, torch, random, numpy as np
from torch import nn
from torch.optim import Adam
from torch.nn.utils import prune
from torch_geometric.loader import DataLoader
from train_tu_baseline import get_dataset, build_model, eval_model

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def ft_last_layer(model):
    for p in model.parameters(): p.requires_grad = False
    for p in model.lin.parameters(): p.requires_grad = True

def ft_all_layers(model):
    for p in model.parameters(): p.requires_grad = True

def train_loop(model, train_loader, test_loader, device, epochs=30, lr=1e-3, wd=5e-4):
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = crit(out, batch.y.long())
            loss.backward(); opt.step()
        acc = eval_model(model, test_loader, device)
        best = max(best, acc)
    return best

@torch.no_grad()
def logits_teacher(model, batch):
    return model(batch.x.float(), batch.edge_index, batch.batch)

def distill_student(teacher, student, train_loader, test_loader, device, epochs=50, lr=1e-3, wd=5e-4, T=2.0, alpha=0.7):
    teacher.eval()
    opt = Adam(student.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction='batchmean')
    best = 0.0
    for _ in range(epochs):
        student.train()
        for batch in train_loader:
            batch = batch.to(device)
            with torch.no_grad():
                t_logits = logits_teacher(teacher, batch)
            s_logits = student(batch.x.float(), batch.edge_index, batch.batch)
            loss_soft = kl(nn.functional.log_softmax(s_logits/T, dim=-1),
                           nn.functional.softmax(t_logits/T, dim=-1)) * (T*T)
            loss_hard = ce(s_logits, batch.y.long())
            loss = alpha*loss_soft + (1-alpha)*loss_hard
            opt.zero_grad(); loss.backward(); opt.step()
        acc = eval_model(student, test_loader, device)
        best = max(best, acc)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ENZYMES","PROTEINS","AIDS"])
    ap.add_argument("--arch", default="GIN", choices=["GCN","GIN","GraphSAGE"])
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--mode", required=True, choices=["ft_last","ft_all","prune","distill"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results_bank")
    ap.add_argument("--prune_pct", type=float, default=0.2)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    args = ap.parse_args()

    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = get_dataset(args.dataset)
    tr = DataLoader(ds.train_data, batch_size=128, shuffle=True)
    te = DataLoader(ds.test_data,  batch_size=128, shuffle=False)

    sample = ds.train_data[0]
    in_dim = int(sample.x.size(-1)) if getattr(sample, "x", None) is not None else 1
    out_dim = int(ds.graph_dataset.num_classes)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode in ["ft_last","ft_all","prune"]:
        model = build_model(args.arch, in_dim, out_dim).to(dev)
        sd = torch.load(args.baseline_ckpt, map_location=dev)
        model.load_state_dict(sd, strict=False)

        if args.mode == "ft_last":
            ft_last_layer(model)
            best = train_loop(model, tr, te, dev, epochs=args.epochs, lr=args.lr, wd=args.wd)
            tag = f"{args.dataset.lower()}_{args.arch.lower()}_ft-last_s{args.seed}.pt"

        elif args.mode == "ft_all":
            ft_all_layers(model)
            best = train_loop(model, tr, te, dev, epochs=args.epochs, lr=args.lr, wd=args.wd)
            tag = f"{args.dataset.lower()}_{args.arch.lower()}_ft-all_s{args.seed}.pt"

        else:  # prune
            if hasattr(model, "lin") and hasattr(model.lin, "weight"):
                prune.l1_unstructured(model.lin, name="weight", amount=args.prune_pct)
            ft_all_layers(model)
            best = train_loop(model, tr, te, dev, epochs=max(20, args.epochs//2), lr=args.lr, wd=args.wd)
            tag = f"{args.dataset.lower()}_{args.arch.lower()}_prune{int(args.prune_pct*100)}_s{args.seed}.pt"

        out = os.path.join(args.out_dir, tag)
        torch.save(model.state_dict(), out)
        with open(os.path.join(args.out_dir, tag.replace(".pt",".metrics.txt")), "w") as f:
            f.write(f"pos_mode={args.mode}\nseed={args.seed}\nbest_acc={best:.6f}\n")
        print("Saved:", out, "best_acc=", best)

    else:  # distill
        teacher = build_model(args.arch, in_dim, out_dim).to(dev)
        teacher.load_state_dict(torch.load(args.baseline_ckpt, map_location=dev), strict=False)
        student = build_model(args.arch, in_dim, out_dim).to(dev)
        best = distill_student(teacher, student, tr, te, dev, epochs=args.epochs, lr=args.lr, wd=args.wd, T=args.T, alpha=args.alpha)
        tag = f"{args.dataset.lower()}_{args.arch.lower()}_distillT{args.T}_s{args.seed}.pt"
        out = os.path.join(args.out_dir, tag)
        torch.save(student.state_dict(), out)
        with open(os.path.join(args.out_dir, tag.replace(".pt",".metrics.txt")), "w") as f:
            f.write(f"pos_mode=distill\nT={args.T}\nalpha={args.alpha}\nseed={args.seed}\nbest_acc={best:.6f}\n")
        print("Saved:", out, "best_acc=", best)

if __name__ == "__main__":
    main()
