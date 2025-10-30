# train_univerifier_cli.py
import os, random, argparse, re, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import numpy as np


_POS_PAT = re.compile(r'(?:^|[_\-\s])(ft|finetune|distill|prune)(?:[_\-\s]|$)', re.IGNORECASE)

class UniverifierLegacy(nn.Module):
    def __init__(self, input_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.LeakyReLU(0.1),
            nn.Linear(1024, 256),       nn.LeakyReLU(0.1),
            nn.Linear(256, 2),
        )
    def forward(self, x): return self.net(x)

def is_positive(fname: str) -> bool:
    n = os.path.basename(fname).lower()
    # broad, safe matches
    if "distill" in n:      # handles "distillT2.0"
        return True
    if "prune" in n:
        return True
    # fine-tuning patterns (avoid "soFTware" false hits)
    if "ft-all" in n or "ft-last" in n:
        return True
    if re.search(r'(^|[_\-\s])(ft|finetune)([_\-\s]|$)', n):
        return True
    return False

class FingerprintDataset(Dataset):
    def __init__(self, folder="results_bank", input_dim=4096, zscore=False, file_list=None):
        self.folder = folder
        if file_list:
            with open(file_list, "r") as f:
                names = [line.strip() for line in f if line.strip()]
            self.paths = [os.path.join(folder, n) for n in names]
        else:
            self.paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pt")])
        self.labels = [1 if is_positive(p) else 0 for p in self.paths]  # 1=F+, 0=F-
        self.input_dim = input_dim
        self.zscore = zscore

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        import torch
        path = self.paths[idx]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        state = torch.load(path, map_location="cpu")
        # flatten weights
        if isinstance(state, dict):
            flat = []
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    flat.append(v.flatten())
            flat = torch.cat(flat) if flat else torch.zeros(1)
        elif isinstance(state, torch.Tensor):
            flat = state.flatten()
        else:
            flat = torch.zeros(1)
        if self.zscore:
            m = flat.mean()
            s = flat.std().clamp_min(1e-6)
            flat = (flat - m) / s
        # fixed length
        D = self.input_dim
        if flat.numel() < D:
            x = F.pad(flat, (0, D - flat.numel()))
        else:
            x = flat[:D]
        return x.float(), y

class Univerifier(nn.Module):
    def __init__(self, input_dim=8192):  # we'll train at 8192
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.30),

            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),   # spreads logits; fixes “all ~0.8”
            nn.Dropout(p=0.30),

            nn.Linear(128, 2),
        )
    def forward(self, x):
        return self.net(x)




@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="results_bank")
    ap.add_argument("--train_list", default=None, help="txt of filenames for training set")
    ap.add_argument("--test_list", default=None, help="txt of filenames for test set")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--input_dim", type=int, default=4096)
    ap.add_argument("--zscore", action="store_true")
    args = ap.parse_args()

    random.seed(0); torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build datasets
    ds_train = FingerprintDataset(args.folder, args.input_dim, args.zscore, file_list=args.train_list)
    ds_test  = FingerprintDataset(args.folder, args.input_dim, args.zscore, file_list=args.test_list)

    # Sanity counts
    import numpy as np
    y_tr = np.array(ds_train.labels); y_te = np.array(ds_test.labels)
    print(f"Train: {len(ds_train)} (F+: {(y_tr==1).sum()}, F-: {(y_tr==0).sum()})")
    print(f"Test : {len(ds_test)}  (F+: {(y_te==1).sum()}, F-: {(y_te==0).sum()})")
    if (y_tr==0).sum()==0 or (y_tr==1).sum()==0 or (y_te==0).sum()==0 or (y_te==1).sum()==0:
        raise RuntimeError("Both classes must appear in train and test. Check your split lists.")

    from torch.utils.data import WeightedRandomSampler
    import numpy as np

    labels = np.array(ds_train.labels)
    class_count = np.bincount(labels, minlength=2) + 1e-6
    sample_w = 1.0 / class_count[labels]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(labels),
        replacement=True
    )

    # ---- Balanced sampler (Option B2) ----
    labels = np.array(ds_train.labels)
    class_count = np.bincount(labels, minlength=2) + 1e-6
    sample_w = 1.0 / class_count[labels]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(labels),
        replacement=True
    )

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False)

    # quick sanity check (optional)
    xb, yb = next(iter(train_loader))
    print("Batch balance -> F-:", (yb==0).sum().item(), "| F+:", (yb==1).sum().item())

    model = Univerifier(args.input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ce = nn.CrossEntropyLoss()

    best = 0.0
    patience = 5
    stale = 0
    out_model = os.path.join(args.folder, "univerifier_best.pt")

    for ep in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward(); opt.step()
            total_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        improved = acc > best + 1e-6
        if improved:
            best = acc; stale = 0
            torch.save(model.state_dict(), out_model)
        else:
            stale += 1

        print(f"[Epoch {ep:03d}] loss={total_loss/len(train_loader):.4f} val_acc={acc:.4f} best={best:.4f} stale={stale}")
        if stale >= patience:
            print("Early stop (no val improvement).")
            break

    # Save a tiny metrics file
    with open(os.path.join(args.folder, "univerifier_metrics.json"), "w") as f:
        json.dump({"best_val_acc": best, "input_dim": args.input_dim, "zscore": args.zscore}, f, indent=2)
    print(f"✅ Training complete. Best val acc = {best:.4f}. Saved to {out_model}")

if __name__ == "__main__":
    main()

