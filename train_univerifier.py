import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utils
# -----------------------------
def is_positive(fname: str) -> bool:
    """F+ if obfuscated variant; otherwise F-."""
    n = fname.lower()
    return ("ft-" in n) or ("prune" in n) or ("distill" in n)

# -----------------------------
# Dataset
# -----------------------------
class FingerprintDataset(Dataset):
    def __init__(self, folder="results_bank"):
        self.paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pt")])
        self.labels = [1 if is_positive(p) else 0 for p in self.paths]  # 1=F+, 0=F-
        # sanity
        pos = sum(self.labels); neg = len(self.labels) - pos
        print(f"Loaded {len(self.paths)} fingerprints  (F+: {pos}, F-: {neg})")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
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

        # fixed-length 4096
        if flat.numel() < 4096:
            x = F.pad(flat, (0, 4096 - flat.numel()))
        else:
            x = flat[:4096]
        return x.float(), y

# -----------------------------
# Model
# -----------------------------
class Univerifier(nn.Module):
    def __init__(self, input_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 2),
        )
    def forward(self, x): return self.net(x)

# -----------------------------
# Eval
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)

# -----------------------------
# Train
# -----------------------------
def main():
    random.seed(0); torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = FingerprintDataset("results_bank")
    # stratified 80/20 split
    idx_pos = [i for i, l in enumerate(ds.labels) if l == 1]
    idx_neg = [i for i, l in enumerate(ds.labels) if l == 0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        raise RuntimeError("Need BOTH classes (F+ and F-) in results_bank/*.pt. Check filenames and labeling.")

    def split_idx(idxs, frac=0.8):
        k = max(1, int(len(idxs)*frac))
        return idxs[:k], idxs[k:]
    idx_pos_tr, idx_pos_te = split_idx(idx_pos)
    idx_neg_tr, idx_neg_te = split_idx(idx_neg)

    train_idx = idx_pos_tr + idx_neg_tr
    test_idx  = idx_pos_te + idx_neg_te

    train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx), batch_size=8, shuffle=True)
    test_loader  = DataLoader(torch.utils.data.Subset(ds, test_idx),  batch_size=8, shuffle=False)

    model = Univerifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    ce = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, 51):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = ce(out, y)
            loss.backward(); opt.step()
            total_loss += loss.item()
        acc = evaluate(model, test_loader, device)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "results_bank/univerifier_best.pt")
        print(f"[Epoch {ep:03d}] loss={total_loss/len(train_loader):.4f} val_acc={acc:.4f} best={best:.4f}")

    print(f"✅ Training complete. Best val acc = {best:.4f}")

if __name__ == "__main__":
    main()
