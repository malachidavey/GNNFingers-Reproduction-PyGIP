# eval_univerifier_cli_legacy.py
import os, argparse, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from train_univerifier_cli import FingerprintDataset  # reuse your dataset

import torch.nn as nn
class UniverifierLegacy(nn.Module):
    """Legacy verifier: no dropout; layers match your old checkpoint."""
    def __init__(self, input_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.LeakyReLU(0.1),
            nn.Linear(1024, 256),       nn.LeakyReLU(0.1),
            nn.Linear(256, 2),
        )
    def forward(self, x): return self.net(x)

def main():
    # ------------------- args -------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="results_bank")
    ap.add_argument("--input_dim", type=int, default=4096)  # legacy ckpt was 4096
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--file_list", default=None, help="txt of filenames to eval (e.g., test.txt)")
    ap.add_argument("--train_list", default=None, help="txt of filenames to fit threshold (optional)")
    args = ap.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------- dataset -------------------
    ds = FingerprintDataset(args.folder, args.input_dim, args.zscore, file_list=args.file_list)
    y_counts = {0: int((np.array(ds.labels) == 0).sum()), 1: int((np.array(ds.labels) == 1).sum())}
    print(f"Loaded {len(ds)} fingerprints  (F+: {y_counts[1]}, F-: {y_counts[0]})")
    if y_counts[0] == 0 or y_counts[1] == 0:
        raise RuntimeError("Need BOTH classes in the evaluation set. Check your --file_list or labels.")

    # ------------------- model -------------------
    best_path = os.path.join(args.folder, "univerifier_best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(best_path)
    model = UniverifierLegacy(args.input_dim).to(DEVICE)
    state = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(state)  # shapes/keys must match legacy arch
    model.eval()

    # ------------------- collect TEST probs -------------------
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
            probs.append(p); labels.append(y.numpy())
    probs = np.concatenate(probs); labels = np.concatenate(labels)

    # ------------------- choose threshold from TRAIN (optional) -------------------
    t_star = 0.5
    if args.train_list:
        print("Computing threshold from TRAIN set (Youden J)...")
        ds_tr = FingerprintDataset(args.folder, args.input_dim, args.zscore, file_list=args.train_list)
        loader_tr = torch.utils.data.DataLoader(ds_tr, batch_size=64, shuffle=False)
        probs_tr, labels_tr = [], []
        with torch.no_grad():
            for x, y in loader_tr:
                p = torch.softmax(model(x.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
                probs_tr.append(p); labels_tr.append(y.numpy())
        probs_tr = np.concatenate(probs_tr); labels_tr = np.concatenate(labels_tr)
        fpr_tr, tpr_tr, thr_tr = roc_curve(labels_tr, probs_tr)
        j = tpr_tr - fpr_tr
        t_star = float(thr_tr[np.argmax(j)])
        print(f"Chosen threshold t* = {t_star:.4f}")

    # ------------------- metrics -------------------
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    preds = (probs >= t_star).astype(int)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tnr = tn / max(tn + fp, 1)          # uniqueness
    tpr_at_t = tp / max(tp + fn, 1)     # robustness

    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    # ------------------- plots/files -------------------
    os.makedirs(args.folder, exist_ok=True)

    # R-U
    plt.figure()
    plt.plot(1 - fpr, tpr, marker='o')
    plt.xlabel("Uniqueness (TNR)")
    plt.ylabel("Robustness (TPR)")
    plt.title(f"R-U Curve (ARUC/AUC={roc_auc:.3f})")
    plt.grid(True)
    plt.tight_layout()
    ru_path = os.path.join(args.folder, "univerifier_eval_ru.png")
    plt.savefig(ru_path); plt.close()

    # ROC
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (AUC={roc_auc:.3f})")
    plt.grid(True)
    plt.tight_layout()
    roc_path = os.path.join(args.folder, "univerifier_eval_roc.png")
    plt.savefig(roc_path); plt.close()

    # Curves
    np.savez(os.path.join(args.folder, "univerifier_curves.npz"),
             fpr=fpr, tpr=tpr, probs=probs, labels=labels, prec=prec, rec=rec, t_star=t_star)

    # ------------------- print summary -------------------
    print(f"✅ ARUC/AUC = {roc_auc:.4f} | ACC@t* = {acc:.4f} | AP = {ap:.4f}")
    print(f"    TPR@t* = {tpr_at_t:.4f} | TNR@t* = {tnr:.4f} | CM = {cm.tolist()}")
    print(f"Plots: {ru_path} , {roc_path}")
    print(f"Curves saved to {os.path.join(args.folder, 'univerifier_curves.npz')}")

if __name__ == "__main__":
    main()

