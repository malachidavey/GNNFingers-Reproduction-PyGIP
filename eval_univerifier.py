# eval_univerifier.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_recall_curve, average_precision_score
from train_univerifier import FingerprintDataset, UniverifierLegacy as Univerifier

ROOT = "results_bank"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESH = 0.5  # decision threshold for ACC/CM

def load_model(path):
    model = Univerifier().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    # Graceful error if shape mismatch (e.g., you changed input_dim)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            "Model shape mismatch. Did you change input_dim (e.g., 4096 -> 8192) "
            "since training? Re-train or align eval model class."
        ) from e
    model.eval()
    return model

@torch.no_grad()
def collect_scores(model, ds):
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    probs, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        out = model(x)
        p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)

def main():
    # -------------------- dataset --------------------
    ds = FingerprintDataset(folder=ROOT)
    y_counts = {0: int((np.array(ds.labels) == 0).sum()), 1: int((np.array(ds.labels) == 1).sum())}
    print(f"Loaded {len(ds)} fingerprints  (F+: {y_counts[1]}, F-: {y_counts[0]})")

    if y_counts[0] == 0 or y_counts[1] == 0:
        raise RuntimeError("Need BOTH classes in dataset. Check filename-based labeling and results_bank contents.")

    # -------------------- model ----------------------
    best_path = os.path.join(ROOT, "univerifier_best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"{best_path} not found. Train first.")
    model = load_model(best_path)

    # -------------------- inference ------------------
    probs, labels = collect_scores(model, ds)

    # -------------------- metrics --------------------
    # ROC/AUC (ARUC)
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # ACC + confusion at 0.5
    preds = (probs >= THRESH).astype(int)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0,1])  # [[TN, FP],[FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    tnr = tn / max(tn + fp, 1)  # uniqueness
    tpr_ = tp / max(tp + fn, 1) # robustness

    # PR (nice when classes are imbalanced)
    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    # -------------------- plots/files ----------------
    os.makedirs(ROOT, exist_ok=True)

    # RU plot
    plt.figure()
    plt.plot(1 - fpr, tpr, marker='o')
    plt.xlabel("Uniqueness (TNR)")
    plt.ylabel("Robustness (TPR)")
    plt.title(f"R-U Curve (ARUC/AUC={roc_auc:.3f})")
    plt.grid(True)
    plt.tight_layout()
    ru_path = os.path.join(ROOT, "univerifier_eval_ru.png")
    plt.savefig(ru_path)
    plt.close()

    # ROC plot (optional but familiar)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (AUC={roc_auc:.3f})")
    plt.grid(True)
    plt.tight_layout()
    roc_path = os.path.join(ROOT, "univerifier_eval_roc.png")
    plt.savefig(roc_path)
    plt.close()

    # Save curve data for reproducibility
    np.savez(os.path.join(ROOT, "univerifier_curves.npz"),
             fpr=fpr, tpr=tpr, probs=probs, labels=labels, prec=prec, rec=rec)

    # -------------------- report ---------------------
    print(f"✅ ARUC/AUC = {roc_auc:.4f} | ACC@0.5 = {acc:.4f} | AP = {ap:.4f}")
    print(f"    TPR (robustness)@0.5 = {tpr_:.4f} | TNR (uniqueness)@0.5 = {tnr:.4f}")
    print(f"    Confusion Matrix [[TN, FP],[FN, TP]] = {cm.tolist()}")
    print(f"Plots saved: {ru_path}  and  {roc_path}")
    print(f"Curve arrays saved to {os.path.join(ROOT, 'univerifier_curves.npz')}")
    # Pro-tip for your write-up:
    # Report both AUC and ACC, and include CM. If you later do leave-seed-out,
    # save those metrics with a suffix (e.g., *_lso.npz) for side-by-side tables.

if __name__ == "__main__":
    main()
