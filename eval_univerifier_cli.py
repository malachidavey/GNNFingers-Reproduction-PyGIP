# eval_univerifier_cli.py
import os, argparse, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from train_univerifier_cli import FingerprintDataset, Univerifier

def main():
    # ------------------- args -------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="results_bank")
    ap.add_argument("--input_dim", type=int, default=8192)  # default 8192 for your setup
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--file_list", default=None, help="txt of filenames to eval (e.g., test.txt)")
    ap.add_argument("--train_list", default=None, help="txt of filenames to fit threshold (optional)")
    ap.add_argument("--thresh_mode", choices=["youden","ba","target_fpr"], default="target_fpr")
    ap.add_argument("--target_fpr", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_prefix", type=str, default="")
    args = ap.parse_args()

    # ---- seed everything ----
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        torch.manual_seed(args.seed)
    except Exception:
        pass

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # small helper for prefixed output paths
    def with_prefix(path: str) -> str:
        d, f = os.path.split(path)
        return os.path.join(d, f"{args.save_prefix}{f}") if args.save_prefix else path

    # ------------------- dataset (TEST/EVAL set) -------------------
    ds = FingerprintDataset(args.folder, args.input_dim, args.zscore, file_list=args.file_list)
    y_counts = {0: int((np.array(ds.labels) == 0).sum()), 1: int((np.array(ds.labels) == 1).sum())}
    print(f"Loaded {len(ds)} fingerprints  (F+: {y_counts[1]}, F-: {y_counts[0]})")
    if y_counts[0] == 0 or y_counts[1] == 0:
        raise RuntimeError("Need BOTH classes in the evaluation set. Check your --file_list or labels.")

    # ------------------- model -------------------
    best_path = os.path.join(args.folder, "univerifier_best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(best_path)
    model = Univerifier(args.input_dim).to(DEVICE)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    # ------------------- collect TEST probs -------------------
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
            probs.append(p); labels.append(y.numpy())
    probs = np.concatenate(probs); labels = np.concatenate(labels)

    # ------------------- ROC/PR on TEST -------------------
    fpr_te, tpr_te, thr_te = roc_curve(labels, probs)
    roc_auc = auc(fpr_te, tpr_te)

    # ------------------- (optional) TRAIN set for threshold selection -------------------
    use_train_for_thresh = args.train_list is not None
    if use_train_for_thresh:
        print(f"Computing threshold from TRAIN set (mode={args.thresh_mode})...")
        ds_tr = FingerprintDataset(args.folder, args.input_dim, args.zscore, file_list=args.train_list)
        loader_tr = torch.utils.data.DataLoader(ds_tr, batch_size=64, shuffle=False)
        probs_tr, labels_tr = [], []
        with torch.no_grad():
            for x, y in loader_tr:
                p = torch.softmax(model(x.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
                probs_tr.append(p); labels_tr.append(y.numpy())
        probs_tr = np.concatenate(probs_tr); labels_tr = np.concatenate(labels_tr)
        fpr_src, tpr_src, thr_src = roc_curve(labels_tr, probs_tr)
    else:
        # fall back to TEST set for threshold picking
        fpr_src, tpr_src, thr_src = fpr_te, tpr_te, thr_te

    # ------------------- choose threshold once (no duplicate logic) -------------------
    mode = args.thresh_mode
    t_star = 0.5

    if mode == "youden":
        j = tpr_src - fpr_src
        t_star = float(thr_src[np.argmax(j)])

    elif mode == "ba":
        thr_grid = np.linspace(0.05, 0.95, 181)
        best_ba, best_t = -1.0, 0.5
        # evaluate BA on the same source set used for picking (TRAIN if provided, else TEST)
        y_src = labels_tr if use_train_for_thresh else labels
        p_src = probs_tr if use_train_for_thresh else probs
        for t in thr_grid:
            preds_src = (p_src >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_src, preds_src, labels=[0,1]).ravel()
            tpr = tp / max(tp+fn, 1)
            tnr = tn / max(tn+fp, 1)
            ba  = 0.5 * (tpr + tnr)
            if ba > best_ba:
                best_ba, best_t = ba, float(t)
        t_star = best_t

    else:  # target_fpr
        target = float(args.target_fpr)
        ok = np.where(fpr_src <= target)[0]
        if len(ok) > 0:
            best = ok[np.argmax(tpr_src[ok])]
            t_star = float(thr_src[best])
        else:
            # percentile fallback on negatives of the source set
            neg_src = (probs_tr[labels_tr == 0] if use_train_for_thresh else probs[labels == 0])
            q = 1.0 - target
            t_star = float(np.quantile(neg_src, q))

    # clamp extremes to avoid degenerate edges on tiny splits
    t_star = float(np.clip(t_star, 0.2, 0.8))
    print(f"Chosen threshold t* = {t_star:.4f} (mode={mode}, "
          f"{'TRAIN' if use_train_for_thresh else 'TEST'} src, target_fpr={args.target_fpr:.2f})")

    # ------------------- TEST metrics at t* -------------------
    preds = (probs >= t_star).astype(int)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tnr = tn / max(tn + fp, 1)  # uniqueness
    tpr_at_t = tp / max(tp + fn, 1)  # robustness

    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    # ------------------- save plots/files (with prefix) -------------------
    os.makedirs(args.folder, exist_ok=True)

    # R-U plot (1 - FPR vs TPR) on TEST
    plt.figure()
    plt.plot(1 - fpr_te, tpr_te, marker='o')
    plt.xlabel("Uniqueness (TNR)")
    plt.ylabel("Robustness (TPR)")
    plt.title(f"R-U Curve (ARUC/AUC={roc_auc:.3f})")
    plt.grid(True); plt.tight_layout()
    ru_path = with_prefix(os.path.join(args.folder, "univerifier_eval_ru.png"))
    plt.savefig(ru_path); plt.close()

    # ROC on TEST
    plt.figure()
    plt.plot(fpr_te, tpr_te)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (AUC={roc_auc:.3f})")
    plt.grid(True); plt.tight_layout()
    roc_path = with_prefix(os.path.join(args.folder, "univerifier_eval_roc.png"))
    plt.savefig(roc_path); plt.close()

    # Curves/metadata
    npz_path = with_prefix(os.path.join(args.folder, "univerifier_curves.npz"))
    np.savez(npz_path, fpr=fpr_te, tpr=tpr_te, probs=probs, labels=labels, prec=prec, rec=rec, t_star=t_star)

    # ------------------- print summary -------------------
    print(f"✅ ARUC/AUC = {roc_auc:.4f} | ACC@t* = {acc:.4f} | AP = {ap:.4f}")
    print(f"    TPR@t* = {tpr_at_t:.4f} | TNR@t* = {tnr:.4f} | CM = {cm.tolist()}")
    print(f"Plots: {ru_path} , {roc_path}")
    print(f"Curves saved to {npz_path}")

if __name__ == "__main__":
    main()
