# run_attack_with_checkpoint.py
import os, torch, time, csv, random, numpy as np
from pygip.datasets import Cora
from pygip.models.attack import ModelExtractionAttack0
from pygip.utils.metrics import AttackCompMetric

# ---------------- runtime hotfix (no repo edits) ----------------
_old_update = AttackCompMetric.update
def _patched_update(self, *a, **kw):
    if "inference_target_time" in kw:
        kw["query_target_time"] = kw.pop("inference_target_time")
    return _old_update(self, *a, **kw)
AttackCompMetric.update = _patched_update
# ----------------------------------------------------------------

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def try_load_checkpoint_into_attack(atk, ckpt_path):
    """
    Robust loader: handles either (1) full object saved with torch.save(obj),
    or (2) state_dict saved with torch.save(model.state_dict()).
    Attempts to place state_dict into common attributes.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # detect if it's a state_dict
    is_state_dict = isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values())
    candidates = ["target_model", "target", "model", "net", "gnn", "target_net"]

    if not is_state_dict:
        for name in candidates:
            if hasattr(atk, name):
                try:
                    setattr(atk, name, ckpt)
                    print(f"Loaded full object into atk.{name}")
                    return True
                except Exception:
                    continue
        setattr(atk, "_loaded_target", ckpt)
        print("Saved full checkpoint as atk._loaded_target (no common attr matched).")
        return True

    # state_dict case: try to load into candidate submodules
    for name in candidates:
        target = getattr(atk, name, None)
        if target is None:
            continue
        try:
            if hasattr(target, "load_state_dict"):
                target.load_state_dict(ckpt, strict=False)
                print(f"Loaded state_dict into atk.{name} via load_state_dict(..., strict=False)")
                return True
        except Exception as e:
            print(f"Attempt to load into atk.{name} failed: {e}")
            continue

    # Last resort: try to create a fresh target via atk._train_target_model() and load there
    try:
        model_candidate = atk._train_target_model()
        if hasattr(model_candidate, "load_state_dict"):
            model_candidate.load_state_dict(ckpt, strict=False)
            for name in candidates:
                if hasattr(atk, name):
                    setattr(atk, name, model_candidate)
                    print(f"Replaced atk.{name} with freshly trained+loaded model")
                    return True
            setattr(atk, "_trained_and_loaded_target", model_candidate)
            print("Saved trained+loaded model to atk._trained_and_loaded_target")
            return True
    except Exception as e:
        print("Could not auto-create model via atk._train_target_model():", e)

    print("Warning: checkpoint was not loaded into any known attribute.")
    return False

def run_one(ckpt_path, seed=0, a_ratio=0.25, x_ratio=0.25):
    set_seed(seed)
    ds = Cora()
    atk = ModelExtractionAttack0(ds, attack_a_ratio=a_ratio, attack_x_ratio=x_ratio)
    # Try to load the baseline checkpoint into the attack object's internal target
    loaded = try_load_checkpoint_into_attack(atk, ckpt_path)
    print("Checkpoint loaded:", loaded)
    print("Running attack now...")
    t0 = time.time()
    ret = atk.attack()
    dt = time.time() - t0
    # try to extract acc/fidelity from return value if present
    if isinstance(ret, tuple) and len(ret) >= 2:
        acc, fidelity = ret[0], ret[1]
    elif isinstance(ret, dict):
        acc = ret.get("acc", None) or ret.get("accuracy", None)
        fidelity = ret.get("fidelity", None)
    else:
        acc, fidelity = None, None
    return {"ckpt": os.path.basename(ckpt_path), "seed": seed, "attack_a_ratio": a_ratio,
            "attack_x_ratio": x_ratio, "acc": acc, "fidelity": fidelity, "time_s": round(dt,3)}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="reproducibility/checkpoints/cora_target_seed0.pth.obj")
    parser.add_argument("--seeds", default="0", help="comma list, e.g. 0,1,2")
    parser.add_argument("--a_ratio", type=float, default=0.25)
    parser.add_argument("--x_ratio", type=float, default=0.25)
    args = parser.parse_args()

    ckpt_path = args.ckpt
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for s in seeds:
        rows.append(run_one(ckpt_path, seed=s, a_ratio=args.a_ratio, x_ratio=args.x_ratio))

    out = "results_mea_with_checkpoint.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("Wrote", out)
