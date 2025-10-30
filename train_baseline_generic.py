import os, torch, random, numpy as np, argparse
from pygip.models.attack import ModelExtractionAttack0
from pygip.datasets.datasets import Cora, CiteSeer, ENZYMES, PROTEINS, AIDS

def get_dataset(name):
    name = name.lower()
    if name == "cora":
        return Cora()
    if name == "citeseer":
        return CiteSeer()
    if name == "enzymes":
        return ENZYMES(api_type="pyg", path="./data")
    if name == "proteins":
        return PROTEINS(api_type="pyg", path="./data")
    if name == "aids":
        return AIDS(api_type="pyg", path="./data")
    raise ValueError(f"Unknown dataset: {name}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["Cora", "Citeseer", "ENZYMES", "PROTEINS", "AIDS"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--a_ratio", type=float, default=0.25)
    parser.add_argument("--x_ratio", type=float, default=0.25)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs("reproducibility/checkpoints", exist_ok=True)

    ds = get_dataset(args.dataset)
    atk = ModelExtractionAttack0(ds, attack_a_ratio=args.a_ratio, attack_x_ratio=args.x_ratio)

    print(f"Training target via MEA._train_target_model() for {args.dataset}")
    trained_target = atk._train_target_model()

    ckpt_path = f"reproducibility/checkpoints/{args.dataset.lower()}_target_seed{args.seed}.pth"
    try:
        if hasattr(trained_target, "state_dict"):
            torch.save(trained_target.state_dict(), ckpt_path)
            print("Saved checkpoint:", ckpt_path)
        else:
            torch.save(trained_target, ckpt_path + ".obj")
            print("Saved object checkpoint:", ckpt_path + ".obj")
    except Exception as e:
        print("Could not save checkpoint automatically:", e)
