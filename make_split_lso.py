# make_split_lso.py
import os, argparse, re, random
SEED_RE = re.compile(r"_s(\d+)\.pt$", re.IGNORECASE)

def get_seed(name):
    m = SEED_RE.search(name)
    return int(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="results_bank")
    ap.add_argument("--train_seeds", default="1,2,3,4")
    ap.add_argument("--test_seeds", default="5,6")
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    files = sorted([f for f in os.listdir(args.folder) if f.endswith(".pt")])
    tr_seeds = set(int(x) for x in args.train_seeds.split(",") if x)
    te_seeds = set(int(x) for x in args.test_seeds.split(",") if x)

    train, test = [], []
    for f in files:
        s = get_seed(f)
        if s in tr_seeds: train.append(f)
        elif s in te_seeds: test.append(f)
    if args.shuffle:
        random.seed(0); random.shuffle(train); random.shuffle(test)

    # Ensure both classes exist in each split (user will train with guards anyway)
    with open("train.txt","w") as ft: ft.write("\n".join(train) + ("\n" if train else ""))
    with open("test.txt","w") as fe:  fe.write("\n".join(test)  + ("\n" if test else ""))
    print(f"Wrote train.txt ({len(train)}) and test.txt ({len(test)}).")

if __name__ == "__main__":
    main()

