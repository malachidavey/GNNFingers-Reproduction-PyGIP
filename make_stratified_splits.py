#!/usr/bin/env python3
import re, sys, random, os
from pathlib import Path

# CONFIG (edit if you want)
SEEDS = [0,1,2,3,4]
TRAIN_NEG_MIN = 4           # ensure >= this many negatives in TRAIN
TRAIN_RATIO = 0.8           # 80/20 split
OUTDIR = Path("splits")
OUTDIR.mkdir(exist_ok=True)

# Inputs: we’ll combine whatever exists
candidates = []
for f in ["train.txt","test.txt","all.txt","file_list.txt","manifest.txt"]:
    p = Path(f)
    if p.exists():
        candidates.append(p)

if not candidates:
    print("ERROR: Couldn’t find any of: train.txt, test.txt, all.txt, file_list.txt, manifest.txt", file=sys.stderr)
    sys.exit(1)

def parse_line(line:str):
    """
    Accepts lines like:
      /path/to/item_foo.pt
      /path/to/item_foo.pt <label>
    Returns (path, label) where label in {+1,-1} or None if undetected.
    """
    s = line.strip()
    if not s: return None
    parts = s.split()
    path = parts[0]
    label = None
    if len(parts) >= 2:
        # If a numeric/+-1 label is present explicitly
        if re.fullmatch(r"[-+]?1|0", parts[-1]):
            # map 1->pos, 0 or -1->neg (tweak if your format differs)
            label = +1 if parts[-1] == "1" else -1

    # If no explicit label, infer from filename
    if label is None:
        fname = Path(path).name.lower()
        if re.search(r"(pos|f\+|fplus|\bpositive\b)", fname):
            label = +1
        elif re.search(r"(neg|f\-|fminus|\bnegative\b)", fname):
            label = -1

    return (path, label)

# Collect unique items
items = {}
for p in candidates:
    for line in p.read_text().splitlines():
        parsed = parse_line(line)
        if parsed is None: 
            continue
        path, label = parsed
        items[path] = label if label is not None else items.get(path)

# Filter to those with detectable labels
pos = [p for p,l in items.items() if l == +1]
neg = [p for p,l in items.items() if l == -1]

if not pos or not neg:
    print(f"ERROR: Couldn’t detect labels from filenames/lines.\n"
          f"Sample names:\n  POS? {pos[:3]}\n  NEG? {neg[:3]}\n"
          f"Tip: ensure filenames contain 'pos'/'neg' or 'F+'/'F-' or provide labels in the list files.",
          file=sys.stderr)
    sys.exit(2)

print(f"Found {len(pos)} positives and {len(neg)} negatives across {len(items)} items.")

def stratified_split(pos_list, neg_list, seed):
    rng = random.Random(seed)
    P = pos_list[:]
    N = neg_list[:]
    rng.shuffle(P)
    rng.shuffle(N)

    p_train = int(round(len(P)*TRAIN_RATIO))
    n_train = int(round(len(N)*TRAIN_RATIO))

    # ensure minimum negatives in train
    n_train = max(n_train, min(len(N), TRAIN_NEG_MIN))

    train = P[:p_train] + N[:n_train]
    test  = P[p_train:] + N[n_train:]

    # shuffle the concatenated lists deterministically
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test

for seed in SEEDS:
    tr, te = stratified_split(pos, neg, seed)
    (OUTDIR / f"train_{seed}.txt").write_text("\n".join(tr) + "\n")
    (OUTDIR / f"test_{seed}.txt").write_text("\n".join(te) + "\n")
    print(f"[seed {seed}] train={len(tr)}  test={len(te)}  (neg min in train = {TRAIN_NEG_MIN})")

