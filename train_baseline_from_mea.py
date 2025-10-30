import os, torch, random, numpy as np
from pygip.datasets import Citeseer  # change to Citeseer/Pubmed as needed
from pygip.models.attack import ModelExtractionAttack0

# Set seeds for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

os.makedirs("reproducibility/checkpoints", exist_ok=True)
ds = Citeseer()   # swap to Citeseer() or Pubmed() if desired

# instantiate MEA object (uses the repo's default target-train routine)
atk = ModelExtractionAttack0(ds, attack_a_ratio=0.25, attack_x_ratio=0.25)

print("Training target model via MEA._train_target_model() ... (this uses the repo's training routine)")
# call the internal trainer (it's fine to call for reproducibility)
trained_target = atk._train_target_model()

# try to save the model weights (best-effort; adapt if the model wrapper differs)
try:
    ckpt_path = f"reproducibility/checkpoints/{ds.__class__.__name__.lower()}_target_seed{seed}.pth"
    # If trained_target is a torch.nn.Module:
    if hasattr(trained_target, "state_dict"):
        torch.save(trained_target.state_dict(), ckpt_path)
        print("Saved checkpoint:", ckpt_path)
    else:
        # If it returns a dict or custom object, try to save via torch.save anyway
        torch.save(trained_target, ckpt_path + ".obj")
        print("Saved object checkpoint:", ckpt_path + ".obj")
except Exception as e:
    print("Warning: could not save checkpoint automatically:", e)
    print("You can inspect 'trained_target' in Python to find the model to save.")
