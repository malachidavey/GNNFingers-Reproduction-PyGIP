from .backbones.gcn import GCN

# Optional: try to import others if they exist
try:
    from .backbones.graphsage import GraphSAGE
except ImportError:
    GraphSAGE = None

try:
    from .backbones.shadownet import ShadowNet
except ImportError:
    ShadowNet = None

try:
    from .backbones.attacknet import AttackNet
except ImportError:
    AttackNet = None

