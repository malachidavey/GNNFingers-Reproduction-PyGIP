from pygip.datasets import Cora
from pygip.models.defense import IntegrityVerification


# TODO test datasets
# TODO test gpu
# TODO verify performance
# TODO record metrics (original acc, defense acc, fidelity)
# TODO verification after attack on defense
# TODO record metrics (AUC[watermark], Acc[fingerprint])

def integrity():
    dataset = Cora(api_type='dgl')
    med = IntegrityVerification(dataset, attack_node_fraction=0.1)
    med.defend()


if __name__ == '__main__':
    integrity()
