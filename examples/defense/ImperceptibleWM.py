from pygip.datasets import Cora
from pygip.models.defense import ImperceptibleWM


# TODO test datasets
# TODO test gpu
# TODO verify performance
# TODO record metrics (original acc, defense acc, fidelity)
# TODO verification after attack on defense
# TODO record metrics (AUC[watermark], Acc[fingerprint])

def imperceptiblewm():
    dataset = Cora(api_type='pyg')
    med = ImperceptibleWM(dataset, attack_node_fraction=0.1)
    med.defend()


if __name__ == '__main__':
    imperceptiblewm()
