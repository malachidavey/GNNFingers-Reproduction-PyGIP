from pygip.datasets import Cora
from pygip.models.defense import RandomWM


# TODO test datasets
# TODO test gpu
# TODO verify performance
# TODO record metrics (original acc, defense acc, fidelity)
# TODO verification after attack on defense
# TODO record metrics (AUC[watermark], Acc[fingerprint])

def randomwm():
    dataset = Cora(api_type='dgl')
    med = RandomWM(dataset, attack_node_fraction=0.1, wm_node=50, pr=0.1, pg=0.1)
    med.defend()


if __name__ == '__main__':
    randomwm()
