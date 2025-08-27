Quick Start
=================
This guide will help you get started with PyGIP quickly.

Attack Examples
---------------

Model Extraction Attack
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from datasets import Cora
    from models.attack import ModelExtractionAttack0

    # Load the Cora dataset
    dataset = Cora()

    # Initialize the attack with a sampling ratio of 0.25
    mea = ModelExtractionAttack0(dataset, 0.25)

    # Execute the attack
    mea.attack()

To run the attack example:

.. code-block:: bash

    python examples/attack/MEAs.py

Defense Examples
----------------

RandomWM Defense
~~~~~~~~~~~~~~~~

.. code-block:: python

    from datasets import Cora
    from models.defense import RandomWM

    # Load the Cora dataset
    dataset = Cora()

    # Initialize the attack with a sampling ratio of 0.25
    med = RandomWM(dataset, 0.25)

    # Execute the defense
    med.defend()

To run the defense example:

.. code-block:: bash

    python examples/defense/RandomWM.py

GPU Support
-----------

If you want to use cuda, please set environment variable:

.. code-block:: bash

    export PYGIP_DEVICE=cuda:0

Alternatively, you can explicitly specify the device in your code:

.. code-block:: python

    from pygip.utils.hardware import set_device

    set_device("cuda:0")


Next Steps
----------

For more detailed documentation, please refer to:

- :doc:`pygip_datasets` - Available datasets
- :doc:`pygip_models_attack` - Detailed attack mechanisms
- :doc:`pygip_models_defense` - Detailed defense mechanisms
