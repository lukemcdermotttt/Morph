# Morph
Morph: A Model Optimization Toolkit for Physics

See our slides for a brief background and outline for this toolkit: https://docs.google.com/presentation/d/1K_j91n0vi3mIQRGt6x2fCow_2H_Oc68IvZFtkST-uvs/edit?usp=sharing


# Directions
Load the virtual environemt: ``source .env/bin/activate``

run ``python get_dataset.py`` to generate the dataset

run ``python testrun.py`` to use optuna to minimize mean_distance, this also reports inference times (x50) and val loss
