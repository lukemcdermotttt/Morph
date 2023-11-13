# Morph
Morph: A Model Optimization Toolkit for Physics

See our slides for a brief background and outline for this toolkit: https://docs.google.com/presentation/d/1K_j91n0vi3mIQRGt6x2fCow_2H_Oc68IvZFtkST-uvs/edit?usp=sharing

# Directions
### Set up Environment
Create a virtual environment: `` python3 -m venv .env ``
Load the virtual environment: ``source .env/bin/activate``
Install dependencies: ``pip install -r requirements.txt``


### Set up dataset
Generate the dataset by running ``python get_dataset.py``

### Search
Run ``python run.py`` to use optuna to minimize mean_distance & inference time, the reports will be at optuna_trials.txt
