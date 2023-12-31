# Morph: A Model Optimization Toolkit for Physics
We plan to add more tasks & examples to this toolkit such as Jet Classification, Tagging, & Anomaly Detection.
This repository is in the process of getting cleaned, see Rework branch for updates.

# Directions
### Set up Environment
Create a virtual environment: `` python3 -m venv .env ``
Load the virtual environment: ``source .env/bin/activate``
Install dependencies: ``pip install -r requirements.txt``

### Set up dataset
Generate the dataset by running ``python data/get_dataset.py``

### Global Search
Run ``python run.py`` to search across architectures that minimize mean_distance & BOPs, the reports will be at optuna_trials.txt

### Examples of using blocks.py to recreate these architectures in global search
Check out ``model_examples.py`` to see how we can create architectures from these blocks.

### HPO for Better Training
In ``HPO.py``, initialize the model you want to optimize training for & update the filename for saving the trials. Then, run ``python HPO.py``.
To rerun HPO for BraggNN and OpenHLS models, we saved separate files. 
