# Morph: A Model Optimization Toolkit for Physics
See our slides for a brief background and outline for this toolkit: https://docs.google.com/presentation/d/1K_j91n0vi3mIQRGt6x2fCow_2H_Oc68IvZFtkST-uvs/edit?usp=sharing
We plan to add more tasks & examples to this toolkit such as Jet Classification, Tagging, & Anomaly Detection.

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
In ``HPO_NAC.py``, we initialize the best model from Global Search; however, can replace this with any model you want. This will save all the trials & create the file ``NAC_HPO_trials.txt``. To run this, change the cuda device and run ``python HPO_NAC.py``.

To rerun HPO for BraggNN and OpenHLS models, we saved separate files. You can run ``HPO_BraggNN.py`` & ``HPO_OpenHLS.py`` which saves to ``BraggNN_HPO_trials.txt`` and ``OpenHLS_HPO_trials.txt`` accordingly.

### Compress to minimize BOPs
Once you have an optimal training, edit the hyperparameters in ``Compress.py`` and run  ``python Compress.py``.