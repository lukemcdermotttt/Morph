# Morph: A Model Optimization Toolkit for Physics
We plan to add more tasks & examples to this toolkit such as Jet Classification, Tagging, & Anomaly Detection.
This repository is in the process of getting cleaned, see Rework branch for updates.

# Directions
### Set up Environment
Create a virtual environment: `` python3 -m venv .env ``
Load the virtual environment: ``source .env/bin/activate``
Install dependencies: ``pip install -r requirements.txt``

### Set up dataset
For the dataset used in BraggNN, run ``python data/get_dataset.py`` to generate the set.

For Deepsets dataset, download the normalized_data3.zip file and unzip it into /data/normalized_data3/

### Global Search
Run ``python global_search.py`` to search across architectures that minimize mean_distance & BOPs, the reports will be at global_search.txt. Note: this will run on cuda:0, so make sure to change the device variable in main if this needs to run on another device.

#### Examples of using blocks.py to recreate these architectures in global search
Check out ``examples/model_examples.py`` to see how we can create architectures from these blocks.
For examples to see how the Optuna selects the hyperparameters to create these blocks, see ``examples/hyperparam_examples.py``

### HPO for Training Optimization
In ``examples/NAC/HPO_NAC.py``, we initialize the best model from Global Search; however, can replace this with any model you want. This will save all the trials & create the file ``NAC_HPO_trials.txt``. To run this, change the cuda device and run ``python examples/NAC/HPO_NAC.py``.

To rerun HPO for BraggNN and OpenHLS models, we saved separate files. You can run ``HPO_BraggNN.py`` & ``HPO_OpenHLS.py`` which saves to ``BraggNN_HPO_trials.txt`` and ``OpenHLS_HPO_trials.txt`` accordingly in their respective folders.

### Model Compression to minimize BOPs further
Once you have an optimal training, edit the hyperparameters in ``compress.py`` and run  ``python compress.py``. This will perform iterative magnitude pruning with quantization-aware training. Change the max pruning iteration and the bit_width ``b`` for different compression levels. We saved our results with compressing NAC in ``examples/NAC/NAC_Compress.txt``.
