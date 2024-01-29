# context-shapes-language
This repository accompanies the submission to LREC-COLING 2024: Context Shapes Emergent Communication about Concepts at
Different Levels of Abstraction

Implementation of a concept-level reference game in a language emergence paradigm using [EGG](https://github.com/facebookresearch/EGG/tree/main). The implementation 
builds on the [hierarchical reference game](https://github.com/XeniaOhmer/hierarchical_reference_game/tree/master) by Ohmer et al. (2022) and the [concept game](https://github.com/jayelm/emergent-generalization/tree/master) by Mu & Goodman (2021).

## Installing dependencies
We used Python 3.9.15 and PyTorch 1.13.0. Generally, the minimal requirements are Python 3.6 and PyTorch 1.1.0.
`requirements.txt` lists the python packages needed to run this code. Additionally, please make sure you install EGG 
following the instructions [here](https://github.com/facebookresearch/EGG#installing-egg).
1. (optional) Create a new conda environement:
```
conda create --name emergab python=3.9
conda activate emergab
```
2. EGG can be installed as a package to be used as a library (see [here](https://github.com/facebookresearch/EGG#installing-egg) for more options):
```
pip install git+https://github.com/facebookresearch/EGG.git
```
3. Install packages from the requirements file:
```
pip install -r requirements.txt
```

## Dataset

The datasets are automatically created when a new training run is started (see 'Training' section below). Datasets can 
also be generated before training and stored as a pickle file with the script 'pickle_ds.py'. This has been done for the larger datasets, i.e. D(4,8)
and D(3,16). These datasets are stored in the 'data' folder.

## Training

Agents can be trained using 'train.py'. The file provides explanations for how to configure agents and training using 
command line parameters.

For example, to train the agents on data set D(4,8) (4 attributes, 8 values) with vocab size factor 3 (default), using 
the same hyperparameters as in the paper, you can execute

`python train.py --dimensions 8 8 8 8 --n_epochs 300 --batch_size 32`

Similarly, for data set D(3, 4), the dimensions flag would be

`--dimensions 4 4 4`

Per default, this conducts one run. If you would like to change the number of runs, e.g. to 5, you can specify that using

`--num_of_runs 5`

If you would like to save the results (interaction file, agent checkpoints, a file storing all hyperparameter values, 
training and validation accuracies over time, plus test accuracy for generalization to novel objects) you can add the flag

`--save True`

## Evaluation

Our results can be found in 'results/'. The subfolders contain the metrics for each run. We stored the final interaction 
for each run which logs all game-relevant information such as sender input, messages, receiver input, and receiver 
selections for the training and validation set. Based on these interactions, we evaluated additional metrics after 
training using the notebook 'evaluate_metrics.ipynb'. We uploaded all metrics but not the interaction files due to their 
large size.

## Visualization

Visualizations of training and results can be found in the notebooks 'analysis.ipynb' and 'analysis_qualitative.ipynb'. 
The former reports all results for the 6 different data sets (D(3,4), D(3,8), D(3,16), D(4,4), D(4,8), D(5,4)). The 
latter contains code to extract message-concept pairs from the interactions and evaluate qualitatively which messages 
have been produced to refer to which concept in which context condition.

## Grid search results

The folder 'grid_search/' contains the results for the hyperparameter grid search. 
