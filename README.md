# Graph Optimal Transport for Similarity Computation (GOTSim)


## Setup the environment

GOTSim requires python 3.7+ and conda environment. Please refer to `requirements.txt` file for the dependenencies.

## Training and Evaluation

GOTSim's training and evaluation processes are encapsulated inside the script `train_gotsim.py'. To train and evaluate using the provided 5-fold evaluation, simply run:

```
export PYTHONPATH=external/:python/:$PYTHONPATH
python python train_gotsim.py --basedir exp/final/PTC_ged/GOTSim/ --dataset data/PTC_ged/
```

