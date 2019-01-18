# Deep Bayesian Bandits Library

This library is a forked project from the github from the *[Deep Bayesian Bandits Showdown: An Empirical
Comparison of Bayesian Deep Networks for Thompson
Sampling](https://arxiv.org/abs/1802.09127)* paper, published in
[ICLR](https://iclr.cc/) 2018.

The forked github is available at https://github.com/pedevineau/models.

Two scripts were used to produce the results from the report :

1.__run_full_analysis.py__ that was used for comparing a variety of models on linear/wheel/covertype/mushrooms datasets (change name attribute in script to select).

2.__run_nn_analysis.py__ that was used for comparing performance of neural greedy model with different hyperparameters.

Note that you need to create a __"results"__ folder at the same level of the README.md to run these scripts.


Additions from the original project are :
* LinUCB, neuralLinUCB and Lin Epsilon algorithms
* CovertypeGAN and MushroomGAN for general context vectors and categorical context vectors.
* Use of an artifical data generator in neural network based algorithms
* Custom mushroom and covertype dataset readers
* Benchmarker class : allows to run exxperiments several times, display results in a png, store results in pickle, store algo performance in csv format (csv for later tex export) - DataReader class is the associated class that allows to read again the pickle file and process data once again.

 
