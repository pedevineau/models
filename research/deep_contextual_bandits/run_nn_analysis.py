#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple example of contextual bandits simulation.

Code corresponding to:
Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks
for Thompson Sampling, by Carlos Riquelme, George Tucker, and Jasper Snoek.
https://arxiv.org/abs/1802.09127
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time,sys
import tensorflow as tf
from absl import app
from absl import flags
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
# print(matplotlib.get_backend())
from bandits.helpers.benchmarker import Benchmarker
from bandits.algorithms.lin_ucb import LinUCB
from bandits.algorithms.lin_epsilon import LinEpsilon
from bandits.algorithms.neural_lin_ucb import NeuralLinUCB

# from bandits.algorithms.bootstrapped_bnn_sampling import BootstrappedBNNSampling
from bandits.core.contextual_bandit import run_contextual_bandit
# from bandits.data.data_sampler import sample_mushroom_data
from bandits.algorithms.fixed_policy_sampling import FixedPolicySampling
from bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
# from bandits.algorithms.parameter_noise_sampling import ParameterNoiseSampling
from bandits.algorithms.posterior_bnn_sampling import PosteriorBNNSampling
from bandits.data.synthetic_data_sampler import sample_linear_data
# from bandits.data.bootstrap_thompson_sampling import generate_artificial_data

from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
# from bandits.data.synthetic_data_sampler import sample_sparse_linear_data
# from bandits.data.synthetic_data_sampler import sample_wheel_bandit_data
from bandits.algorithms.uniform_sampling import UniformSampling

base_route = os.getcwd()
data_route = 'contextual_bandits/datasets'

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)

flags.DEFINE_string('logdir', base_route, 'Base directory to save output')
FLAGS(sys.argv)

############# STARTS HERE ##############"""
# Create dataset template
num_actions = 8
context_dim = 10
num_contexts = 1500
# noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
noise_stds = [0.1 for i in range(num_actions)]

def dataset_proto():
    dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim,
                                                num_actions, sigma=noise_stds)
    return dataset, opt_linear

# artificial_data_generator = lambda : generate_artificial_data(n_samples=50, n_actions=num_actions, n_features=context_dim)
# Params for algo templates
hparams = tf.contrib.training.HParams(num_actions=num_actions)

hparams_linear = tf.contrib.training.HParams(num_actions=num_actions,
                                             context_dim=context_dim,
                                             a0=6,
                                             b0=6,
                                             lambda_prior=0.25,
                                             initial_pulls=2)

hparams_linucb = tf.contrib.training.HParams(num_actions=num_actions,
                                        context_dim=context_dim,
                                        alpha=1,
                                        lam=0.1)

hparams_lineps = tf.contrib.training.HParams(num_actions=num_actions,
                                        context_dim=context_dim,
                                        lam=0.1,
                                        eps=0.05)

layer_sizes = [[50],[50,50]]
batch_sizes = [512,64,8]
optimizers = ["SGD","RMS"]
neural_greedy_protos = []
for layer_size in layer_sizes:
    for batch_size in batch_sizes:
        for optimizer in optimizers:
            print(batch_size, layer_size, optimizer, 'NG_bs%s_ls%ix50_%s'%(batch_size, len(layer_size), optimizer))
            neural_greedy_protos.append(lambda param=[batch_size, layer_size, optimizer]: PosteriorBNNSampling(
                'NG_bs%s_ls%ix50_%s'%(param[0], len(param[1]), param[2]),
                tf.contrib.training.HParams(num_actions=num_actions,
                                                        context_dim=context_dim,
                                                        init_scale=0.3,
                                                        activation=tf.nn.relu,
                                                        layer_sizes=param[1],
                                                        batch_size=param[0],
                                                        activate_decay=True,
                                                        initial_lr=0.1,
                                                        max_grad_norm=5.0,
                                                        show_training=False,
                                                        freq_summary=1000,
                                                        buffer_s=-1,
                                                        initial_pulls=2,
                                                        optimizer=param[2],
                                                        reset_lr=True,
                                                        lr_decay_rate=0.5,
                                                        training_freq=50,
                                                        training_epochs=50),
                'RMSProp'))



print(len(neural_greedy_protos))

random_proto = lambda : UniformSampling('Uniform Sampling', hparams)
linThompson_proto = lambda : LinearFullPosteriorSampling('linThompson', hparams_linear)
linUCB_proto = lambda : LinUCB('linUCB', hparams_linucb)
linEps_proto = lambda : LinEpsilon('LinEpsilon', hparams_lineps)

algo_protos = neural_greedy_protos + [linUCB_proto, linEps_proto, linThompson_proto, random_proto]
# for algo_proto in algo_protos:
#     algo = algo_proto()
#     print(algo.name, algo.hparams)
# print (algo_protos[0]==algo_protos[1])

# Run experiments several times save and plot results
benchmarker = Benchmarker(algo_protos, dataset_proto, num_actions, context_dim, nb_contexts=num_contexts, test_name='NNparams_linear_test1_0_10')

benchmarker.run_experiments(50)
benchmarker.save_results('./results/')
benchmarker.display_results(save_path='./results/')
