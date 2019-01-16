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
from bandits.data.synthetic_data_sampler import sample_wheel_bandit_data
from bandits.data.environments import  *

from bandits.data.bootstrap_thompson_sampling import generate_artificial_data

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

# Linear Dataet params
# num_actions = 8
# context_dim = 10
# num_contexts = 1500
# noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
# noise_stds = [1 for i in range(num_actions)]

#wheel params
# num_actions = 5
# context_dim = 2
# num_contexts = 1500
# delta = 0.95
# mean_v = [1.0, 1.0, 1.0, 1.0, 1.2]
# std_v = [0.05, 0.05, 0.05, 0.05, 0.05]
# mu_large = 50
# std_large = 0.01


#Mushrooms params
# num_actions = 2
# context_dim = 117
# num_contexts = 3000

#Covertype params
num_actions = 7
context_dim = 54
num_contexts = 3000

def dataset_proto():
    # dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim,
    #                                             num_actions, sigma=noise_stds)
    # dataset, opt_linear = sample_wheel_bandit_data(num_contexts, delta,
    #                                               mean_v, std_v,
    #                                               mu_large, std_large)
    # mush = Mushrooms(num_contexts=num_contexts)
    mush = Covertype(num_contexts=num_contexts)
    # actions = np.ones(len(mush.opts))
    # /print(mush.get_stochastic_regret())
    dataset = mush.table
    opt_rewards, opt_actions = mush.opts[:,0], mush.opts[:,1]
    opt_linear = (opt_rewards, opt_actions)
    return dataset, opt_linear

print(dataset_proto()[0].shape)

# Params for algo templates
hparams = tf.contrib.training.HParams(num_actions=num_actions)
random_proto = lambda : UniformSampling('Uniform Sampling', hparams)
algo_protos = [random_proto, random_proto]


# Run experiments several times save and plot results
benchmarker = Benchmarker(algo_protos, dataset_proto, num_actions, context_dim, nb_contexts=num_contexts, test_name='dummy')

benchmarker.run_experiments(5)
benchmarker.save_results('./results/')
benchmarker.display_results(save_path='./results/')
