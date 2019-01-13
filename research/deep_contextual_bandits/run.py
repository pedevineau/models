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


from bandits.algorithms.bootstrapped_bnn_sampling import BootstrappedBNNSampling
from bandits.core.contextual_bandit import run_contextual_bandit
from bandits.data.data_sampler import sample_mushroom_data
from bandits.algorithms.fixed_policy_sampling import FixedPolicySampling
from bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.algorithms.parameter_noise_sampling import ParameterNoiseSampling
from bandits.algorithms.posterior_bnn_sampling import PosteriorBNNSampling
from bandits.data.synthetic_data_sampler import sample_linear_data
from bandits.data.synthetic_data_sampler import sample_sparse_linear_data
from bandits.data.synthetic_data_sampler import sample_wheel_bandit_data
from bandits.algorithms.uniform_sampling import UniformSampling


# Create dataset
num_actions = 8
context_dim = 10
num_contexts = 1500

# noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
noise_stds = [1 for i in range(num_actions)]

# dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim,
#                                             num_actions, sigma=noise_stds)
# opt_rewards, opt_actions = opt_linear
def dataset_proto():
    dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim,
                                                num_actions, sigma=noise_stds)
    return dataset, opt_linear
# type(dataset), dataset.shape, dataset
# type(opt_actions), opt_actions.shape, opt_actions

base_route = os.getcwd()
data_route = 'contextual_bandits/datasets'

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)

flags.DEFINE_string('logdir', base_route, 'Base directory to save output')
FLAGS(sys.argv)


# In[11]:






def display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, name):
  """Displays summary statistics of the performance of each algorithm."""

  print('---------------------------------------------------')
  print('---------------------------------------------------')
  print('{} bandit completed after {} seconds.'.format(
    name, time.time() - t_init))
  print('---------------------------------------------------')

  performance_pairs = []
  for j, a in enumerate(algos):
    performance_pairs.append((a.name, np.sum(h_rewards[:, j])))
  performance_pairs = sorted(performance_pairs,
                             key=lambda elt: elt[1],
                             reverse=True)
  for i, (name, reward) in enumerate(performance_pairs):
    print('{:3}) {:20}| \t \t total reward = {:10}.'.format(i, name, reward))

  print('---------------------------------------------------')
  print('Optimal total reward = {}.'.format(np.sum(opt_rewards)))
  print('Frequency of optimal actions (action, frequency):')
  print([[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)])
  print('---------------------------------------------------')
  print('---------------------------------------------------')


# In[3]:

hparams = tf.contrib.training.HParams(num_actions=num_actions)

hparams_linear = tf.contrib.training.HParams(num_actions=num_actions,
                                             context_dim=context_dim,
                                             a0=6,
                                             b0=6,
                                             lambda_prior=0.25,
                                             initial_pulls=2)

hparams_rms = tf.contrib.training.HParams(num_actions=num_actions,
                                        context_dim=context_dim,
                                        init_scale=0.3,
                                        activation=tf.nn.relu,
                                        layer_sizes=[50],
                                        batch_size=512,
                                        activate_decay=True,
                                        initial_lr=0.1,
                                        max_grad_norm=5.0,
                                        show_training=False,
                                        freq_summary=1000,
                                        buffer_s=-1,
                                        initial_pulls=2,
                                        optimizer='RMS',
                                        reset_lr=True,
                                        lr_decay_rate=0.5,
                                        training_freq=50,
                                        training_epochs=50)

hparams_rms2 = tf.contrib.training.HParams(num_actions=num_actions,
                                        context_dim=context_dim,
                                        init_scale=0.3,
                                        activation=tf.nn.relu,
                                        layer_sizes=[50],
                                        batch_size=64,
                                        activate_decay=True,
                                        initial_lr=0.1,
                                        max_grad_norm=5.0,
                                        show_training=False,
                                        freq_summary=1000,
                                        buffer_s=-1,
                                        initial_pulls=2,
                                        optimizer='RMS',
                                        reset_lr=True,
                                        lr_decay_rate=0.5,
                                        training_freq=50,
                                        training_epochs=50)

neural_greedy_proto = lambda : PosteriorBNNSampling('NeuralGreedy_bs_512', hparams_rms, 'RMSProp')
neural_greedy_proto2 = lambda : PosteriorBNNSampling('NeuralGreedy_bs_64', hparams_rms2, 'RMSProp')

random_proto = lambda : UniformSampling('Uniform Sampling', hparams)
linThompson_proto = lambda : LinearFullPosteriorSampling('linThompson', hparams_linear)

algo_protos = [neural_greedy_proto, neural_greedy_proto2, linThompson_proto, random_proto]
# algo_protos = [linThompson_proto, random_proto]


benchmarker = Benchmarker(algo_protos, dataset_proto, num_actions, context_dim, nb_contexts=num_contexts, test_name='linear_test3_noise_1_00')

benchmarker.run_experiments(50)
benchmarker.save_results('./results/')
benchmarker.display_results(save_path='./results/')
