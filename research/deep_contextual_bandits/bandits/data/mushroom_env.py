import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder


def get_labels_contexts():
	'''
	:return hot_encoded arrays: labels (n_samples, 2) and contexts (n_samples, n_hot_encoded_features)
	'''
	with open("/home/pierre/Documents/Info/DeepRL/data/agaricus-lepiota.data") as f:
		buffer = []
		for line in f:
			buffer.append(line.split(','))
		buffer = np.array(buffer)
		encoded_data = one_hot_encoding(buffer[:, 1:])
	is_e = np.array((buffer[:, 0]=="e"), dtype=float)
	return is_e, encoded_data


def one_hot_encoding(buffer):
	enc = OneHotEncoder(sparse=False, )
	encoded_data = enc.fit_transform(buffer)
	return encoded_data


class Environment():
	def __init__(self):
		"""Initialize the environment."""
		pass

	def get_observation(self):
		"""Returns an observation from the environment."""
		pass

	def get_optimal_reward(self):
		"""Returns the optimal possible reward for the environment at that point."""
		pass

	def get_expected_reward(self, action):
		"""Gets the expected reward of an action."""
		pass

	def get_stochastic_reward(self, action, label):
		"""Gets a stochastic reward for the action."""
		pass

	def advance(self, action, reward):
		"""Updating the environment (useful for nonstationary bandit)."""
		pass


class Mushrooms(Environment):
	def __init__(self, r_idle=0., r_guy_is_fine=5., r_guy_is_poisoned=-35., pr_poisoned=0.5):
		# default rewards values taken from the Deep Bandits article
		Environment.__init__(self)
		self.r_idle = r_idle
		self.r_guy_is_fine = r_guy_is_fine
		self.r_guy_is_poisoned = r_guy_is_poisoned
		self.pr_poisoned = pr_poisoned
		labels, contexts = get_labels_contexts()
		nb_actions, nb_features = 2, contexts.shape[-1]
		n_rows = nb_actions + nb_features
		self.table = np.empty((len(labels), n_rows))
		self.table[:, :nb_features] = contexts
		self.table[:, nb_features] = self.get_stochastic_rewards(np.zeros_like(labels), labels)
		self.table[:, nb_features+1] = self.get_stochastic_rewards(np.ones_like(labels), labels)

		self.opts = np.concatenate((np.expand_dims(self.get_stochastic_rewards(labels, labels),axis=1),
									np.expand_dims(labels, axis=1)), axis=1)
	#
	# def pick_mushrooms(self, n_samples):
	# 	nb_mushrooms = len(self.labels)
	# 	picked = np.random.choice(range(nb_mushrooms), n_samples)
	# 	contexts = np.take(self.contexts, indices=picked, axis=0)
	# 	return labels, contexts

	def get_stochastic_rewards(self, actions, labels):
		poisoned = (np.random.random(len(labels)) < self.pr_poisoned) *(1-labels)
		return actions*(5-poisoned*40)

	def get_stochastic_regret(self, labels, actions):
		return self.get_optimal_reward(labels) - self.get_stochastic_reward(labels, actions)


if __name__ == '__main__':
	mush = Mushrooms()
	print(np.min(mush.opts[:, 0]))


