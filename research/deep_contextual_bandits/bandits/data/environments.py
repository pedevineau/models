import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_labels_contexts_mushroom(path):
	'''
	:return hot_encoded arrays: labels (n_samples, 1) and contexts (n_samples, n_hot_encoded_features)
	'''
	with open(path) as f:
		buffer = []
		for line in f:
			buffer.append(line.split(','))
		buffer = np.array(buffer)
		encoded_data = one_hot_encoding(buffer[:, 1:])
	is_e = np.array((buffer[:, 0]=="e"), dtype=float)
	return is_e, encoded_data


def get_labels_contexts_covertype(path):
	with open(path) as f:
		buffer = []
		for line in f:
			buffer.append(line.split(','))
		buffer = np.array(buffer)
		encoded_data = one_hot_encoding(buffer[:, :-1])
	return buffer[:, -1], encoded_data


def one_hot_encoding(buffer):
	enc = OneHotEncoder(sparse=False, )
	encoded_data = enc.fit_transform(buffer)
	return encoded_data


class Environment:
	def __init__(self, path, nb_actions):
		"""Initialize the environment."""
		self.path = path
		self.nb_actions = nb_actions

	def get_optimal_rewards(self):
		"""Returns the optimal possible reward for the environment at that point."""
		pass

	def get_expected_rewards(self, action):
		"""Gets the expected reward of an action."""
		pass

	def get_stochastic_rewards(self, actions, labels):
		"""Gets a stochastic reward for the action."""
		pass


class Mushrooms(Environment):
	def __init__(self, r_idle=0., r_guy_is_fine=5., r_guy_is_poisoned=-35., pr_poisoned=0.5):
		# default rewards values taken from the Deep Bandits article
		Environment.__init__(self, path="datasets/agaricus-lepiota.data", nb_actions=2)
		self.r_idle = r_idle
		self.r_guy_is_fine = r_guy_is_fine
		self.r_guy_is_poisoned = r_guy_is_poisoned
		self.pr_poisoned = pr_poisoned
		labels, contexts = get_labels_contexts_covertype(self.path)
		nb_features = contexts.shape[-1]
		n_rows = self.nb_actions + nb_features
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
		return actions*(self.r_guy_is_fine-poisoned*(self.r_guy_is_fine+self.r_guy_is_poisoned))

	def get_stochastic_regret(self, actions, labels):
		return (self.opts[:, 0] - self.get_stochastic_rewards(actions, labels)).sum()


class Covertype(Environment):
	def __init__(self):
		# default rewards values taken from the Deep Bandits article
		Environment.__init__(self, path="datasets/covtype.data", nb_actions=7)

		labels, contexts = get_labels_contexts_covertype(self.path)
		nb_features = contexts.shape[-1]
		n_rows = self.nb_actions + nb_features
		self.table = np.empty((len(labels), n_rows))
		self.table[:, :nb_features] = contexts
		self.table[:, nb_features] = self.get_stochastic_rewards(np.zeros_like(labels), labels)
		self.table[:, nb_features+1] = self.get_stochastic_rewards(np.ones_like(labels), labels)

		self.opts = np.concatenate((np.expand_dims(self.get_stochastic_rewards(self, actions, labels), axis=1),
									np.expand_dims(labels, axis=1)), axis=1)

	def get_stochastic_rewards(self, actions, labels):
		return np.array(actions == labels, dtype=float)

	def get_stochastic_regret(self, actions, labels):
		return len(actions) - self.get_stochastic_rewards(actions, labels).sum()


if __name__ == '__main__':
	mush = Mushrooms()
	actions = np.ones(len(mush.opts))
	labels = np.ones()
	print(mush.get_stochastic_regret())


