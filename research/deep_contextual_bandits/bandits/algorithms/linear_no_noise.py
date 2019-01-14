# authors: pdevineau & mprouveur

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset

class LinearNoNoise(BanditAlgorithm):
	"""implements Thompson Sampling with independent linear models and no noise"""


def __init__(self, name, hparams):
	"""
	Assume a linear model withous noise for each action i: reward = context^T beta_i

	Args:
	  name: Name of the algorithm.
	  hparams: Hyper-parameters of the algorithm.
	"""

	self.name = name
	self.hparams = hparams

	# linear regression => minimization of (context * weight - reward)**2
	# after training,