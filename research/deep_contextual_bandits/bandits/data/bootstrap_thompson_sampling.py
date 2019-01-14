import numpy as np


def generate_artificial_data(n_samples, n_features=112):
	contexts = np.random.randint(0, 2, (n_samples, n_features))
	actions = np.random.randint(0, 2, n_samples)
	rewards = np.random.randint(0, 10, n_samples)
	return np.asarray(contexts, dtype=float), actions, rewards


if __name__ == '__main__':
	print(generate_artificial_data(10))