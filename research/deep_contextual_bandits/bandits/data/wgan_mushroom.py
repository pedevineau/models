# Wasserstein GAN adapted from https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np


class WGANMushroom:
	def __init__(self):
		self.n_features = 112
		self.n_noise = 100

		# Following parameter and optimizer set as recommended in paper
		self.n_critic = 5
		self.clip_value = 0.01
		optimizer = RMSprop(lr=0.00005)

		# Build and compile the critic
		self.critic = self.build_critic()
		self.critic.compile(loss=self.wasserstein_loss,
							optimizer=optimizer,
							metrics=['accuracy'])

		# Build the generator
		self.generator = self.build_generator()

		# The generator takes noise as input and generated imgs
		z = Input(shape=(self.n_noise,))
		img = self.generator(z)

		# For the combined model we will only train the generator
		self.critic.trainable = False

		# The critic takes generated images as input and determines validity
		valid = self.critic(img)

		# The combined model  (stacked generator and critic)
		self.combined = Model(z, valid)
		self.combined.compile(loss=self.wasserstein_loss,
							  optimizer=optimizer,
							  metrics=['accuracy'])

	def wasserstein_loss(self, y_true, y_pred):
		return K.mean(y_true * y_pred)

	def build_generator(self):

		model = Sequential()

		model.add(Dense(128, activation="relu", input_dim=self.n_noise))
		model.add(Dropout(0.25))
		# model.add(UpSampling2D())
		model.add(Dense(128, activation="relu"))
		model.add(Dropout(0.25))
		# model.add(Activation("relu"))
		# model.add(UpSampling2D())
		model.add(Dense(64, activation="relu"))
		model.add(Dropout(0.25))
		# model.add(BatchNormalization(momentum=0.8))
		# model.add(Activation("relu"))
		# model.add(Conv2D(self.channels))
		model.add(Dense(self.n_features, activation="sigmoid"))

		model.summary()

		noise = Input(shape=(self.n_noise,))
		img = model(noise)

		return Model(noise, img)

	def build_critic(self):

		model = Sequential()

		model.add(Dense(16, input_dim=self.n_features))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Dense(32))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Dense(64, activation="relu"))
		model.add(Dropout(0.25))
		model.add(Dense(128, activation="relu"))
		model.add(Dropout(0.25))
		# model.add(Flatten())
		model.add(Dense(1, activation="sigmoid"))

		model.summary()

		img = Input(shape=(self.n_features,))
		validity = model(img)

		return Model(img, validity)

	def train(self, epochs, batch_size=128, sample_interval=50):

		# Load the dataset
		# (X_train, _), (_, _) = mnist.load_data()
		from models.research.deep_contextual_bandits.bandits.data.environments import get_labels_contexts_mushroom
		# _, X_train = get_labels_contexts_mushroom(path="/home/pierre/Documents/Info/DeepRL/data/agaricus-lepiota.data")
		X_train = np.ones((1000, self.n_features))
		X_train[:, 0:60] = 0
		print(X_train)
		# Rescale -1 to 1
		# X_train = (X_train.astype(np.float32) - 127.5) / 127.5
		# X_train = np.expand_dims(X_train, axis=3)

		# Adversarial ground truths
		valid = -np.ones((batch_size, 1))
		fake = np.ones((batch_size, 1))

		for epoch in range(epochs):

			for _ in range(self.n_critic):

				# ---------------------
				#  Train Discriminator
				# ---------------------

				# Select a random batch of images
				idx = np.random.randint(0, X_train.shape[0], batch_size)
				imgs = X_train[idx]

				# Sample noise as generator input
				noise = np.random.normal(0, 1, (batch_size, self.n_noise))

				# Generate a batch of new images
				gen_imgs = self.generator.predict(noise)

				# Train the critic
				d_loss_real = self.critic.train_on_batch(imgs, valid)
				d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
				d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
				# print("piche", d_loss_real, d_loss_fake)

				# Clip critic weights
				for l in self.critic.layers:
					weights = l.get_weights()
					weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
					l.set_weights(weights)

			# ---------------------
			#  Train Generator
			# ---------------------

			g_loss = self.combined.train_on_batch(noise, valid)

			# Plot the progress
			print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

			# If at save interval => save generated image samples
			# if epoch % sample_interval == 0:
			# 	self.sample_images(epoch)

	def sample_images(self, epoch):
		r, c = 5, 5
		noise = np.random.normal(0, 1, (r * c, self.n_noise))
		gen_imgs = self.generator.predict(noise)
		print(gen_imgs)

		# Rescale images 0 - 1
		# gen_imgs = 0.5 * gen_imgs + 1

		# fig, axs = plt.subplots(r, c)
		# cnt = 0
		# for i in range(r):
		# 	for j in range(c):
		# 		axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
		# 		axs[i, j].axis('off')
		# 		cnt += 1
		# # fig.savefig("images/mnist_%d.png" % epoch)
		# plt.close()


if __name__ == '__main__':
	wgan = WGANMushroom()
	wgan.train(epochs=4000, batch_size=32, sample_interval=50)