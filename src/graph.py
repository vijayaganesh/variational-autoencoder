import argparse
import os
import sys
import cv2
import numpy as np
import random

from network import VAENetwork
import matplotlib.pyplot as plt

class GraphGenerator:
    def __init__(self, network_object, mnist_data):
        self.encoder_model = network_object.encoder_model
        self.model = network_object.model
        self.decoder_model = network_object.decoder_model
        test_data, test_label = mnist_data.load_testing()
        self.test_data = np.array([np.reshape(np.array(x, dtype=np.uint8), (28, 28, 1)) for x in test_data])
        self.test_label = np.array([np.array(x) for x in test_label])

    def generate_latent_plot(self):
        latent_vector = self.encoder_model.predict(self.test_data, batch_size=128)[2]
        print(latent_vector)
        x = np.zeros(len(latent_vector))
        y = np.zeros(len(latent_vector))

        for i,data in enumerate(latent_vector):
            x[i] = data[0]
            y[i] = data[1]

        plt.scatter(x, y, c=self.test_label, cmap=plt.get_cmap('Set1'))
        plt.colorbar()
        plt.show()

    def generate_reconstructed_image(self):

        n = 20
        digit_size = 28
        epsilon_std = 0.1
        figure = np.zeros((digit_size * n, digit_size * n))
        grid_x = np.linspace(-250, 50, n)
        grid_y = np.linspace(-125, 200, n)

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]]) * epsilon_std
                x_decoded = self.decoder_model.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                        j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure)
        plt.show()

    def generate_reconstructed_image(self):

        n = 20
        digit_size = 28
        epsilon_std = 0.1
        figure = np.zeros((digit_size * n, digit_size * n))
        grid_x = np.linspace(-250, 50, n)
        grid_y = np.linspace(-125, 200, n)

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]]) * epsilon_std
                x_decoded = self.decoder_model.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                        j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure)
        plt.show()

        pass

    def generate_reconstructed_image_random_samples(self):

        n = 20
        digit_size = 28
        epsilon_std = 1
        figure = np.zeros((digit_size * n, digit_size * n))

        #Get the space of latent vectors
        x_encoded = self.encoder_model.predict(self.test_data, batch_size=128)[2]
        min_vals = x_encoded.min(axis=0)
        max_vals = x_encoded.max(axis=0)
        lv_size = len(min_vals)

        # Pick a random set of 400 n-dim vectors
        samples = np.zeros((n*n,lv_size))
        for idx_vec in range(0,lv_size):
            #samples[:,idx_vec] = random.sample(range(min_vals[idx_vec], max_vals[idx_vec]), n*n)
            samples[:,idx_vec] = random.randrange(int(min_vals[idx_vec]), int(max_vals[idx_vec]), n*n)
            samples[:,idx_vec] = np.array([random.randint(int(min_vals[idx_vec]),int(max_vals[idx_vec])) for x in range(n*n)])


        print(samples.shape)

        for i in range(0,n):
            for j in range(0,n):
                x_decoded = self.decoder_model.predict(np.array([samples[i*n+j,:]])*epsilon_std)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                        j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure)
        plt.show()

        pass
