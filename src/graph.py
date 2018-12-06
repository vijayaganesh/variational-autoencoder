import argparse
import os
import sys
import cv2
import numpy as np

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



        pass