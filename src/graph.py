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

        
        plt.scatter(x, y, c=self.test_label, cmap=plt.get_cmap('Paired'))
        plt.colorbar()
        plt.show()

        print((x[0], y[0]))
        pass

    def generate_input_output_plot(self):
        pass