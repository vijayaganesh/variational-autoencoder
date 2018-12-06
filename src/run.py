import argparse
import os
import sys
import cv2
import numpy as np

from network import VAENetwork
from graph import GraphGenerator
import mnist

MNIST_DATA_PATH = "../data"
MODEL_WEIGHT_DIRECTORY = "../saved_weights"

def main(args):
    mnist_data = None
    if os.path.exists(MNIST_DATA_PATH):
        mnist_data = mnist.MNIST(MNIST_DATA_PATH)
        
    else:
        print("MNIST data not found in the Data/ directory. Add the MNIST data to the path: ")
        print(os.path.join(os.getcwd(), '..', 'data'))

        sys.exit(1)

    x_train, _ = mnist_data.load_training()
    x_test, _ = mnist_data.load_testing()

    x_train = np.array([np.reshape(np.array(x, dtype=np.uint8), (28, 28, 1)) for x in x_train])
    x_test = np.array([np.reshape(np.array(x, dtype=np.uint8), (28, 28, 1)) for x in x_test])
    
    if not os.path.exists(MODEL_WEIGHT_DIRECTORY):
        os.makedirs(MODEL_WEIGHT_DIRECTORY, exist_ok=True)

    vae_obj = VAENetwork(x_train[2].shape)
    vae_model = vae_obj.get_model()
    # vae_obj.train(x_train, x_test)

    # vae_obj.save_weights(MODEL_WEIGHT_DIRECTORY)

    vae_obj.load_weights(MODEL_WEIGHT_DIRECTORY)
    vae_model.summary()

    graph_generator = GraphGenerator(vae_obj, mnist_data)
    graph_generator.generate_latent_plot()

    lat_model = vae_obj.encoder_model
    print(lat_model.predict(np.reshape(x_test[32], (1, 28, 28, 1))))

    img = vae_model.predict(np.reshape(x_test[32], (1, 28, 28, 1)))

    cv2.imshow('img', np.reshape(x_test[32],(28, 28)))  
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('img', np.reshape(np.array(img, dtype=np.uint8), (28, 28)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VAE on top of the MNIST Data.")  
    args = parser.parse_args()
    main(args)
