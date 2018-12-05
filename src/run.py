import argparse
import os
import sys

from network import VAENetwork
import mnist

MNIST_DATA_PATH = "../data"

def main(args):
    mnist_data = None
    if os.path.exists(MNIST_DATA_PATH):
        mnist_data = mnist.MNIST(MNIST_DATA_PATH)
        
    else:
        print("MNIST data not found in the Data/ directory. Add the MNIST data to the path: ")
        print(os.path.join(os.getcwd(), '..', 'data'))

        sys.exit(1)

    x_train, _ = mnist.MNIST.load_training()
    x_test, _ = mnist.MNIST.load_testing()

    vae_obj = VAENetwork((x_train.shape[1], x_train.shape[1], 1))
    vae_model = vae_obj.create_model()
    vae_obj.train(x_train, x_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VAE on top of the MNIST Data.")  
    args = parser.parse_args()
    main(args)
