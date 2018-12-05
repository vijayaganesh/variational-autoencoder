from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.utils import plot_model
from utils import reparametrization
from keras import backend

import numpy as np

class VAENetwork:

    BATCH_SIZE = 128
    KERNEL_SIZE = 3
    NUM_FILTERS = 16
    LATENT_DIMENSIONS = 2
    FINAL_LAYERS = 100
    NUM_EPOCHS = 30

    def __init__(self, input_shape):

        self.input = Input(input_shape)
        self.encoder_model = self.generate_encoder()
        self.encoder_conv_shape = None
        self.latent_vector_output = None
        self.decoder_model = self.generate_decoder()
        self.model = self.create_model()


    def create_model(self):
        
        reconstructed_image = self.decoder_model(self.encoder_model(self.input)[2])
        model = Model(self.input, reconstructed_image)

        reconstruction_loss = binary_crossentropy(backend.flatten(self.input),
                                                  backend.flatten(reconstructed_image))

        kl_loss = 1 + self.latent_vector_output[1] - backend.square(self.latent_vector_output[0]) - backend.exp(self.latent_vector_output[1])
        kl_loss = -backend.sum(kl_loss, axis=-1)/2

        model.add_loss(backend.mean(reconstruction_loss + kl_loss))
        model.compile(optimizer='adam')
        return model

    def generate_encoder(self):

        conv_layer_1 = Conv2D(filters=VAENetwork.NUM_FILTERS, 
                        kernel_size=VAENetwork.KERNEL_SIZE,
                        activation='relu',
                        strides=2,
                        padding='same')(self.input)
        
        conv_layer_2 = Conv2D(filters=VAENetwork.NUM_FILTERS*2, 
                        kernel_size=VAENetwork.KERNEL_SIZE,
                        activation='relu',
                        strides=2,
                        padding='same')(conv_layer_1)

        conv_layer_3 = Conv2D(filters=VAENetwork.NUM_FILTERS*4, 
                        kernel_size=VAENetwork.KERNEL_SIZE,
                        activation='relu',
                        strides=2,
                        padding='same')(conv_layer_2)

        self.encoder_conv_shape = backend.int_shape(conv_layer_3)

        fully_connected = Flatten()(conv_layer_3)
        fully_connected = Dense(VAENetwork.FINAL_LAYERS, activation='relu')(fully_connected)
        self.latent_vector_output = self.generate_latent_vector(fully_connected)
        encoder = Model(self.input, [self.latent_vector_output[0], self.latent_vector_output[1], self.latent_vector_output[2]])

        return encoder

    def generate_decoder(self):
        
        decoder_input = Input(shape=(VAENetwork.NUM_FILTERS))
        fully_connected = Dense(self.encoder_conv_shape[1]*self.encoder_conv_shape[2]*self.encoder_conv_shape[3], activation='relu')(decoder_input)
        deconv_input = Reshape((self.encoder_conv_shape[1], self.encoder_conv_shape[2], self.encoder_conv_shape[3]))(fully_connected)

    
        deconv_layer_1 = Conv2DTranspose(filters=VAENetwork.NUM_FILTERS*4,
                            kernel_size=VAENetwork.KERNEL_SIZE,
                            activation='relu',
                            strides=2,
                            padding='same')(deconv_input)

        deconv_layer_2 = Conv2DTranspose(filters=VAENetwork.NUM_FILTERS*2,
                            kernel_size=VAENetwork.KERNEL_SIZE,
                            activation='relu',
                            strides=2,
                            padding='same')(deconv_layer_1)

        deconv_layer_3 = Conv2DTranspose(filters=VAENetwork.NUM_FILTERS,
                            kernel_size=VAENetwork.KERNEL_SIZE,
                            activation='relu',
                            strides=2,
                            padding='same')(deconv_layer_2)
        
        reconstructed_image = Conv2DTranspose(filters=1,
                            kernel_size=VAENetwork.KERNEL_SIZE,
                            activation='relu',
                            strides=2,
                            padding='same')(deconv_layer_3)
                            
        decoder = Model(decoder_input, reconstructed_image)

        return decoder

    def train(self, train_data, test_data):
        self.model.fit(x=train_data, epochs=VAENetwork.NUM_EPOCHS, batch_size=VAENetwork.BATCH_SIZE,
                        validation_data=(test_data, None))

    def save_weights(self, save_dir):
        self.model.save_weights(save_dir+"/vae.h5", overwrite=True)

    def generate_latent_vector(self, fully_connected):

        mean = Dense(VAENetwork.LATENT_DIMENSIONS)(fully_connected)
        log_var = Dense(VAENetwork.LATENT_DIMENSIONS)(fully_connected)
        latent_vector = Lambda(reparametrization)([mean, log_var])
        return (mean, log_var, latent_vector)

    def get_model(self):
        return self.model
        