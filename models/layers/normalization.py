#! coding: utf-8
"""
__author__ = 'Dan (naubull2@snu.ac.kr)'

Layer Normalization
Introduced in https://arxiv.org/abs/1607.06450
(Jimmy Lei Ba et. al. 2016)
"""
from keras.engine.topology import Layer
from keras.initializers import (Ones, Zeros)
import keras.backend as K


class LayerNormalization(Layer):
    """Applies layer normalization
    """
    def __init__(self, eps=1e-8, **kwargs):
        # Set constant variables
        self.eps = eps
        self.beta = None
        self.gamma = None
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create model weights
        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[-1:],
                                    initializer=Zeros(),
                                    trainable=True)
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[-1:],
                                     initializer=Ones(),
                                     trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Forward computation
        mean = K.mean(inputs, axis=-1, keepdims=True)
        std = K.std(inputs, axis=-1, keepdims=True)
        normalized = (inputs - mean) / (std + self.eps)
        outputs = self.gamma * normalized + self.beta

        return outputs

    def compute_output_shape(self, input_shape):
        # return output shape for auto-build
        return input_shape
