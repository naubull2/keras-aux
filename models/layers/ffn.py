#! coding: utf-8
"""
__author__ = 'Dan (naubull2@snu.ac.kr)'

Functional API for
* Positionwise Feedforward layer
  - https://arxiv.org/abs/1706.03762

"""
from keras.layers import Add, Conv1D, Dropout

from .normalization import LayerNormalization


class PositionwiseFeedForward():
    """Feed forward connection per unit
    Same as two convolution with kernel size 1,
    where inner convolution has relu activation.
    """
    def __init__(self, n_units=None, dropout=0.1):
        """n_units would usually be [4 * ndim, ndim]
        where ndim is input dimension
        """
        if n_units is None:
            n_units = [2048, 512]
        self.conv_1 = Conv1D(n_units[0], 1, activation='relu')
        self.conv_2 = Conv1D(n_units[1], 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        outputs = self.conv_1(x)
        outputs = self.conv_2(outputs)
        outputs = self.dropout(outputs)

        # Residual connection
        outputs = Add()([outputs, x])

        # Normalize
        outputs = self.layer_norm(outputs)

        return outputs
