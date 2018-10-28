#! coding: utf-8
"""
__author__ = 'Dan (naubull2@snu.ac.kr)'

Functional API for
* Multihead-scaled dot product attention

For simplicity, masking sequence padding and causality(future blinding)
mask creation is taken out to module mask.py

"""
import numpy as np
import tensorflow as tf
from keras import backend as K

from keras.layers import Add, Lambda, Activation, Dropout, TimeDistributed, Dense

from .normalization import LayerNormalization


class ScaledDotProductAttention():
    """Scaled Dot Product Attention
    Introduced in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, n_units, dropout=0.1):
        self.scale = np.sqrt(n_units)
        self.dropout = Dropout(dropout)

    def __call__(self, q, k, v, mask):
        # Scaled mat-mul on Q, K
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.scale)([q, k])

        # Assumption: 0 are used for padding sequences
        if mask is not None:
            # mask for softmax and future blinding
            paddings = Lambda(lambda x: (-2**32 + 1) * (1-x))(mask)
            attn = Add()([attn, paddings])

        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)

        outputs = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])

        return outputs, attn


class MultiHeadAttention():
    """Multi Head Attention
    Introduced in https://arxiv.org/pdf/1706.03762.pdf
    This is a wrapper for Scaled Dot Product Attention

    For model code simplicity, use n_units for K and V's dimensions
    instead of using dim_k, dim_v
    """
    def __init__(self, n_head, n_units, dropout):
        self.n_head = n_head
        self.dropout = dropout
        self.n_units = n_units

        self.Q_layer = Dense(n_head*n_units, activation='relu', use_bias=False)
        self.K_layer = Dense(n_head*n_units, activation='relu', use_bias=False)
        self.V_layer = Dense(n_head*n_units, activation='relu', use_bias=False)

        self.attention = ScaledDotProductAttention(n_units)

        self.layer_norm = LayerNormalization()

        self.linear_out = TimeDistributed(Dense(n_units))

    def __call__(self, q, k, v, mask=None):
        n_head = self.n_head

        Q_ = self.Q_layer(q)  # [N, T_q, n_head*n_units]
        K_ = self.K_layer(k)
        V_ = self.V_layer(v)

        def split_heads(x):
            input_shape = tf.shape(x)   # [N, T_q, n_head*n_units]
            x = tf.reshape(x, [input_shape[0], input_shape[1], n_head, self.n_units])
            x = tf.transpose(x, [2, 0, 1, 3])
            x = tf.reshape(x, [n_head * input_shape[0], input_shape[1], -1])
            return x

        Q_ = Lambda(split_heads)(Q_)
        K_ = Lambda(split_heads)(K_)
        V_ = Lambda(split_heads)(V_)

        if mask is not None:
            mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
        head, attn = self.attention(Q_, K_, V_, mask=mask)

        def merge_heads(x):
            s = tf.shape(x)   # [n_head*N, T_v, n_units]
            x = tf.reshape(x, [n_head, -1, s[1], s[2]])
            x = tf.transpose(x, [1, 2, 0, 3])
            x = tf.reshape(x, [-1, s[1], n_head*self.n_units])  # [N, T_v, n_head*n_units]
            return x

        head = Lambda(merge_heads)(head)

        outputs = self.linear_out(head)
        outputs = Dropout(self.dropout)(outputs)

        # Residual connection
        outputs = Add()([outputs, q])

        # Layer norm
        outputs = self.layer_norm(outputs)

        return outputs, attn
