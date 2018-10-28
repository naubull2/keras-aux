#! coding: utf-8
"""
__author__ = 'Dan (naubull2@snu.ac.kr)'

Functional API for

* Positional encoding
  - https://arxiv.org/abs/1706.03762

* Pad / Causuality mask
  - Pad mask is used for attending only on the valid tokens
  - Causality mask is used for attending only on the past tokens
"""
import numpy as np
import tensorflow as tf

from keras import backend as K


def get_pos_seq(seq_input):
    """Get position offsets for each time step 
    Create a tensor of [0, 1, 2, 3...] in the batch form,
    for later positional embedding vector lookup.
    """
    T = tf.shape(seq_input)[1]
    pos = tf.tile(tf.expand_dims(tf.range(T), 0), [tf.shape(seq_input)[0], 1])
    return pos

def positional_encoding(max_len, d_emb):
    """Position encoding weight matrix
    Directly computed by
      sin(pos/(1e4^2i/d)) for 2i
      cos(pos/(1e4^2i/d)) for 2i+1
    returns a weight matrix of shape [max_len, d_emb]
    """
    pos_enc = np.array([
        [pos / np.power(10000, 2 * i / d_emb) for i in range(d_emb)]
        for pos in range(max_len)])

    pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2]) # dimension 2i
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2]) # dimension 2i+1
    return pos_enc

def padding_mask(q, k):
    """Mask of (Q * K^T) for softmax activation
    """
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask

def lower_triangle_mask(s):
    """Mask of lower triangular matrix for causality
    Future blinding in decoder
    """
    seq_len = tf.shape(s)[1]
    bs = tf.shape(s)[:2]
    mask = K.cumsum(tf.eye(seq_len, batch_shape=bs), 1)
    return mask
