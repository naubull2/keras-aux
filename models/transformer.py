#! coding: utf-8
"""
__author__ = 'Dan (naubull2@snu.ac.kr)'

Keras Layer API for Transformer Encoder
"""
from keras import backend as K
from keras.layers import Add, Lambda

from .layers.normalization import LayerNormalization
from .layers.attention import MultiHeadAttention
from .layers.ffn import PositionwiseFeedForward
from .layers.auxiliary import padding_mask, get_pos_seq


class TransformerEncoderLayer():
    def __init__(self, n_units, n_head, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head,
                                                 n_units,
                                                 dropout=dropout)
        self.ffn_layer = PositionwiseFeedForward(n_units=[4*n_units, n_units],
                                                 dropout=dropout)

    def __call__(self, x, mask=None):
        outputs, attn = self.self_att_layer(x, x, x, mask=mask)
        outputs = self.ffn_layer(outputs)

        return outputs, attn

class TransformerEncoder():
    """Stacked multihead attention encoder
    """
    def __init__(self, n_units, n_head,
                 layers=6,
                 dropout=0.1,
                 word_emb=None,
                 pos_emb=None):
        self.embedding = word_emb
        self.pos_embedding = pos_emb
        self.layers = [TransformerEncoderLayer(n_units, n_head, dropout) for _ in range(layers)]

    def __call__(self, src_seq, encode_pos=True, return_att=False, flatten=False):
        enc = self.embedding(src_seq)
        # Add sinusoidal positional embedding
        if encode_pos:
            src_pos = Lambda(get_pos_seq)(src_seq)
            pos = self.pos_embedding(src_pos)
            enc = Add()([enc, pos])

        if return_att:
            atts = []

        mask = Lambda(lambda x: padding_mask(x, x))(src_seq)
        for encoder_layer in self.layers:
            enc, att = encoder_layer(enc, mask)
            if return_att:
                atts.append(att)

        # For Keras's auto shape inference, specify output shape if possible
        if flatten:
            assert(self.embedding.input_length is not None)
            assert(self.pos_embedding.input_length is not None)

            flat_dim = self.embedding.input_length * self.embedding.output_dim
            enc = Lambda(lambda x: K.reshape(x, [-1, flat_dim]))(enc)

        elif self.embedding.input_length is not None:
            output_shape = [-1, self.embedding.input_length, self.embedding.output_dim]
            enc = Lambda(lambda x: K.reshape(x, output_shape))(enc) 

        return (enc, atts) if return_att else enc


        
