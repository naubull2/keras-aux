#! coding: utf-8
"""
__author__ = 'Dan (naubull2@snu.ac.kr)'

A working example of a text classifier using
a Keras-ported transformer as a sequence encoder.
"""
from __future__ import print_function

import pickle as pkl
import codecs
import sys
import os

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.models import Model
from keras.layers import Embedding
from keras.layers import (
    Dense, Input, Flatten, Dropout,
    Conv1D, MaxPooling1D, GRU
)

from models.transformer import TransformerEncoder
from models.layers.auxiliary import positional_encoding

os.environ['KERAS_BACKEND'] = 'tensorflow'

MAX_SEQUENCE_LENGTH = 100
MAX_WORDS = 20000
EMBEDDING_DIM = 128
VALIDATION_SPLIT = 0.2


### Define model
def build_model(word_index, class_names):
    """Construct computational graph
    """
    # You must specify input length if Conv. layer will be involved.
    sequence_input = Input(shape=(None,), dtype='int32')

    char_embed = Embedding(len(word_index)+1,
                           EMBEDDING_DIM,
                           trainable=True)

    position_embed = Embedding(MAX_SEQUENCE_LENGTH,
                               EMBEDDING_DIM,
                               trainable=False,
                               weights=[positional_encoding(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)])

    ### Transformer 2-block encoder + RNN for variable length handling 
    # Either call with flatten=True or specify input length from the Input above
    l_tfencoder = TransformerEncoder(128,
                          8,
                          layers=2,
                          word_emb=char_embed,
                          pos_emb=position_embed)(sequence_input)
    l_out = GRU(128)(l_tfencoder)

    ### Compare against a simple convolutional encoder
    #l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)
    #l_pool1 = MaxPooling1D(5)(l_cov1)
    #l_cov2 = Conv1D(128, 3, activation='relu')(l_pool1)
    #l_pool2 = MaxPooling1D(3)(l_cov2)
    #l_cov3 = Conv1D(128, 2, activation='relu')(l_tfenc2)
    #l_pool3 = MaxPooling1D(2)(l_cov3)
    #l_flat = Flatten()(l_pool3)

    l_dense = Dense(62, activation='relu')(l_out)
    preds = Dense(len(class_names), activation='softmax')(l_dense)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    print("model fitting - a simple text classifier")
    model.summary()
    return model

### Load corpus
def load_corpus(file_name='./data/train.tsv'):
    texts = []
    labels = []
    class_names = {}
    key = -1
    with codecs.open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            try:
                text, label = line.split('\t')
                texts.append(' '.join(text.split()))
                labeled = False
                for code, classname in class_names.items():
                    if classname == label:
                        labels.append(code)
                        labeled = True
                if not labeled:
                    key += 1
                    class_names[key] = label
                    labels.append(key)
            except:
                continue
    return texts, labels, class_names

# Check if pretrained model is available
if os.path.isfile('model.weight.hdf5'):
    _, _, class_names = load_corpus()
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pkl.load(handle)
    word_index = tokenizer.word_index
    model = build_model(word_index, class_names)
    model.load_weights('model.weight.hdf5')
else:
    # Train a model
    texts, labels, class_names = load_corpus()

    tokenizer = Tokenizer(num_words=MAX_WORDS, char_level=False)
    tokenizer.fit_on_texts(texts) # update internal dictionary
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('%s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data:', data.shape)
    print('Shape of label:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    with open('tokenizer.pkl', 'wb') as handle:
        pkl.dump(tokenizer, handle)

    model = build_model(word_index, class_names)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128)

    model.save_weights('model.weight.hdf5')

while True:
    try: line = sys.stdin.readline()
    except: break
    if not line: break

    if isinstance(line, str):
        line = line.decode('utf-8').rstrip('\n')

    seq = tokenizer.texts_to_sequences([line])
    seq = pad_sequences(seq, maxlen=len(line))
    pred = model.predict(seq)[0]
    best = np.argmax(pred)
    prob = pred[best]
    print('<{}> {:.2f}%'.format(class_names[best], prob))
