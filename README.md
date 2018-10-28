# Collection of Keras models/layers for NLP

## Description

* Studying famous neural network based NLP papers
* Keras port for anything that come handy for quick prototyping and experimenting
* Provides de-facto standard models/layers in a simple, functional API
	* Transformer-encoder

## Requirements

* Keras >= 2.1.1
* Tensorflow >= 1.4.1
* h5py >= 2.8.0

## Notes

* Transformer Encoder
	* Reduces sequential computation compared to RNN.
	* Learn dependency between two symbols independently of their positional distance in the  sequence.
	* Parallize very well for fast training.


## Example

* `text_classifier.py` : A simple text classifier for testing out the Transformer encoder

* `Dataset` : [NLU benchmark dataset](https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines)

### Sample run

```
Train on 9690 samples, validate on 2422 samples
Epoch 1/10
9690/9690 [==============================] - 6s 620us/step - loss: 1.9247 - acc: 0.1659 - val_loss: 1.8820 - val_acc: 0.1623
Epoch 2/10
9690/9690 [==============================] - 4s 409us/step - loss: 1.3544 - acc: 0.4215 - val_loss: 0.7585 - val_acc: 0.6821
Epoch 3/10
9690/9690 [==============================] - 4s 409us/step - loss: 0.6125 - acc: 0.7717 - val_loss: 0.4927 - val_acc: 0.8493

...

Epoch 9/10
9690/9690 [==============================] - 4s 409us/step - loss: 0.0645 - acc: 0.9808 - val_loss: 0.2534 - val_acc: 0.9401
Epoch 10/10
9690/9690 [==============================] - 4s 410us/step - loss: 0.0656 - acc: 0.9798 - val_loss: 0.2264 - val_acc: 0.9414

Send my location to my little brother
<RequestRide> 0.23%
What's the nearest chicken shop?
<GetPlaceDetails> 0.35%
What's the cheapest place between The Chickens and the Porks?
<GetPlaceDetails> 0.22%
Show me photos of tonight's dinner schedule
<SearchScreeningEvent> 1.00%
Is Grills better than the Downtowner?
<ShareCurrentLocation> 0.19%
Find Lion King
<SearchScreeningEvent> 1.00%
Is the new Starwars movie playing at the nearest cinema?
<SearchScreeningEvent> 1.00%
Do I need an unbrella tomorrow?
<BookRestaurant> 0.99%
Will it rain tonight?
<GetWeather> 1.00%
```

## Acknowledgements

* The original paper, [Attention is all you need](https://arxiv.org/abs/1706.03762)
* Official [tensor2tensor repo](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)
* Lsdefine's [Transformer seq2seq model](https://github.com/Lsdefine/attention-is-all-you-need-keras)
