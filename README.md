[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bond005/impartial_text_cls/blob/master/LICENSE)
![Python 3.6, 3.7](https://img.shields.io/badge/python-3.6%20%7C%203.7-green.svg)

# impartial_text_cls
**Impartial Text Classifier**: text classifier, based on the BERT and a Bayesian neural network, which can train on small labeled texts and doubt its decision.

The goal of this project is developing of simple and power text classifier based on transfer learning and Bayesian neural networks. The important subtask of text classification is user's intent classification for chat-bots, information retrieval etc. And the intent classification task has two nuances:

1. If we solve a sentiment analysis task, then we always can attribute some input text to one of sentiment classes. If we solve a sentiment analysis task, then we always can attribute some input text to one of sentiment classes.  But in the intent classification the most part of all input texts isn't related to any intent, and such texts are "foreign", or some "background noise" (for example, questions about eastern philosophy instead of finance, deposits and ATMs to a bank chat-bot, or trolling attempts of a virtual assistant, and so on). Thus, capabilities of intent classifier for uncertainty and rejecting at recognition are very important.

2. Intents are very specific for each chat-bot development task. So, set of recognizable user's intents in chat-bot for pizza delivery service will be differ from analogous set of user's intents in bank chat-bot. Intents are not standard objects for recognition in contrast to sentiments. Therefore we cannot build a large standard corpus of user's intents, and size of any practical dataset, annotated by set of user's intents, will be small.

A transfer learning (particularly, well-known BERT) and Bayesian neural networks help to account these nuances.

[BERT](https://arxiv.org/abs/1810.04805) (**B**idirectional **E**ncoder **R**epresentations from **T**ransformers) generates special contextual embeddings for text tokens, which provide a better discrimination ability in feature space, than classical word embeddings. Therefore we can use smaller labeled data for training of final classifier.

[Bayesian neural network](https://arxiv.org/abs/1505.05424v2) in final classifier models uncertainty in data, owing to this fact probabilities of recognized classes returned by this network are more fair, and bayesian neural network is more robust to overfitting.

Installing
----------


For installation you have to use Python 3.6 or later. To install this project on your local machine, you should run the following commands in the Terminal:

```
git clone https://github.com/bond005/impartial_text_cls.git
cd impartial_text_cls
sudo python setup.py install
```

If you want to install the **Impartial Text Classifier** into a some virtual environment, than you don't have to use `sudo`, but before installing you have to activate this virtual environment (for example, using `source /path/to/your/python/environment/bin/activate` in the command prompt).

You can also run the tests

```
python setup.py test
```

The **Impartial Text Classifier** requires tensorflow library for its working (see `requirements.txt`). We recommend to install `tensorflow-gpu` for fast training, but you can use `tensorflow` for CPU in the inference mode for pre-trained models of the **Impartial Text Classifier** (certainly, you can use CPU for training too, but it is not a good idea).

Usage
-----

After installing the **Impartial Text Classifier** can be used as Python package in your projects. For example, you can apply this classifier for [the 20 newsgroups dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html):

```python
import pickle
from sklearn.datasets import fetch_20newsgroups  # import dataset for experiments
from sklearn.metrics import classification_report  # import evaluation module
from impartial_text_cls.impartial_text_cls import ImpartialTextClassifier  # import the module with classifier

# Create new classifier for English language
cls = ImpartialTextClassifier(
    bert_hub_module_handle='https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
    filters_for_conv2=50, filters_for_conv3=50, filters_for_conv4=50, filters_for_conv5=50,
    hidden_layer_size=1000, n_hidden_layers=2, batch_size=16, num_monte_carlo=100,
    bayesian=True, multioutput=False, random_seed=42, validation_fraction=0.15,
    max_epochs=100, patience=5, verbose=True, gpu_memory_frac=0.95,
    kl_weight_init=0.5, kl_weight_fin=1e-2
)

# Load and prepare dataset for training
data_for_training = fetch_20newsgroups(subset='train')
X_train = [' '.join(cur.split()) for cur in data_for_training['data']]
y_train = [data_for_training['target_names'][class_idx]
           for class_idx in data_for_training['target']]

# Load and prepare dataset for final testing
data_for_testing = fetch_20newsgroups(subset='test')
X_test = [' '.join(cur.split()) for cur in data_for_testing['data']]
y_test = data_for_testing['target']

# Train classifier
cls.fit(X_train, y_train)

# Evaluate classifier
y_pred = cls.predict(X_test)
print(classification_report(y_test, y_pred))

# Save classifier for further usage
with open('bert_bayesian_for_20newsgroups.pkl', 'wb') as fp:
    pickle.dump(cls, fp)
``````

In this example we created classifier with special pre-trained BERT for English language from the **TensorFlow Hub**, and we specified path to this BERT in the `bert_hub_module_handle` parameter of constructor.

Neural architecture of the impartial text classifier is shown in the next figure (source of the BERT image can be found in [this paper](https://arxiv.org/pdf/1810.04805.pdf)). 

![][nn_structure]

[nn_structure]: images/bert_bayesian_nn.png "Structure of the convolutional bayesian neural network with BERT as feature extractor"

BERT is used as generator of token embeddings and whole text embedding. The sequence output of BERT is used for token embeddings calculation, and the pooled output generates text embedding. Therefore we add convolutional neural network in [Yoon Kim's style](https://arxiv.org/abs/1408.5882) after BERT's sequence output and concatenate outputs of this convolutional neural network with BERT's pooled output. The first and only convolutional layer of this network contains feature maps with multiple filter widths from 1 to 5. Feature map quantity for each filter width is specified by the `filters_for_conv1` ... `filters_for_conv5` parameters. After average-over-time pooling all outputs of convolutional layer are concatenated with the pooled output of BERT, and a resulting signal are processed by sequence of hidden layers the size of the `hidden_layer_size` (optionally, because number of hidden layers, specified by the `n_hidden_layers`, can be zero). Besides, boolean parameter `bayesian` specifies kind of all weights in the above described convolutional network: if `bayesian` is True, then these weights are bayesian, i.e. stochastic, else they are usual.

The `num_monte_carlo` parameter corresponds to number of samples from bayesian neural network in the inference mode. The large value of this parameter is better, but at the same time procedure of inference can become a little slower.

The `kl_weight_init` and `kl_weight_fin` determine initial and final points of a KL weight changing schedule for the bayesian neural network (see chapter 3.4 of [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424v2) about the KL weighting). If these values are same, then the KL weight is constant.

In the training process we need early stopping: we calculate some quality criterion on independent subset of data, called as validation dataset, and monitor its changing by epochs. If the value of this criterion become decrease during `patience` epochs on end, then we have to stop. Fraction of training data which will be randomly single out for validation is determined by the `validation_fraction` parameter. But if early stopping will not work, then we continue training process no more than `max_epochs` epochs, whereupon we stop, in spite of everything.

In both modes (training and inference) we don't process whole dataset at once, but we divide it by fixed-size mini-batches. It is especially urgent for calculations on GPU, because large dataset can not be upload to the GPU memory, and we have to process such datasets by small parts (mini-batches). The used size of one mini-batch is described by the `batch_size` parameter. And value of the `gpu_memory_frac` parameter corresponds to fraction of all GPU memory which must be allowed for our neural network.

In aforecited example we solve a topic classification problem, and each text can be related to one class only. But in real life some texts can be related to several classes (for example, in the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#description) many texts had two or three toxicity classes at once). In such case the `multioutput` parameter have to set in _True_, and some class labels (items of `y_train` and `y_test`) can be sequences (sets) of integers instead of single integer values (at present the `multioutput` parameter is set in _False_, and all items of `y_train` and `y_test` are just integers).

Training can be logged in stdout, and logging mode is specified by the `verbose` parameter. If this parameter is _True_, then important events of training process are logged, else neural network is trained in silent mode.

Experiments and discussion
-----

We think that addition of bayesian neural network after frozen BERT allows to do rejecting at recognition (using probability distribution of recognized classes) more efficiently than in case of usual non-stochastic neural network instead of bayesian one. But we need experimental verification of this hypothesis. For that we prepared some experiments with [the SNIPS-2017 dataset](https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines), the **Brown corpus** and the **Genesis corpus** from the [NLTK Corpora Collection](http://www.nltk.org/nltk_data/). In all experiments we regarded the **SNIPS-2017** as labeled corpus for intent classification (there are 7 classes of intents), and we used texts of the **Brown and Genesis corpuses** as typical non-intents, i.e. foreign texts for intent classification task. At that we divided the **SNIPS-2017** by training and testing subsets according to scheme proposed by authors of this corpus, the **Brown corpus** was used for training (and validation) only, and the **Genesis corpus** need for final testing.

We realized three experiments:

1. With Bayesian neural network and rejecting at recognition for non-intents by probability threshold

2. With usual neural network and rejecting at recognition for non-intents by probability threshold

3. With usual neural network and modeling of non-intents as yet another class (i.e. we trained neural network to recognize 8 classes instead of 7 ones)

We used F1 measure as quality criterion for final testing, at that we accounted non-intents as additional class. The results of final testing with macro-averaging for all intents and for the non-intent class are described in following table.

| Algorithm name | F1-macro for intents only | F1-macro for intents and non-intents |
| -------------- | ------------------------: | ----------------------------: |
| Bayesian neural network with rejecting at recognition | 0.9771 | 0.7192 |
| Usual neural network with rejecting at recognition | 0.9728 | 0.5108 |
| Usual neural network with yet another class for non-intents | - | 0.5309 |

Also you can see more detailed results by separate classes:

| F1 by intents | Bayesian neural network with rejecting at recognition | Usual neural network with rejecting at recognition | Usual neural network with yet another class for non-intents |
| ------------- | ----------------------------------------------------: | -------------------------------------------------: | ----------------------------------------------------------: |
| _AddToPlaylist_ | 0.8506 | 0.7671 | 0.4962 |
| _BookRestaurant_ | 0.7841 | 0.4541 | 0.6978 |
| _GetWeather_ | 0.8526 | 0.5289 | 0.6092 |
| _PlayMusic_ | 0.7152 | 0.4863 | 0.4569 |
| _RateBook_ | 0.7036 | 0.3745 | 0.4808 |
| _SearchCreativeWork_ | 0.1812 | 0.0248 | 0.0715 |
| _SearchScreeningEvent_ | 0.6750 | 0.5455 | 0.4759 |
| FOREIGN (non-intent) | 0.9914 | 0.9050 | 0.9589 |   


As you see, rejecting at recognition by probability of recognized class with the Bayesian neural network is better than any variant with the usual neural network. Both types of neural network (Bayesian and usual) are same if we don't account non-intents. But if we add non-intents as additional class in the test set, then there are evident differences between various types of neural networks. So, there is reason to suppose that described experiments corroborate initial hypothesis about effectiveness of the Bayesian neural network.

All aforecited experiments organize as special Python script which is available in the `demo/snips2017` subdirectory. Also, in the `demo` subdirectory there are several another demo-scripts, which may be helpful for various experiments and for better understanding of working with the **Impartial Text Classifier**.

Breaking Changes
-----

**Breaking changes in version 0.0.4**

- security vulnerabilities in dependencies have been eliminated. 

**Breaking changes in version 0.0.3**
- hidden layers has been added (but number of hidden layers can be zero, and in this case structure of neural network come same as previous version);
- an average-over-time pooling with masking has been come to use now instead of a max-over-time one (special masks are applied after convolution layer output for more correctly averaging);
- all outputs of BERT are used now: sequence outputs are processed by convolution neurons with various kernel sizes, as it was in previous version, and pooled outputs of BERT are concatenated with outputs of convolutional neurons after their average-over-time pooling;
- the SNIPS-2017 demo has been improved; in particular, a special shell script to run experiment series with various hyper-parameters of NN has been implemented, and more appropriate text corpuses have become used as "foreign" texts.

**Breaking changes in version 0.0.2**
- logging become more pretty: particularly, full class names may be printed instead of class indices in the training process (if you specify `y` as sequence of text labels).

**Breaking changes in version 0.0.1**
- initial (alpha) version of the Impartial Text Classifier has been released.
 
License
-----

The **Impartial Text Classifier** (`impartial-text-cls`) is Apache 2.0 - licensed.

Acknowledgment
-----


The work was supported by National Technology Initiative and PAO Sberbank project ID 0000000007417F630002.
