[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bond005/impartial_text_cls/blob/master/LICENSE)
![Python 3.6, 3.7](https://img.shields.io/badge/python-3.6%20%7C%203.7-green.svg)

# impartial_text_cls
**Impartial Text Classifier**: text classifier, based on the BERT and a Bayesian neural network, which can train on small labeled texts and doubt its decision.

The goal of this project is developing of simple and power text classifier based on transfer learning and Bayesian neural networks. Important subtask of text classification is user's intent classification for chat-bots, information retrieval etc. And intent classification task has two nuances:

1. If we solve a sentiment analysis task, then we always can attribute any input text to one of sentiment classes.If we solve a sentiment analysis task, then we always can attribute any input text to one of sentiment classes.  But at intent classification most part of all input texts isn't related to any intent, and such texts are "foreign", or some "background noise" (for example, questions about eastern philosophy istead of finance, deposits and ATMs to a bank chat-bot, or trolling attempts of a virtual assistant, and so on). Thus, capabilities of intent classifier for uncertainty and rejecting at recognition are very important.

2. Intents are very specific for each chat-bot development task. So, set of recognizable user intents in chat-bot for pizza delivery service will be differ from analogous set of user intents in bank chat-bot. Intents are not standard objects for recognition in contrast to sentiments. Therefore we cannot build a large standard corpus of user intents, and size of any practical dataset, annotated by set of user intents, will be small.

A transfer learning (particulary, well-known BERT) and Bayesian neural networks help to account these nuances.

[BERT](https://arxiv.org/abs/1810.04805) (**B**idirectional **E**ncoder **R**epresentations from **T**ransformers) generates special contextual embeddings for text tokens, which provide a better discrimination ability in feature space, than classical word embeddings. Therefore we can use smaller labeled data for training of final classifier.

[Bayesian neural network](https://arxiv.org/abs/1505.05424v2) in final classifier models uncertainty in data, owing to this fact probabilities of recognized classes returned by this network are more fair, and bayesian neural network is more robust to overfitting.

Installing
----------


For installation you need to use Python 3.6 or later. To install this project on your local machine, you should run the following commands in the Terminal:

```
git clone https://github.com/bond005/impartial_text_cls.git
cd impartial_text_cls
sudo python setup.py install
```

If you want to install the **Impartial Text Classifier** into a some virtual environment, than you don't need to use `sudo`, but before installing you have to activate this virtual environment (for example, using `source /path/to/your/python/environment/bin/activate` in the command prompt).

You can also run the tests

```
python setup.py test
```

The **Impartial Text Classifier** requires tensorflow library for its working (see `requirements.txt`). We recommend to install `tensorflow-gpu` for fast training, but you can use `tensorflow` for CPU in the inference mode for pre-trained models of the **Impartial Text Classifier** (certainly, you can use CPU for training too, but it is not good idea).

Usage
-----

After installing the **Impartial Text Classifier** can be used as Python package in your projects. For example, you can apply this classifier for [the 20 newsgroups dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html):

```python
import pickle
from sklearn.datasets import fetch_20newsgroups  # import dataset for experiments
from sklearn.metrics import classification_report  # import evaluation module
from impatial_text_cls.impatial_text_cls import ImpatialTextClassifier  # import the module with classifier

# Create new classifier for English language
cls = ImpatialTextClassifier(
    bert_hub_module_handle='https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
    filters_for_conv2=50, filters_for_conv3=50, filters_for_conv4=50, filters_for_conv5=50,
    batch_size=16, num_monte_carlo=100, gpu_memory_frac=0.95, verbose=True, multioutput=False,
    random_seed=42, validation_fraction=0.15, max_epochs=100, patience=5, bayesian=True
)

# Load and prepare dataset for training
data_for_training = fetch_20newsgroups(subset='train')
X_train = [' '.join(cur.split()) for cur in data_for_training['data']]
y_train = data_for_training['target']

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

BERT is used as generator of token embeddings, therefore we add convolutional neural network in [Yoon Kim's style](https://arxiv.org/abs/1408.5882) after this BERT. First and only convolutional layer of this network contains feature maps with multiple filter widths from 1 to 5. Feature map quanity for each filter width is specified by the `filters_for_conv1` ... `filters_for_conv5` paramaters. Besides, boolean parameter `bayesian` specifies kind of all weights in the above described convolutional network: if `bayesian` is True, then these weights are bayesian, i.e. stohastic, else they are usual.

The `num_monte_carlo` parameter corresponds to number of sampes from bayesian neural network in the inference mode. Large value of this parameter is better, but at the same time procedure of inference can become a little slower.

In the training process we need early stopping: we calculate some quality criterion on independent subset of data, called as validation dataset, and monitor its changing by epochs. If value of this criterion become decrease during `patience` epochs on end, then we have to stop. Fraction of training data which will be randomly sinlge out for validation is determined by the `validation_fraction` parameter. But if early stopping will not work, then we continue training process no more than `max_epochs` epochs, whereupon we stop, in spite of everything.

In both modes (training and inference) we don't process whole dataset at once, but we divide it by fixed-size mini-batches. It is especially urgent for calculations on GPU, because large dataset can not be upload to the GPU memory, and we have to process such datasets by small parts (mini-batches). Used size of one mini-batch is described by the `batch_size` parameter. And value of the `gpu_memory_frac` parameter corresponds to fraction of all GPU memory which must be allowed for our neural network.

In aforecited example we solve a topic classification problem, and each text can be related to one class only. But in real life some texts can be related to several classes (for example, in the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#description) many texts had two or three toxicity classes at once). In such case the `multioutput` parameter have to set in _True_, and some class labels (items of `y_train` and `y_test`) can be sequences (sets) of integers instead of single integer values (at present the `multioutput` parameter is set in _False_, and all items of `y_train` and `y_test` are just integers).

Training can be logged in stdout, and logging mode is specified by the `verbose` parameter. If this parameter is _True_, then important events of traning process are logged, else neural network is trained in silent mode.

Experiments and discussion
-----

We think that addition of bayesian neural network after frozen BERT allows to do rejecting at recognition (using probability distribution of recognized classes) more efficiently than in case of usual non-stohastic neural network instead of bayesian one. But we need experimental verification of this hypothesis. For that we prepared some experiments with [the SNIPS-2017 dataset](https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines), the **Brown corpus** and the **Genesis corpus** from the [NLTK Corpora Collection](http://www.nltk.org/nltk_data/). In all experiments we regarded the **SNIPS-2017** as labeled corpus for intent classification (there are 7 classes of intents), and we used texts of the **Brown and Genesis corpuses** as typical non-intents, i.e. foreign texts for intent classification task. At that we divided the **SNIPS-2017** by training and testging subsets according to scheme proposed by authors of this corpus, the **Brown corpus** was used for training (and validation) only, and the **Genesis corpus** need for final testing.

We realized three experiments:

1. With Bayesian neural network and rejecting at recognition for non-intents by probability threshold

2. With usual neural network and rejecting at recognition for non-intents by probability threshold

3. With usual neural network and modeling of non-intents as yet another class (i.e. we trained neural network to recognize 8 classes instead of 7 ones)

We used F1 measure as quality criterion for final testing, at that we accounted non-intents as additional class. The results of final testing with micro- and macro-averaging are described in following table.

| Algorithm name | F1-micro | F1-macro |
| -------------- | -------: | -------: |
| Bayesian neural network with rejecting at recognition | 0.99 | 0.88 |
| Usual neural network with rejecting at recognition | 0.98 | 0.89 |
| Usual neural network with yet another class for non-intents | 0.97 | 0.86 |

Also you can see more detailed results by separate classes:

| F1 by intents | Bayesian neural network with rejecting at recognition | Usual neural network with rejecting at recognition | Usual neural network with yet another class for non-intents |
| ------------- | ----------------------------------------------------: | -------------------------------------------------: | ----------------------------------------------------------: |
| _AddToPlaylist_ | 0.95 | 0.99 | 0.97 |
| _BookRestaurant_ | 0.81 | 0.91 | 0.98 |
| _GetWeather_ | 0.97 | 0.96 | 0.58 |
| _PlayMusic_ | 0.77 | 0.65 | 0.48 |
| _RateBook_ | 0.98 | 0.97 | 1.00 |
| _SearchCreativeWork_ | 0.75 | 0.73 | 0.97 |
| _SearchScreeningEvent_ | 0.79 | 0.93 | 0.91 |
| FOREIGN (non-intent) | 0.99 | 0.99 | 0.98 |   


As you see, rejecting at recognition by probability of recognized class is better than modeling of foreign data as additional class in training set. Results of Bayesian and usual neural networks with rejecting at recognition are like, but, as is well known, micro-averaging of F1-measure is more significant in case of class imbalance. Our dataset for final testing is sufficiently imbalanced, because it includes 100 test samples per each intent and more than 10000 unlabeled test samples from the Genesis corpus considered as non-intents. So, there is reason to suppose that described experiments corroborate initial hypothesis about effectiveness of the Bayesian neural network.

All aforecited experiments organize as special Python script which is available in the `demo/snips2017` subdirectory. Also, in the `demo` subdirectory there are several another demo-scripts, which may be helpful for various experiments and for better understanding of working with the **Impartial Text Classifier**.

Breaking Changes
-----

**Breaking changes in version 0.0.2**
- logging become more pretty: particulary, full class names may be printed instead of class indices in the training process (if you specify `y` as sequence of text labels).

**Breaking changes in version 0.0.1**
- initial (alpha) version of the Impartial Text Classifier has been released.
 
License
-----

The **Impartial Text Classifier** (`impartial-text-cls`) is Apache 2.0 - licensed.

Acknowledgment
-----


The work was supported by National Technology Initiative and PAO Sberbank project ID 0000000007417F630002.
