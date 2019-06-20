from argparse import ArgumentParser
import os
import pickle
import sys

from nltk import sent_tokenize, word_tokenize
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report


try:
    from impatial_text_cls.impatial_text_cls import ImpatialTextClassifier
    from impatial_text_cls.utils import read_snips2017_data
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from impatial_text_cls.impatial_text_cls import ImpatialTextClassifier
    from impatial_text_cls.utils import read_snips2017_data


def load_unlabeled_data(subset_name: str) -> np.ndarray:
    data = fetch_20newsgroups(data_home=os.path.join(os.path.dirname(__file__), 'data'), subset=subset_name)
    if data is None:
        raise ValueError('Data for training and testing cannot be downloaded!')
    texts = [' '.join(list(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), cur.lower().split()))))
             for cur in data['data']]
    sentences = []
    for cur in texts:
        src = sent_tokenize(cur)
        lengths = list(map(lambda it: len(word_tokenize(it)), src))
        indices = list(filter(lambda idx: (lengths[idx] >= 5) and (lengths[idx] <= 30), range(len(src))))
        for idx in indices:
            sentences.append(src[idx])
    return np.array(sentences, dtype=object)


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The binary file with the text classifier.')
    parser.add_argument('-d', '--data_dir', dest='data_dir', type=str, required=True,
                        help='Path to the directory with SNIPS-2017 data (see `2017-06-custom-intent-engines` subfolder'
                             ' of the repository https://github.com/snipsco/nlu-benchmark).')
    parser.add_argument('--conv1', dest='size_of_conv1', type=int, required=False, default=20,
                        help='Size of the Bayesian convolution layer with kernel size 1.')
    parser.add_argument('--conv2', dest='size_of_conv2', type=int, required=False, default=20,
                        help='Size of the Bayesian convolution layer with kernel size 2.')
    parser.add_argument('--conv3', dest='size_of_conv3', type=int, required=False, default=20,
                        help='Size of the Bayesian convolution layer with kernel size 3.')
    parser.add_argument('--conv4', dest='size_of_conv4', type=int, required=False, default=20,
                        help='Size of the Bayesian convolution layer with kernel size 4.')
    parser.add_argument('--conv5', dest='size_of_conv5', type=int, required=False, default=20,
                        help='Size of the Bayesian convolution layer with kernel size 5.')
    parser.add_argument('--num_monte_carlo', dest='num_monte_carlo', type=int, required=False, default=10,
                        help='Number of generated Monte Carlo samples for each data sample.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, required=False, default=16,
                        help='Size of mini-batch.')
    parser.add_argument('--gpu_frac', dest='gpu_memory_frac', type=float, required=False, default=0.9,
                        help='Allocable part of the GPU memory for the classifier.')
    parser.add_argument('--nn_type', dest='nn_type', type=str, choices=['bayesian', 'usual', 'additional_class'],
                        required=False, default='bayesian',
                        help='Neural network type: `bayesian`, `usual` or `additional_class` (it is same as `usual` '
                             'but unlabeled samples are modeled as additional class).')
    args = parser.parse_args()

    model_name = os.path.normpath(args.model_name)
    data_dir = os.path.normpath(args.data_dir)

    train_data, val_data, test_data, classes_list = read_snips2017_data(data_dir)
    print('Classes list: {0}'.format(classes_list))
    print('Number of samples for training is {0}.'.format(len(train_data[0])))
    print('Number of samples for validation is {0}.'.format(len(val_data[0])))
    print('Number of samples for final testing is {0}.'.format(len(test_data[0])))
    print('')
    unlabeled_texts_for_training = load_unlabeled_data('train')
    unlabeled_texts_for_testing = load_unlabeled_data('test')
    print('Number of unlabeled (unknown) samples for training is {0}.'.format(len(unlabeled_texts_for_training)))
    print('Number of unlabeled (unknown) samples for final testing is {0}.'.format(len(unlabeled_texts_for_testing)))
    print('')

    if os.path.isfile(model_name):
        with open(model_name, 'rb') as fp:
            nn = pickle.load(fp)
    else:
        train_texts = np.concatenate((train_data[0], unlabeled_texts_for_training))
        train_labels = np.concatenate(
            (
                train_data[1],
                np.full(shape=(len(unlabeled_texts_for_training),),
                        fill_value=(len(classes_list) if args.nn_type == 'additional_class' else -1), dtype=np.int32)
            )
        )
        nn = ImpatialTextClassifier(filters_for_conv1=args.size_of_conv1, filters_for_conv2=args.size_of_conv2,
                                    filters_for_conv3=args.size_of_conv3, filters_for_conv4=args.size_of_conv4,
                                    filters_for_conv5=args.size_of_conv5, batch_size=args.batch_size,
                                    num_monte_carlo=args.num_monte_carlo, gpu_memory_frac=args.gpu_memory_frac,
                                    verbose=True, multioutput=False, random_seed=42, validation_fraction=0.15,
                                    max_epochs=100, patience=5, bayesian=(args.nn_type == 'bayesian'))
        nn.fit(train_texts, train_labels, validation_data=val_data)
        print('')
        with open(model_name, 'wb') as fp:
            pickle.dump(nn, fp)
    test_texts = np.concatenate((test_data[0], unlabeled_texts_for_testing))
    test_labels = np.concatenate(
        (
            test_data[1],
            np.full(shape=(len(unlabeled_texts_for_testing),),
                    fill_value=len(classes_list), dtype=np.int32)
        )
    )
    y_pred = nn.predict(test_texts)
    for sample_idx in range(len(y_pred)):
        if y_pred[sample_idx] < 0:
            y_pred[sample_idx] = len(classes_list)
    print('Results of {0}:'.format(
        'bayesian neural network' if args.nn_type == 'bayesian' else
        ('usual neural network' if args.nn_type == 'usual' else 'usual neural network with additional class')
    ))
    print(classification_report(test_labels, y_pred, target_names=classes_list + ['UNKNOWN']))


if __name__ == '__main__':
    main()
