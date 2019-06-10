from argparse import ArgumentParser
import os
import pickle
import sys
from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_20newsgroups

try:
    from impatial_text_cls.impatial_text_cls import ImpatialTextClassifier
    from impatial_text_cls.utils import str_to_layers
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from impatial_text_cls.impatial_text_cls import ImpatialTextClassifier
    from impatial_text_cls.utils import str_to_layers


def load_data(subset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    data = fetch_20newsgroups(data_home=os.path.join(os.path.dirname(__file__), 'data'), subset=subset_name)
    if data is None:
        raise ValueError('Data for training and testing cannot be downloaded!')
    return np.array(
        [' '.join(list(filter(
            lambda it2: len(it2) > 0,
            map(lambda it1: it1.strip(), cur.lower().split())
        ))) for cur in data['data']], dtype=object
    ), data['target']


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The binary file with the text classifier.')
    parser.add_argument('--layers', dest='sizes_of_layers', type=str, required=False, default='1000-1000',
                        help='Sizes of the Bayesian neural network layers.')
    parser.add_argument('--num_monte_carlo', dest='num_monte_carlo', type=int, required=False, default=10,
                        help='Number of generated Monte Carlo samples for each data sample.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, required=False, default=16,
                        help='Size of mini-batch.')
    parser.add_argument('--gpu_frac', dest='gpu_memory_frac', type=float, required=False, default=0.9,
                        help='Allocable part of the GPU memory for the classifier.')
    args = parser.parse_args()

    model_name = os.path.normpath(args.model_name)
    layers = str_to_layers(args.sizes_of_layers)

    if os.path.isfile(model_name):
        with open(model_name, 'rb') as fp:
            nn = pickle.load(fp)
    else:
        train_texts, train_labels = load_data('train')
        print('Number of samples for training is {0}.'.format(len(train_texts)))
        nn = ImpatialTextClassifier(hidden_layer_sizes=layers, batch_size=args.batch_size,
                                    num_monte_carlo=args.num_monte_carlo, gpu_memory_frac=args.gpu_memory_frac,
                                    verbose=True, multioutput=False, random_seed=42, validation_fraction=0.15,
                                    max_epochs=100, patience=5)
        nn.fit(train_texts, train_labels)
        print('')
        with open(model_name, 'wb') as fp:
            pickle.dump(nn, fp)
    test_texts, test_labels = load_data('test')
    print('')
    print('Number of samples for final testing is {0}.'.format(len(test_texts)))
    print('Test F1-macro is {0:.2%}.'.format(nn.score(test_texts, test_labels)))


if __name__ == '__main__':
    main()
