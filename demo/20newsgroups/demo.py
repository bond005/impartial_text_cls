# Copyright 2019 Ivan Bondarenko
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser
import os
import pickle
import sys
from typing import Tuple, List

from sklearn.datasets import fetch_20newsgroups

try:
    from impartial_text_cls.impartial_text_cls import ImpatialTextClassifier
    from impartial_text_cls.utils import parse_hidden_layers_description
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from impartial_text_cls.impartial_text_cls import ImpatialTextClassifier
    from impartial_text_cls.utils import parse_hidden_layers_description


def load_data(subset_name: str) -> Tuple[List[str], List[str]]:
    data = fetch_20newsgroups(data_home=os.path.join(os.path.dirname(__file__), 'data'), subset=subset_name)
    if data is None:
        raise ValueError('Data for training and testing cannot be downloaded!')
    return [' '.join(list(filter(
        lambda it2: len(it2) > 0,
        map(lambda it1: it1.strip(), cur.lower().split())
    ))) for cur in data['data']], [data['target_names'][target_idx] for target_idx in data['target']]


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The binary file with the text classifier.')
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
    parser.add_argument('--hidden', dest='hidden_layer_size', type=str, required=False, default='500',
                        help='Size of each hidden layer and total number of hidden layers (separate them with colons).')
    parser.add_argument('--num_monte_carlo', dest='num_monte_carlo', type=int, required=False, default=10,
                        help='Number of generated Monte Carlo samples for each data sample.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, required=False, default=16,
                        help='Size of mini-batch.')
    parser.add_argument('--gpu_frac', dest='gpu_memory_frac', type=float, required=False, default=0.9,
                        help='Allocable part of the GPU memory for the classifier.')
    parser.add_argument('--nn_type', dest='nn_type', type=str, choices=['bayesian', 'usual'],
                        required=False, default='bayesian', help='Neural network type: `bayesian` or `usual`.')
    args = parser.parse_args()

    model_name = os.path.normpath(args.model_name)
    if os.path.isfile(model_name):
        with open(model_name, 'rb') as fp:
            nn = pickle.load(fp)
    else:
        hidden_layer_size, n_hidden_layers = parse_hidden_layers_description(args.hidden_layer_size)
        train_texts, train_labels = load_data('train')
        print('Number of samples for training is {0}.'.format(len(train_texts)))
        nn = ImpatialTextClassifier(filters_for_conv1=args.size_of_conv1, filters_for_conv2=args.size_of_conv2,
                                    filters_for_conv3=args.size_of_conv3, filters_for_conv4=args.size_of_conv4,
                                    filters_for_conv5=args.size_of_conv5, hidden_layer_size=hidden_layer_size,
                                    n_hidden_layers=n_hidden_layers, batch_size=args.batch_size,
                                    num_monte_carlo=args.num_monte_carlo, gpu_memory_frac=args.gpu_memory_frac,
                                    verbose=True, multioutput=False, random_seed=42, validation_fraction=0.15,
                                    max_epochs=100, patience=5, bayesian=(args.nn_type == 'bayesian'),
                                    kl_weight_init=0.05, kl_weight_fin=0.05)
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
