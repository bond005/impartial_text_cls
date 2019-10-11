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
import codecs
import os
import pickle
import sys
from typing import List

import numpy as np
from sklearn.metrics import classification_report
from skopt import gp_minimize
from skopt.space import Integer, Real


try:
    from impartial_text_cls.impartial_text_cls import ImpatialTextClassifier
    from impartial_text_cls.utils import read_csv
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from impartial_text_cls.impartial_text_cls import ImpatialTextClassifier
    from impartial_text_cls.utils import read_csv


def load_unlabeled_texts(file_name: str) -> np.ndarray:
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        texts = list(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), fp.readlines())))
    return np.array(texts, dtype=object)


def print_classes_distribution(y: np.ndarray, classes: List[str]):
    classes_distr = dict()
    for class_idx in range(len(classes)):
        classes_distr[class_idx] = 0
        for cur in y:
            if isinstance(cur, set):
                if class_idx in cur:
                    classes_distr[class_idx] += 1
            else:
                if cur == class_idx:
                    classes_distr[class_idx] += 1
    number_width = max([len(str(classes_distr[val])) for val in classes_distr.keys()])
    name_width = max([len(cur) for cur in classes])
    print('Distribution of classes in labeled dataset:')
    for class_idx, class_name in enumerate(classes):
        print('  {0:<{1}} {2:>{3}}'.format(class_name, name_width, classes_distr[class_idx], number_width))
    print('')


def main():

    def func(args):
        conv1_ = int(args[0])
        conv2_ = int(args[1])
        conv3_ = int(args[2])
        conv4_ = int(args[3])
        conv5_ = int(args[4])
        hidden_ = int(args[5])
        quality = 0.0
        print('Filters number for different convolution kernels: ({0}, {1}, {2}, {3}, {4})'.format(
            conv1_, conv2_, conv3_, conv4_, conv5_))
        if hidden_ > 0:
            print('Hidden layer size is {0}.'.format(hidden_))
        if nn_type == 'bayesian':
            init_kl_weight = float(args[6])
            fin_kl_weight = float(args[7])
            print('Optimal value of initial KL weight is {0:.6f}.'.format(init_kl_weight))
            print('Optimal value of final KL weight is {0:.6f}.'.format(fin_kl_weight))
        else:
            init_kl_weight = 1.0
            fin_kl_weight = 1.0
        if sum(args) == 0:
            return 1.0
        for fold_idx, (train_index, test_index) in enumerate(indices_for_cv):
            cls = ImpatialTextClassifier(bert_hub_module_handle=(None if os.path.exists(os.path.normpath(bert_handle))
                                                                 else bert_handle),
                                         filters_for_conv1=conv1_, filters_for_conv2=conv2_, filters_for_conv3=conv3_,
                                         filters_for_conv4=conv4_, filters_for_conv5=conv5_, hidden_layer_size=hidden_,
                                         multioutput=multioutput, gpu_memory_frac=gpu_memory_frac,
                                         num_monte_carlo=num_monte_carlo, verbose=False, random_seed=42, max_iters=100,
                                         patience=5, batch_size=16, bayesian=(nn_type == 'bayesian'),
                                         kl_weight_init=init_kl_weight, kl_weight_fin=fin_kl_weight)
            if os.path.exists(os.path.normpath(bert_handle)):
                cls.PATH_TO_BERT = os.path.normpath(bert_handle)
            train_texts = labeled_texts[train_index]
            train_labels = labels[train_index]
            train_index_, val_index = cls.train_test_split(train_labels, 0.1)
            val_texts = train_texts[val_index]
            val_labels = train_labels[val_index]
            if unlabeled_texts_for_training is None:
                train_texts = train_texts[train_index_]
                train_labels = train_labels[train_index_]
            else:
                train_texts = np.concatenate(
                    (
                        train_texts[train_index_],
                        unlabeled_texts_for_training
                    )
                )
                train_labels = np.concatenate(
                    (
                        train_labels[train_index_],
                        np.full(shape=(len(unlabeled_texts_for_training),), fill_value=-1, dtype=np.int32)
                    )
                )
            cls.fit(train_texts, train_labels, validation_data=(val_texts, val_labels))
            del train_texts, train_labels, val_texts, val_labels, train_index_, val_index
            if unlabeled_texts_for_testing is None:
                texts_for_final_testing = labeled_texts[test_index]
                labels_for_final_testing = labels[test_index]
            else:
                texts_for_final_testing = np.concatenate(
                    (
                        labeled_texts[test_index],
                        unlabeled_texts_for_testing
                    )
                )
                labels_for_final_testing = np.concatenate(
                    (
                        labels[test_index],
                        np.full(shape=(len(unlabeled_texts_for_testing),), fill_value=-1, dtype=np.int32)
                    )
                )
            instant_quality = cls.score(texts_for_final_testing, labels_for_final_testing)
            quality += instant_quality
            print('Fold {0}: {1:.6f}.'.format(fold_idx + 1, instant_quality))
            del cls, texts_for_final_testing, labels_for_final_testing
        quality /= float(len(indices_for_cv))
        print('Total quality = {0:.6f}.'.format(quality))
        print('')
        return -quality

    def score(args):
        conv1_ = int(args[0])
        conv2_ = int(args[1])
        conv3_ = int(args[2])
        conv4_ = int(args[3])
        conv5_ = int(args[4])
        hidden_ = int(args[5])
        print('Optimal filters number for different convolution kernels: ({0}, {1}, {2}, {3}, {4})'.format(
            conv1_, conv2_, conv3_, conv4_, conv5_))
        if hidden_ > 0:
            print('Optimal size of the hidden layer is {0}.'.format(hidden_))
        if nn_type == 'bayesian':
            init_kl_weight = float(args[6])
            fin_kl_weight = float(args[7])
            print('Optimal value of initial KL weight is {0:.6f}.'.format(init_kl_weight))
            print('Optimal value of final KL weight is {0:.6f}.'.format(fin_kl_weight))
        else:
            init_kl_weight = 1.0
            fin_kl_weight = 1.0
        print('')
        y_pred = []
        y_true = []
        unlabeled_is_added = False
        for train_index, test_index in indices_for_cv:
            cls = ImpatialTextClassifier(bert_hub_module_handle=(None if os.path.exists(os.path.normpath(bert_handle))
                                                                 else bert_handle),
                                         filters_for_conv1=conv1_, filters_for_conv2=conv2_, filters_for_conv3=conv3_,
                                         filters_for_conv4=conv4_, filters_for_conv5=conv5_, hidden_layer_size=hidden_,
                                         batch_size=16, gpu_memory_frac=gpu_memory_frac, verbose=True, random_seed=42,
                                         num_monte_carlo=num_monte_carlo, max_iters=100, patience=5,
                                         multioutput=multioutput, bayesian=(nn_type == 'bayesian'),
                                         kl_weight_init=init_kl_weight, kl_weight_fin=fin_kl_weight)
            if os.path.exists(os.path.normpath(bert_handle)):
                cls.PATH_TO_BERT = os.path.normpath(bert_handle)
            train_texts = labeled_texts[train_index]
            train_labels = labels[train_index]
            train_index_, val_index = cls.train_test_split(train_labels, 0.1)
            val_texts = train_texts[val_index]
            val_labels = train_labels[val_index]
            if unlabeled_texts_for_training is None:
                train_texts = train_texts[train_index_]
                train_labels = train_labels[train_index_]
            else:
                train_texts = np.concatenate(
                    (
                        train_texts[train_index_],
                        unlabeled_texts_for_training
                    )
                )
                train_labels = np.concatenate(
                    (
                        train_labels[train_index_],
                        np.full(shape=(len(unlabeled_texts_for_training),), fill_value=-1, dtype=np.int32)
                    )
                )
            cls.fit(train_texts, train_labels, validation_data=(val_texts, val_labels))
            print('')
            del train_texts, train_labels, val_texts, val_labels, train_index_, val_index
            if (not unlabeled_is_added) and (unlabeled_texts_for_testing is not None):
                y_pred.append(cls.predict(unlabeled_texts_for_testing))
                unlabeled_is_added = True
                y_true.append(np.full(shape=(len(unlabeled_texts_for_testing),), fill_value=-1, dtype=np.int32))
            y_pred.append(cls.predict(labeled_texts[test_index]))
            y_true.append(labels[test_index])
            del cls
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        print('')
        if multioutput:
            for class_idx in range(len(classes_list)):
                y_true_ = np.zeros((len(y_true),), dtype=np.int32)
                y_pred_ = np.zeros((len(y_pred),), dtype=np.int32)
                for sample_idx in range(len(y_true)):
                    if isinstance(y_true[sample_idx], set):
                        if class_idx in y_true[sample_idx]:
                            y_true_[sample_idx] = 1
                    elif class_idx == y_true[sample_idx]:
                        y_true_[sample_idx] = 1
                    if isinstance(y_pred[sample_idx], set):
                        if class_idx in y_pred[sample_idx]:
                            y_pred_[sample_idx] = 1
                    elif class_idx == y_pred[sample_idx]:
                        y_pred_[sample_idx] = 1
                print(classification_report(y_true, y_pred, target_names=['OTHER', classes_list[class_idx]], digits=4))
        else:
            for sample_idx in range(len(y_true)):
                if y_true[sample_idx] < 0:
                    y_true[sample_idx] = len(classes_list)
                if y_pred[sample_idx] < 0:
                    y_pred[sample_idx] = len(classes_list)
            print(classification_report(y_true, y_pred, target_names=classes_list + ['UNKNOWN'], digits=4))
            print('')

    def train(args) -> ImpatialTextClassifier:
        conv1_ = int(args[0])
        conv2_ = int(args[1])
        conv3_ = int(args[2])
        conv4_ = int(args[3])
        conv5_ = int(args[4])
        hidden_ = int(args[5])
        if nn_type == 'bayesian':
            init_kl_weight = float(args[6])
            fin_kl_weight = float(args[7])
        else:
            init_kl_weight = 1.0
            fin_kl_weight = 1.0
        train_index, val_index = ImpatialTextClassifier.train_test_split(labels, 0.1)
        if unlabeled_texts_for_training is None:
            train_texts = labeled_texts[train_index]
            train_labels = labels[train_index]
        else:
            train_texts = np.concatenate(
                (
                    labeled_texts[train_index],
                    unlabeled_texts_for_training
                )
            )
            train_labels = np.concatenate(
                (
                    labels[train_index],
                    np.full(shape=(len(unlabeled_texts_for_training),), fill_value=-1, dtype=np.int32)
                )
            )
        val_texts = labeled_texts[val_index]
        val_labels = labels[val_index]
        cls = ImpatialTextClassifier(bert_hub_module_handle=(None if os.path.exists(os.path.normpath(bert_handle))
                                                             else bert_handle),
                                     filters_for_conv1=conv1_, filters_for_conv2=conv2_, filters_for_conv3=conv3_,
                                     filters_for_conv4=conv4_, filters_for_conv5=conv5_, hidden_layer_size=hidden_,
                                     batch_size=16, gpu_memory_frac=gpu_memory_frac, num_monte_carlo=num_monte_carlo,
                                     verbose=True, random_seed=42, max_iters=1000, patience=5, multioutput=multioutput,
                                     bayesian=(nn_type == 'bayesian'),
                                     kl_weight_init=init_kl_weight, kl_weight_fin=fin_kl_weight)
        if os.path.exists(os.path.normpath(bert_handle)):
            cls.PATH_TO_BERT = os.path.normpath(bert_handle)
        cls.fit(train_texts, train_labels, validation_data=(val_texts, val_labels))
        del train_texts, train_labels, val_texts, val_labels
        return cls

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The binary file with the text classifier.')
    parser.add_argument('-b', '--bert', dest='bert', type=str, required=False,
                        default='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                        help='URL of used TF-Hub BERT model (or path to the BERT model in local drive).')
    parser.add_argument('-c', '--csv', dest='csv_data_file', type=str, required=True,
                        help='Path to the CSV file with labeled data.')
    parser.add_argument('-t', '--train', dest='train_file_name', type=str, required=False, default='',
                        help='Path to the text file with unlabeled data for training.')
    parser.add_argument('-e', '--test', dest='test_file_name', type=str, required=False, default='',
                        help='Path to the text file with unlabeled data for evaluation.')
    parser.add_argument('--gpu_frac', dest='gpu_memory_frac', type=float, required=False, default=0.9,
                        help='Allocable part of the GPU memory for the classifier.')
    parser.add_argument('--nn_type', dest='nn_type', type=str, choices=['bayesian', 'usual'], required=False,
                        default='bayesian', help='Neural network type: `bayesian` or `usual`.')
    parser.add_argument('--num_monte_carlo', dest='num_monte_carlo', type=int, required=False, default=100,
                        help='Number of generated Monte Carlo samples for each data sample.')
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
    parser.add_argument('--hidden', dest='hidden_layer_size', type=int, required=False, default=500,
                        help='Hidden layer size.')
    parser.add_argument('--init_kl_weight', dest='init_kl_weight', type=float, required=False, default=1e-1,
                        help='Initial value of KL weight.')
    parser.add_argument('--fin_kl_weight', dest='fin_kl_weight', type=float, required=False, default=1e-2,
                        help='Final value of KL weight.')
    parser.add_argument('--search', dest='search_hyperparameters', required=False, action='store_true',
                        default=False, help='Will be hyperparameters found by the Bayesian optimization?')
    cmd_args = parser.parse_args()

    num_monte_carlo = cmd_args.num_monte_carlo
    gpu_memory_frac = cmd_args.gpu_memory_frac
    bert_handle = cmd_args.bert
    nn_type = cmd_args.nn_type
    model_name = os.path.normpath(cmd_args.model_name)
    labeled_data_name = os.path.normpath(cmd_args.csv_data_file)
    unlabeled_train_data_name = cmd_args.train_file_name.strip()
    if len(unlabeled_train_data_name) > 0:
        unlabeled_train_data_name = os.path.normpath(unlabeled_train_data_name)
        unlabeled_texts_for_training = load_unlabeled_texts(unlabeled_train_data_name)
        assert len(unlabeled_texts_for_training) > 0, 'File `{0}` is empty!'.format(unlabeled_train_data_name)
    else:
        unlabeled_texts_for_training = None
    unlabeled_test_data_name = cmd_args.test_file_name.strip()
    if len(unlabeled_test_data_name) > 0:
        unlabeled_test_data_name = os.path.normpath(unlabeled_test_data_name)
        unlabeled_texts_for_testing = load_unlabeled_texts(unlabeled_test_data_name)
        assert len(unlabeled_texts_for_testing) > 0, 'File `{0}` is empty!'.format(unlabeled_test_data_name)
    else:
        unlabeled_texts_for_testing = None
    labeled_texts, labels, classes_list = read_csv(labeled_data_name, 7)
    print('Number of labeled texts is {0}.'.format(len(labeled_texts)))
    print('Number of classes is {0}.'.format(len(classes_list)))
    if any(map(lambda it: isinstance(it, set), labels)):
        print('Some data samples can be corresponded to several labels at once.')
        multioutput = True
    else:
        multioutput = False
    print('')
    print_classes_distribution(labels, classes_list)
    np.random.seed(42)
    indices_for_cv = ImpatialTextClassifier.cv_split(labels, 5)
    if cmd_args.search_hyperparameters:
        dimensions = [Integer(0, 300), Integer(0, 300), Integer(0, 300), Integer(0, 300), Integer(0, 300),
                      Integer(100, 2000)]
        if nn_type == 'bayesian':
            dimensions += [Real(1e-5, 1.0, prior='log-uniform'), Real(1e-5, 1.0, prior='log-uniform')]
        optimal_res = gp_minimize(
            func, dimensions=dimensions,
            n_calls=100, n_random_starts=5, random_state=42, verbose=False, n_jobs=1
        )
        print('')
        hyperparameters = optimal_res.x
    else:
        hyperparameters = [cmd_args.size_of_conv1, cmd_args.size_of_conv2, cmd_args.size_of_conv3,
                           cmd_args.size_of_conv4, cmd_args.size_of_conv5, cmd_args.hidden_layer_size,
                           cmd_args.init_kl_weight, cmd_args.fin_kl_weight]
    score(hyperparameters)
    with open(model_name, 'wb') as fp:
        pickle.dump(train(hyperparameters), fp)


if __name__ == '__main__':
    main()
