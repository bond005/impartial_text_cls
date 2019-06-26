from argparse import ArgumentParser
import codecs
import os
import pickle
import sys
from typing import List

import numpy as np
from sklearn.metrics import f1_score
from skopt import gp_minimize
from skopt.space import Integer


try:
    from impatial_text_cls.impatial_text_cls import ImpatialTextClassifier
    from impatial_text_cls.utils import read_csv
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from impatial_text_cls.impatial_text_cls import ImpatialTextClassifier
    from impatial_text_cls.utils import read_csv


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
        quality = 0.0
        print('Filters number for different convolution kernels: ({0}, {1}, {2}, {3}, {4})'.format(
            conv1_, conv2_, conv3_, conv4_, conv5_))
        for fold_idx, (train_index, test_index) in enumerate(indices_for_cv):
            cls = ImpatialTextClassifier(bert_hub_module_handle=(None if os.path.exists(os.path.normpath(bert_handle))
                                                                 else bert_handle),
                                         filters_for_conv1=conv1_, filters_for_conv2=conv2_, filters_for_conv3=conv3_,
                                         filters_for_conv4=conv4_, filters_for_conv5=conv5_, multioutput=multioutput,
                                         gpu_memory_frac=gpu_memory_frac, num_monte_carlo=num_monte_carlo,
                                         verbose=False, random_seed=42, max_epochs=100, patience=5, batch_size=8)
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
        total_quality = 0.0
        quality_by_classes = [0.0 for _ in range(len(classes_list))]
        n = [0 for _ in range(len(classes_list))]
        n_total = 0
        print('Optimal filters number for different convolution kernels: ({0}, {1}, {2}, {3}, {4})'.format(
            conv1_, conv2_, conv3_, conv4_, conv5_))
        print('')
        for train_index, test_index in indices_for_cv:
            cls = ImpatialTextClassifier(bert_hub_module_handle=(None if os.path.exists(os.path.normpath(bert_handle))
                                                                 else bert_handle),
                                         filters_for_conv1=conv1_, filters_for_conv2=conv2_, filters_for_conv3=conv3_,
                                         filters_for_conv4=conv4_, filters_for_conv5=conv5_, batch_size=8,
                                         gpu_memory_frac=gpu_memory_frac, num_monte_carlo=num_monte_carlo, verbose=True,
                                         random_seed=42, max_epochs=100, patience=5, multioutput=multioutput)
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
            y_pred = cls.predict(texts_for_final_testing)
            instant_quality_by_classes = []
            if multioutput:
                for class_idx in range(len(classes_list)):
                    y_true_ = []
                    y_pred_ = []
                    for sample_idx in range(len(labels_for_final_testing)):
                        if isinstance(labels_for_final_testing[sample_idx], set):
                            if class_idx in labels_for_final_testing[sample_idx]:
                                y_true_.append(1)
                            else:
                                y_true_.append(0)
                        else:
                            y_true_.append(1 if class_idx == labels_for_final_testing[sample_idx] else 0)
                        if isinstance(y_pred[sample_idx], set):
                            if class_idx in y_pred[sample_idx]:
                                y_pred_.append(1)
                            else:
                                y_pred_.append(0)
                        else:
                            y_pred_.append(1 if class_idx == y_pred[sample_idx] else 0)
                    if any(map(lambda it: it > 0, y_true_)) or any(map(lambda it: it > 0, y_pred_)):
                        instant_quality_by_classes.append(f1_score(y_true_, y_pred_, average='binary'))
                        quality_by_classes[class_idx] += instant_quality_by_classes[-1]
                        n[class_idx] += 1
                if len(instant_quality_by_classes) > 0:
                    total_quality += (sum(instant_quality_by_classes) / float(len(instant_quality_by_classes)))
                    n_total += 1
            else:
                for class_idx in range(len(classes_list)):
                    if any(map(lambda it: it == class_idx, labels_for_final_testing)) or \
                            any(map(lambda it: it == class_idx, y_pred)):
                        instant_quality_by_classes.append(f1_score(labels_for_final_testing == class_idx,
                                                                   y_pred == class_idx, average='binary'))
                        quality_by_classes[class_idx] += instant_quality_by_classes[-1]
                        n[class_idx] += 1
                if len(instant_quality_by_classes) > 0:
                    total_quality += (sum(instant_quality_by_classes) / float(len(instant_quality_by_classes)))
                    n_total += 1
            del cls, texts_for_final_testing, labels_for_final_testing
        if n_total == 0:
            raise ValueError('Model cannot be evaluated!')
        print('Total F1 score: {0:.6f}.'.format(total_quality / float(n_total)))
        print('F1 score by classes:')
        name_width = max([len(cur) for cur in classes_list])
        for class_idx in range(len(classes_list)):
            if n[class_idx] > 0:
                print('  {0:>{1}} {2:.6f}'.format(classes_list[class_idx], name_width, quality_by_classes[class_idx]))
        print('')

    def train(args) -> ImpatialTextClassifier:
        conv1_ = int(args[0])
        conv2_ = int(args[1])
        conv3_ = int(args[2])
        conv4_ = int(args[3])
        conv5_ = int(args[4])
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
                                     filters_for_conv4=conv4_, filters_for_conv5=conv5_, batch_size=8,
                                     gpu_memory_frac=gpu_memory_frac, num_monte_carlo=num_monte_carlo, verbose=True,
                                     random_seed=42, max_epochs=100, patience=5, multioutput=multioutput)
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
    parser.add_argument('--nn_type', dest='nn_type', type=str, choices=['bayesian', 'usual', 'additional_class'],
                        required=False, default='bayesian',
                        help='Neural network type: `bayesian`, `usual` or `additional_class` (it is same as `usual` '
                             'but unlabeled samples are modeled as additional class).')
    parser.add_argument('--num_monte_carlo', dest='num_monte_carlo', type=int, required=False, default=100,
                        help='Number of generated Monte Carlo samples for each data sample.')
    cmd_args = parser.parse_args()

    num_monte_carlo = cmd_args.num_monte_carlo
    gpu_memory_frac = cmd_args.gpu_memory_frac
    bert_handle = cmd_args.bert
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
    optimal_res = gp_minimize(
        func,
        dimensions=[Integer(0, 200), Integer(0, 200), Integer(0, 200), Integer(0, 200), Integer(0, 200)],
        n_calls=20, n_random_starts=5, random_state=42, verbose=False, n_jobs=1
    )
    print('')
    score(optimal_res.x)
    with open(model_name, 'rb') as fp:
        pickle.dump(train(optimal_res.x), fp)


if __name__ == '__main__':
    main()
