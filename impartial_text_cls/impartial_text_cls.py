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

import copy
import os
import random
import tempfile
import time
from typing import Dict, List, Tuple, Union
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_probability as tfp
from bert.tokenization import FullTokenizer
from bert.modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ImpartialTextClassifier(BaseEstimator, ClassifierMixin):
    MAX_SEQ_LENGTH = 512
    PATH_TO_BERT = None
    EPSILON = 1e-6

    def __init__(self,
                 bert_hub_module_handle: Union[str, None]='https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
                 filters_for_conv1: int=100, filters_for_conv2: int=100, filters_for_conv3: int=100,
                 filters_for_conv4: int=100, filters_for_conv5: int=100, hidden_layer_size: int=500,
                 n_hidden_layers: int=1, batch_size: int=32, validation_fraction: float=0.1, max_epochs: int=10,
                 patience: int=3, num_monte_carlo: int=50, gpu_memory_frac: float=1.0, verbose: bool=False,
                 multioutput: bool=False, bayesian: bool=True, kl_weight_init: float=1.0, kl_weight_fin: float=0.1,
                 random_seed: Union[int, None]=None):
        self.batch_size = batch_size
        self.filters_for_conv1 = filters_for_conv1
        self.filters_for_conv2 = filters_for_conv2
        self.filters_for_conv3 = filters_for_conv3
        self.filters_for_conv4 = filters_for_conv4
        self.filters_for_conv5 = filters_for_conv5
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers = n_hidden_layers
        self.bert_hub_module_handle = bert_hub_module_handle
        self.max_epochs = max_epochs
        self.num_monte_carlo = num_monte_carlo
        self.patience = patience
        self.random_seed = random_seed
        self.gpu_memory_frac = gpu_memory_frac
        self.validation_fraction = validation_fraction
        self.verbose = verbose
        self.multioutput = multioutput
        self.bayesian = bayesian
        self.kl_weight_init = kl_weight_init
        self.kl_weight_fin = kl_weight_fin

    def __del__(self):
        if hasattr(self, 'tokenizer_'):
            del self.tokenizer_
        self.finalize_model()

    def fit(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray],
            validation_data: Union[None, Tuple[Union[list, tuple, np.ndarray], Union[list, tuple, np.ndarray]]]=None):
        classes_dict, classes_reverse_list = self.check_Xy(X, 'X', y, 'y', self.multioutput)
        self.classes_ = classes_dict
        self.classes_reverse_index_ = classes_reverse_list
        if hasattr(self, 'tokenizer_'):
            del self.tokenizer_
        self.finalize_model()
        self.update_random_seed()
        if validation_data is None:
            if self.validation_fraction > 0.0:
                train_index, test_index = self.train_test_split(y, self.validation_fraction)
                X_train_ = [X[idx] for idx in train_index]
                y_train_ = self.prepare_y([y[idx] for idx in train_index])
                X_val_ = [X[idx] for idx in test_index]
                y_val_ = self.prepare_y([y[idx] for idx in test_index])
                del train_index, test_index
            else:
                X_train_ = X
                y_train_ = self.prepare_y(y)
                X_val_ = None
                y_val_ = None
        else:
            if (not isinstance(validation_data, tuple)) and (not isinstance(validation_data, list)):
                raise ValueError('')
            if len(validation_data) != 2:
                raise ValueError('')
            classes_dict_, classes_reverse_list_ = self.check_Xy(validation_data[0], 'X_val', validation_data[1],
                                                                 'y_val', self.multioutput)
            if not (set(classes_dict_.keys()) <= set(classes_dict.keys())):
                unknown_classes = sorted(list(set(classes_dict_.keys()) - set(classes_dict.keys())))
                if len(unknown_classes) == 1:
                    raise ValueError('`y_val` is wrong. Class {0} is unknown.'.format(unknown_classes[0]))
                else:
                    raise ValueError('`y_val` is wrong. Classes {0} are unknown.'.format(unknown_classes))
            X_train_ = X
            y_train_ = self.prepare_y(y)
            X_val_ = validation_data[0]
            y_val_ = self.prepare_y(validation_data[1])
        self.tokenizer_ = self.initialize_bert_tokenizer()
        X_train_tokenized, y_train_tokenized, X_unlabeled_tokenized = self.tokenize_all(X_train_, y_train_)
        if self.verbose:
            lengths_of_texts = []
            sum_of_lengths = 0
            for sample_idx in range(y_train_tokenized.shape[0]):
                lengths_of_texts.append(sum(X_train_tokenized[1][sample_idx]))
                sum_of_lengths += lengths_of_texts[-1]
            if X_unlabeled_tokenized is not None:
                for sample_idx in range(X_unlabeled_tokenized[0].shape[0]):
                    lengths_of_texts.append(sum(X_unlabeled_tokenized[1][sample_idx]))
                    sum_of_lengths += lengths_of_texts[-1]
            mean_length = sum_of_lengths / float(len(lengths_of_texts))
            lengths_of_texts.sort()
            print('')
            print('Maximal length of text (in BPE): {0}'.format(max(lengths_of_texts)))
            print('Mean length of text (in BPE): {0}'.format(mean_length))
            print('Median length of text (in BPE): {0}'.format(lengths_of_texts[len(lengths_of_texts) // 2]))
            print('')
            print('Number of known texts for training is {0}.'.format(len(y_train_tokenized)))
        X_train_tokenized, y_train_tokenized = self.extend_Xy(X_train_tokenized, y_train_tokenized, shuffle=True)
        if (X_val_ is not None) and (y_val_ is not None):
            X_val_tokenized, y_val_tokenized, X_unlabeled_tokenized_ = self.tokenize_all(X_val_, y_val_)
            if self.verbose:
                print('Number of known texts for validation is {0}.'.format(len(y_val_tokenized)))
            X_val_tokenized, y_val_tokenized = self.extend_Xy(X_val_tokenized, y_val_tokenized, shuffle=False)
            if (X_unlabeled_tokenized_ is not None) or (X_unlabeled_tokenized is not None):
                if X_unlabeled_tokenized is None:
                    X_unlabeled_tokenized = X_unlabeled_tokenized_
                elif (X_unlabeled_tokenized_ is not None) and (X_unlabeled_tokenized is not None):
                    for data_column_idx in range(len(X_train_tokenized)):
                        X_unlabeled_tokenized[data_column_idx] = np.vstack(
                            (
                                X_unlabeled_tokenized[data_column_idx],
                                X_unlabeled_tokenized_[data_column_idx]
                            )
                        )
                X_unlabeled_tokenized = self.extend_Xy(X_unlabeled_tokenized, shuffle=False)
            if X_unlabeled_tokenized_ is not None:
                del X_unlabeled_tokenized_
        else:
            X_val_tokenized = None
            y_val_tokenized = None
            if X_unlabeled_tokenized is not None:
                X_unlabeled_tokenized = self.extend_Xy(X_unlabeled_tokenized, shuffle=False)
        if self.verbose:
            if X_unlabeled_tokenized is not None:
                print('Number of unknown (foreign) texts is {0}.'.format(len(X_unlabeled_tokenized[0])))
            print('')
        n_batches = int(np.ceil(X_train_tokenized[0].shape[0] / float(self.batch_size)))
        bounds_of_batches_for_training = []
        for iteration in range(n_batches):
            batch_start = iteration * self.batch_size
            batch_end = min(batch_start + self.batch_size, X_train_tokenized[0].shape[0])
            bounds_of_batches_for_training.append((batch_start, batch_end))
        if X_val_tokenized is None:
            bounds_of_batches_for_validation = None
        else:
            n_batches = int(np.ceil(X_val_tokenized[0].shape[0] / float(self.batch_size)))
            bounds_of_batches_for_validation = []
            for iteration in range(n_batches):
                batch_start = iteration * self.batch_size
                batch_end = min(batch_start + self.batch_size, X_val_tokenized[0].shape[0])
                bounds_of_batches_for_validation.append((batch_start, batch_end))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_op, elbo_loss_, val_loss_, kl_weight_ = self.build_model(X_train_tokenized[0].shape[0])
        if not self.bayesian:
            val_loss_ = elbo_loss_
        init = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        init.run(session=self.sess_)
        tmp_model_name = self.get_temp_model_name()
        if self.verbose:
            if X_val_tokenized is None:
                if self.bayesian:
                    print('Epoch      ELBO loss   Duration (secs)')
                else:
                    print('Epoch           Loss   Duration (secs)')
        n_epochs_without_improving = 0
        try:
            best_acc = None
            for epoch in range(self.max_epochs):
                start_time = time.time()
                X_train_tokenized, y_train_tokenized = self.shuffle_train_data(X_train_tokenized, y_train_tokenized)
                feed_dict_for_batch = None
                train_loss = 0.0
                value_of_kl_weight = self.calculate_kl_weight(
                    epoch=epoch, n_epochs=self.max_epochs,
                    init_kl_weight=self.kl_weight_init, fin_kl_weight=self.kl_weight_fin
                )
                for batch_counter, cur_batch in enumerate(bounds_of_batches_for_training):
                    X_batch = [X_train_tokenized[channel_idx][cur_batch[0]:cur_batch[1]]
                               for channel_idx in range(len(X_train_tokenized))]
                    y_batch = y_train_tokenized[cur_batch[0]:cur_batch[1]]
                    if feed_dict_for_batch is not None:
                        del feed_dict_for_batch
                    if self.bayesian:
                        if abs(self.kl_weight_init - self.kl_weight_fin) > self.EPSILON:
                            feed_dict_for_batch = self.fill_feed_dict(
                                X_batch, y_batch, kl_weight_variable=kl_weight_,
                                kl_weight_value=value_of_kl_weight
                            )
                        else:
                            feed_dict_for_batch = self.fill_feed_dict(X_batch, y_batch)
                    else:
                        feed_dict_for_batch = self.fill_feed_dict(X_batch, y_batch)
                    _, train_loss_ = self.sess_.run([train_op, elbo_loss_], feed_dict=feed_dict_for_batch)
                    train_loss += train_loss_ * self.batch_size
                train_loss /= float(X_train_tokenized[0].shape[0])
                if bounds_of_batches_for_validation is not None:
                    test_loss = 0.0
                    y_pred = None
                    for cur_batch in bounds_of_batches_for_validation:
                        X_batch = [X_val_tokenized[channel_idx][cur_batch[0]:cur_batch[1]]
                                   for channel_idx in range(len(X_val_tokenized))]
                        y_batch = y_val_tokenized[cur_batch[0]:cur_batch[1]]
                        feed_dict_for_batch = self.fill_feed_dict(X_batch, y_batch)
                        test_loss_ = self.sess_.run(val_loss_, feed_dict=feed_dict_for_batch)
                        test_loss += test_loss_ * self.batch_size
                        if self.bayesian:
                            features = self._calculate_features(X_batch)
                            probs = np.asarray(
                                [self.sess_.run(
                                    'LabelsDistribution/probs:0',
                                    feed_dict={'BERT_SequenceOutput:0': features[0], 'input_mask:0': X_batch[1],
                                               'BERT_PooledOutput:0': features[1]}
                                ) for _ in range(self.num_monte_carlo)]
                            )
                            del features
                            mean_probs = np.mean(probs, axis=0)
                            del probs
                        else:
                            if self.multioutput:
                                mean_probs = self.sess_.run('Logits/Sigmoid:0', feed_dict=feed_dict_for_batch)
                            else:
                                mean_probs = self.sess_.run('Logits/Softmax:0', feed_dict=feed_dict_for_batch)
                        del feed_dict_for_batch
                        if self.multioutput:
                            if y_pred is None:
                                y_pred = mean_probs.copy()
                            else:
                                y_pred = np.vstack((y_pred, mean_probs))
                        else:
                            if y_pred is None:
                                y_pred = mean_probs.argmax(axis=-1)
                            else:
                                y_pred = np.concatenate((y_pred, mean_probs.argmax(axis=-1)))
                        del mean_probs
                    test_loss /= float(X_val_tokenized[0].shape[0])
                    if self.verbose:
                        print('Epoch {0}'.format(epoch))
                        print('  Duration is {0:.3f} seconds'.format(time.time() - start_time))
                        if self.bayesian:
                            print('  KL weight: {0:.6f}'.format(value_of_kl_weight))
                            print('  Train ELBO loss: {0:>12.6f}'.format(train_loss))
                        else:
                            print('  Train loss:      {0:>12.6f}'.format(train_loss))
                        print('  Val. loss:       {0:>12.6f}'.format(test_loss))
                    quality_by_classes = self.calculate_quality(y_val_tokenized, y_pred[0:len(y_val_tokenized)])
                    quality_test = 0.0
                    if self.multioutput:
                        for class_name in quality_by_classes.keys():
                            quality_test += quality_by_classes[class_name]
                        quality_test /= float(len(quality_by_classes))
                        if self.verbose:
                            print('  Val. ROC-AUC for all entities: {0:>6.4f}'.format(quality_test))
                            max_text_width = 0
                            for class_name in quality_by_classes.keys():
                                text_width = len(class_name if (hasattr(class_name, 'split') and
                                                                hasattr(class_name, 'strip')) else str(class_name))
                                if text_width > max_text_width:
                                    max_text_width = text_width
                            for class_name in sorted(list(quality_by_classes.keys())):
                                print('    ROC-AUC for {0:<{1}} {2:>6.4f}'.format(
                                    str(class_name) + ':', max_text_width + 1, quality_by_classes[class_name]))
                    else:
                        precision_test = 0.0
                        recall_test = 0.0
                        for class_name in quality_by_classes.keys():
                            precision_test += quality_by_classes[class_name][0]
                            recall_test += quality_by_classes[class_name][1]
                            quality_test += quality_by_classes[class_name][2]
                        precision_test /= float(len(quality_by_classes))
                        recall_test /= float(len(quality_by_classes))
                        quality_test /= float(len(quality_by_classes))
                        if self.verbose:
                            print('  Val. quality for all entities:')
                            print('    F1={0:>6.4f}, P={1:>6.4f}, R={2:>6.4f}'.format(
                                quality_test, precision_test, recall_test))
                            for class_name in sorted(list(quality_by_classes.keys())):
                                print('      Val. quality for {0}:'.format(class_name))
                                print('        F1={0:>6.4f}, P={1:>6.4f}, R={2:>6.4f}'.format(
                                    quality_by_classes[class_name][2], quality_by_classes[class_name][0],
                                    quality_by_classes[class_name][1])
                                )
                    if best_acc is None:
                        best_acc = quality_test
                        self.save_model(tmp_model_name)
                        n_epochs_without_improving = 0
                    elif quality_test > best_acc:
                        best_acc = quality_test
                        self.save_model(tmp_model_name)
                        n_epochs_without_improving = 0
                    else:
                        n_epochs_without_improving += 1
                    del y_pred, quality_by_classes
                else:
                    cur_acc = -train_loss
                    if best_acc is None:
                        best_acc = cur_acc
                        self.save_model(tmp_model_name)
                        n_epochs_without_improving = 0
                    elif cur_acc > best_acc:
                        best_acc = cur_acc
                        self.save_model(tmp_model_name)
                        n_epochs_without_improving = 0
                    else:
                        n_epochs_without_improving += 1
                    if self.verbose:
                        print('{0:>5}   {1:>12.6f}   {2:>15.3f}'.format(epoch, train_loss, time.time() - start_time))
                if n_epochs_without_improving >= self.patience:
                    if self.verbose:
                        print('Epoch %05d: early stopping' % (epoch + 1))
                    break
            if best_acc is not None:
                if hasattr(self, 'sess_'):
                    for k in list(self.sess_.graph.get_all_collection_keys()):
                        self.sess_.graph.clear_collection(k)
                    self.sess_.close()
                    del self.sess_
                tf.compat.v1.reset_default_graph()
                self.load_model(tmp_model_name)
        finally:
            for cur_name in self.find_all_model_files(tmp_model_name):
                os.remove(cur_name)
        self.calculate_certainty_treshold(X_train_tokenized, y_train_tokenized, X_val_tokenized, y_val_tokenized,
                                          X_unlabeled_tokenized)
        return self

    def _calculate_probabilities(self, X: List[np.ndarray]) -> np.ndarray:
        n_batches = int(np.ceil(X[0].shape[0] / float(self.batch_size)))
        probabilities = np.zeros((X[0].shape[0], len(self.classes_)), dtype=np.float32)
        for iteration in range(n_batches):
            batch_start = iteration * self.batch_size
            batch_end = min(batch_start + self.batch_size, X[0].shape[0])
            X_batch = [X[channel_idx][batch_start:batch_end]
                       for channel_idx in range(len(X))]
            if self.bayesian:
                features = self._calculate_features(X_batch)
                probs = np.asarray(
                    [self.sess_.run(
                        'LabelsDistribution/probs:0',
                        feed_dict={'BERT_SequenceOutput:0': features[0], 'input_mask:0': X_batch[1],
                                   'BERT_PooledOutput:0': features[1]}
                    ) for _ in range(self.num_monte_carlo)]
                )
                del features
                mean_probs = np.mean(probs, axis=0)
                probabilities[batch_start:batch_end] = mean_probs[0:(batch_end - batch_start)]
                del probs, mean_probs
            else:
                feed_dict_for_batch = self.fill_feed_dict(X_batch)
                if self.multioutput:
                    probs = self.sess_.run('Logits/Sigmoid:0', feed_dict=feed_dict_for_batch)
                else:
                    probs = self.sess_.run('Logits/Softmax:0', feed_dict=feed_dict_for_batch)
                probabilities[batch_start:batch_end] = probs[0:(batch_end - batch_start)]
                del probs
                del feed_dict_for_batch
        return probabilities

    def _calculate_features(self, X: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        feed_dict = self.fill_feed_dict(X)
        sequence_features, pooled_features = self.sess_.run(['BERT_SequenceOutput:0', 'BERT_PooledOutput:0'],
                                                            feed_dict=feed_dict)
        if len(sequence_features.shape) != 3:
            raise ValueError('Sequence features are wrong! Expected a 3-D array, but got a {0}-D one.'.format(
                len(sequence_features.shape)))
        if sequence_features.shape[1] < self.MAX_SEQ_LENGTH:
            sequence_features = np.concatenate(
                (
                    sequence_features,
                    np.zeros(
                        (
                            sequence_features.shape[0],
                            self.MAX_SEQ_LENGTH - sequence_features.shape[1],
                            sequence_features.shape[2]
                        ),
                        dtype=np.float32
                    )
                ),
                axis=1
            )
        elif sequence_features.shape[1] > self.MAX_SEQ_LENGTH:
            sequence_features = sequence_features[0:sequence_features.shape[0]][0:self.MAX_SEQ_LENGTH][
                                0:sequence_features.shape[2]]
        if len(pooled_features.shape) != 2:
            raise ValueError('Pooled features are wrong! Expected a 2-D array, but got a {0}-D one.'.format(
                len(pooled_features.shape)))
        return (sequence_features, pooled_features)

    def calculate_certainty_treshold(self, X_train: List[np.ndarray], y_train: np.ndarray,
                                     X_val: Union[List[np.ndarray], None], y_val: Union[np.ndarray, None],
                                     X_unlabeled: Union[List[np.ndarray], None]):
        if self.multioutput:
            if y_val is None:
                probabilities = self._calculate_probabilities(X_train)
                y_true = np.copy(y_train)
            else:
                probabilities = self._calculate_probabilities(X_val)
                y_true = np.copy(y_val)
            if X_unlabeled is not None:
                probabilities_for_unlabeled = self._calculate_probabilities(X_unlabeled)
                probabilities = np.vstack(
                    (
                        probabilities,
                        probabilities_for_unlabeled
                    )
                )
                y_true = np.vstack(
                    (
                        y_true,
                        np.zeros((X_unlabeled[0].shape[0], len(self.classes_)))
                    )
                )
            self.certainty_threshold_ = np.full((len(self.classes_),), 1e-3)
            for class_idx in range(len(self.classes_)):
                if any(map(lambda it: it > 0, y_true[:, class_idx])):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        best_f1 = f1_score(y_true[:, class_idx],
                                           probabilities[:, class_idx] >= self.certainty_threshold_[class_idx])
                    threshold = self.certainty_threshold_[class_idx] + 1e-3
                    while threshold < 1.0:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            f1 = f1_score(y_true[:, class_idx], probabilities[:, class_idx] >= threshold)
                        if f1 > best_f1:
                            best_f1 = f1
                            self.certainty_threshold_[class_idx] = threshold
                        threshold += 1e-3
                else:
                    self.certainty_threshold_[class_idx] = 0.5
                print('Certainty threshold for class {0} is {1:.3f}.'.format(
                    class_idx, self.certainty_threshold_[class_idx]))
        else:
            if X_val is None:
                probabilities_for_labeled_samples = self._calculate_probabilities(X_train).max(axis=-1)
            else:
                probabilities_for_labeled_samples = self._calculate_probabilities(X_val).max(axis=-1)
            if X_unlabeled is None:
                probabilities_for_another_samples = None
            else:
                probabilities_for_another_samples = self._calculate_probabilities(X_unlabeled).max(axis=-1)
            if probabilities_for_another_samples is None:
                self.certainty_threshold_ = probabilities_for_labeled_samples.min()
                if self.verbose:
                    print('Certainty threshold has been detected as minimum of maximal recognition probabilities for '
                          'labeled samples.')
                    print('This threshold is {0:.3f}.'.format(self.certainty_threshold_))
            else:
                y_true = np.concatenate(
                    (
                        np.full((len(probabilities_for_labeled_samples)), 1, dtype=np.int32),
                        np.full((len(probabilities_for_another_samples)), 0, dtype=np.int32),
                    )
                )
                probabilities = np.concatenate((probabilities_for_labeled_samples, probabilities_for_another_samples))
                best_threshold = 1e-3
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    best_f1 = f1_score(y_true, probabilities >= best_threshold)
                threshold = best_threshold + 1e-3
                while threshold < 1.0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f1 = f1_score(y_true, probabilities >= threshold)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                    threshold += 1e-3
                self.certainty_threshold_ = best_threshold
                del y_true, probabilities
                if self.verbose:
                    print(
                        'Certainty threshold has been detected as a maximization result of F1-score for the '
                        'friend-or-foe identification.')
                    print('Best F1-score is {0:.6f}.'.format(best_f1))
                    print('Corresponding threshold is {0:.3f}.'.format(self.certainty_threshold_))

    def predict_proba(self, X: Union[list, tuple, np.ndarray]) -> np.ndarray:
        self.check_params(
            bert_hub_module_handle=self.bert_hub_module_handle, batch_size=self.batch_size,
            validation_fraction=self.validation_fraction, max_epochs=self.max_epochs, patience=self.patience,
            gpu_memory_frac=self.gpu_memory_frac, verbose=self.verbose, random_seed=self.random_seed,
            num_monte_carlo=self.num_monte_carlo, filters_for_conv1=self.filters_for_conv1,
            filters_for_conv2=self.filters_for_conv2, filters_for_conv3=self.filters_for_conv3,
            filters_for_conv4=self.filters_for_conv4, filters_for_conv5=self.filters_for_conv5,
            multioutput=self.multioutput, bayesian=self.bayesian, hidden_layer_size=self.hidden_layer_size,
            n_hidden_layers=self.n_hidden_layers, kl_weight_init=self.kl_weight_init, kl_weight_fin=self.kl_weight_fin
        )
        self.check_X(X, 'X')
        self.is_fitted()
        X_tokenized = self.tokenize_all(X)
        n_samples = X_tokenized[0].shape[0]
        X_tokenized = self.extend_Xy(X_tokenized)
        probabilities = self._calculate_probabilities(X_tokenized)
        del X_tokenized
        return probabilities[0:n_samples]

    def predict_log_proba(self, X: Union[list, tuple, np.ndarray]) -> np.ndarray:
        return np.log(self.predict_proba(X) + 1e-9)

    def predict(self, X: Union[list, tuple, np.ndarray]) -> list:
        probabilities = self.predict_proba(X)
        if self.multioutput:
            recognized_classes = list()
            for sample_idx in range(probabilities.shape[0]):
                set_of_classes = set()
                for class_idx in range(probabilities.shape[1]):
                    if probabilities[sample_idx][class_idx] >= self.certainty_threshold_[class_idx]:
                        set_of_classes.add(self.classes_reverse_index_[class_idx])
                if len(set_of_classes) == 0:
                    recognized_classes.append(-1)
                elif len(set_of_classes) == 1:
                    recognized_classes.append(set_of_classes.pop())
                else:
                    recognized_classes.append(copy.copy(set_of_classes))
                del set_of_classes
        else:
            recognized_classes = []
            recognized_classes_ = probabilities.argmax(axis=-1)
            for idx in range(len(recognized_classes_)):
                if probabilities[idx][recognized_classes_[idx]] < self.certainty_threshold_:
                    recognized_classes.append(-1)
                else:
                    recognized_classes.append(self.classes_reverse_index_[recognized_classes_[idx]])
            del probabilities
        return recognized_classes

    def fit_predict(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray], **kwargs):
        return self.fit(X, y).predict(X)

    def score(self, X, y, sample_weight=None) -> float:
        self.check_params(
            bert_hub_module_handle=self.bert_hub_module_handle, batch_size=self.batch_size,
            validation_fraction=self.validation_fraction, max_epochs=self.max_epochs, patience=self.patience,
            gpu_memory_frac=self.gpu_memory_frac, verbose=self.verbose, random_seed=self.random_seed,
            num_monte_carlo=self.num_monte_carlo, filters_for_conv1=self.filters_for_conv1,
            filters_for_conv2=self.filters_for_conv2, filters_for_conv3=self.filters_for_conv3,
            filters_for_conv4=self.filters_for_conv4, filters_for_conv5=self.filters_for_conv5,
            multioutput=self.multioutput, bayesian=self.bayesian, hidden_layer_size=self.hidden_layer_size,
            n_hidden_layers=self.n_hidden_layers, kl_weight_init=self.kl_weight_init, kl_weight_fin=self.kl_weight_fin
        )
        self.is_fitted()
        classes_dict, classes_reverse_index = self.check_Xy(X, 'X', y, 'y', self.multioutput)
        if not (set(classes_dict.keys()) <= set(self.classes_.keys())):
            unknown_classes = sorted(list(set(classes_dict.keys()) - set(self.classes_.keys())))
            if len(unknown_classes) == 1:
                raise ValueError('`y` is wrong. Class {0} is unknown.'.format(unknown_classes[0]))
            else:
                raise ValueError('`y` is wrong. Classes {0} are unknown.'.format(unknown_classes))
        y_pred = self.predict(X)
        if self.multioutput:
            quality = 0.0
            n = 0
            for class_name in sorted(list(self.classes_.keys())):
                y_true_ = np.array(
                    list(map(lambda cur: (class_name in cur) if isinstance(cur, set) else (class_name == cur),
                             self.prepare_y(y))),
                    dtype=np.int32
                )
                y_pred_ = np.array(
                    list(map(lambda cur: class_name in cur if isinstance(cur, set) else class_name == cur, y_pred)),
                    dtype=np.int32
                )
                if (y_true_.max() > 0) or (y_pred_.max() > 0):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        quality += f1_score(y_true=np.array(y_true_, dtype=np.int32),
                                            y_pred=np.array(y_pred_, dtype=np.int32))
                    n += 1
            if n > 0:
                quality /= float(n)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                quality = f1_score(y_true=y, y_pred=y_pred, average='macro')
        return quality

    def update_random_seed(self):
        if self.random_seed is None:
            self.random_seed = int(round(time.time()))
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.compat.v1.random.set_random_seed(self.random_seed)

    def is_fitted(self):
        check_is_fitted(self, ['classes_', 'classes_reverse_index_', 'tokenizer_', 'sess_', 'certainty_threshold_'])

    def fill_feed_dict(self, X: List[np.ndarray], y: np.ndarray = None,
                       kl_weight_variable: tf.Variable=None, kl_weight_value: float=None) -> dict:
        assert len(X) == 3
        assert len(X[0]) == self.batch_size
        feed_dict = {
            ph: x for ph, x in zip(['input_ids:0', 'input_mask:0', 'segment_ids:0'], X)
        }
        if y is not None:
            feed_dict['y_ph:0'] = np.asarray(y, dtype=np.float32) if self.multioutput else y
        if (kl_weight_variable is not None) and (kl_weight_value is not None):
            feed_dict[kl_weight_variable] = kl_weight_value
        return feed_dict

    def extend_Xy(self, X: List[np.ndarray], y: np.ndarray=None, shuffle: bool=False) -> \
            Union[Tuple[List[np.ndarray], np.ndarray], List[np.ndarray]]:
        n_samples = X[0].shape[0]
        n_extend = n_samples % self.batch_size
        if n_extend == 0:
            if y is None:
                return X
            return X, y
        n_extend = self.batch_size - n_extend
        X_ext = [
            np.concatenate(
                (
                    X[idx],
                    np.full(
                        shape=(n_extend, self.MAX_SEQ_LENGTH),
                        fill_value=X[idx][-1],
                        dtype=X[idx].dtype
                    )
                )
            )
            for idx in range(len(X))
        ]
        if y is None:
            if shuffle:
                indices = np.arange(0, n_samples + n_extend, 1, dtype=np.int32)
                np.random.shuffle(indices)
                return [X_ext[idx][indices] for idx in range(len(X_ext))]
            return X_ext
        if self.multioutput:
            y_ext = np.concatenate(
                (
                    y,
                    np.full(shape=(n_extend, len(self.classes_)), fill_value=y[-1], dtype=y.dtype)
                )
            )
        else:
            y_ext = np.concatenate(
                (
                    y,
                    np.full(shape=(n_extend,), fill_value=y[-1], dtype=y.dtype)
                )
            )
        if shuffle:
            indices = np.arange(0, n_samples + n_extend, 1, dtype=np.int32)
            return [X_ext[idx][indices] for idx in range(len(X_ext))], y_ext[indices]
        return X_ext, y_ext

    def tokenize_all(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray] = None) -> \
            Union[Tuple[List[np.ndarray], np.ndarray, Union[List[np.ndarray], None]], List[np.ndarray]]:
        X_tokenized = [
            np.zeros((len(X), self.MAX_SEQ_LENGTH), dtype=np.int32),
            np.zeros((len(X), self.MAX_SEQ_LENGTH), dtype=np.int32),
            np.zeros((len(X), self.MAX_SEQ_LENGTH), dtype=np.int32),
        ]
        n_samples = len(X)
        for sample_idx in range(n_samples):
            source_text = X[sample_idx]
            tokenized_text = self.tokenizer_.tokenize(source_text)
            if len(tokenized_text) > (self.MAX_SEQ_LENGTH - 2):
                tokenized_text = tokenized_text[:(self.MAX_SEQ_LENGTH - 2)]
            tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
            token_IDs = self.tokenizer_.convert_tokens_to_ids(tokenized_text)
            for token_idx in range(len(token_IDs)):
                X_tokenized[0][sample_idx][token_idx] = token_IDs[token_idx]
                X_tokenized[1][sample_idx][token_idx] = 1
            del tokenized_text, token_IDs
        if y is None:
            y_tokenized = None
            X_tokenized_unlabeled = None
        else:
            indices_of_labeled_samples = []
            indices_of_unlabeled_samples = []
            for sample_idx in range(n_samples):
                if isinstance(y[sample_idx], set):
                    indices_of_labeled_samples.append(sample_idx)
                else:
                    if hasattr(y[sample_idx], 'split') and hasattr(y[sample_idx], 'strip'):
                        if len(y[sample_idx].strip()) == 0:
                            is_unlabeled = True
                        else:
                            try:
                                is_unlabeled = (int(y[sample_idx]) < 0)
                            except:
                                is_unlabeled = False
                    else:
                        try:
                            is_unlabeled = (y[sample_idx] < 0)
                        except:
                            raise ValueError('`{0}` is wrong value of class label!'.format(y[sample_idx]))
                    if is_unlabeled:
                        indices_of_unlabeled_samples.append(sample_idx)
                    else:
                        indices_of_labeled_samples.append(sample_idx)
            if len(indices_of_labeled_samples) == 0:
                raise ValueError('There are no labeled data samples!')
            if self.multioutput:
                y_tokenized = np.zeros((len(indices_of_labeled_samples), len(self.classes_)), dtype=np.int32)
                for idx, sample_idx in enumerate(indices_of_labeled_samples):
                    if isinstance(y[sample_idx], set):
                        for class_idx in map(lambda it: self.classes_[it], y[sample_idx]):
                            y_tokenized[idx, class_idx] = 1
                    else:
                        class_idx = self.classes_[y[sample_idx]]
                        y_tokenized[idx, class_idx] = 1
            else:
                y_tokenized = np.empty((len(indices_of_labeled_samples),), dtype=np.int32)
                for idx, sample_idx in enumerate(indices_of_labeled_samples):
                    y_tokenized[idx] = self.classes_[y[sample_idx]]
            if len(indices_of_unlabeled_samples) == 0:
                X_tokenized_unlabeled = None
            else:
                X_tokenized_unlabeled = [
                    np.zeros((len(indices_of_unlabeled_samples), self.MAX_SEQ_LENGTH), dtype=np.int32),
                    np.zeros((len(indices_of_unlabeled_samples), self.MAX_SEQ_LENGTH), dtype=np.int32),
                    np.zeros((len(indices_of_unlabeled_samples), self.MAX_SEQ_LENGTH), dtype=np.int32),
                ]
                for idx, sample_idx in enumerate(indices_of_unlabeled_samples):
                    for data_column_idx in range(len(X_tokenized)):
                        X_tokenized_unlabeled[data_column_idx][idx] = X_tokenized[data_column_idx][sample_idx]
                for data_column_idx in range(len(X_tokenized)):
                    X_tokenized[data_column_idx] = X_tokenized[data_column_idx][indices_of_labeled_samples]
        if y is None:
            return X_tokenized
        return X_tokenized, y_tokenized, X_tokenized_unlabeled

    def calculate_quality(self, y_true: np.ndarray, y_pred: np.ndarray) -> \
            Dict[Union[int, str], Union[float, Tuple[float, float, float]]]:
        res = dict()
        for class_idx in range(len(self.classes_reverse_index_)):
            if self.multioutput:
                y_true_ = np.asarray(y_true[:, class_idx] > 0, dtype=np.int32)
                if any(map(lambda it: it > 0, y_true_)):
                    y_pred_ = y_pred[:, class_idx]
                    if (y_true_.max() > 0) or (y_pred_.max() > 1e-9):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            res[self.classes_reverse_index_[class_idx]] = roc_auc_score(y_true_, y_pred_)
            else:
                y_true_ = np.asarray(y_true == class_idx, dtype=np.int32)
                if any(map(lambda it: it > 0, y_true_)):
                    y_pred_ = np.asarray(y_pred == class_idx, dtype=np.int32)
                    if (y_true_.max() > 0) or (y_pred_.max() > 0):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            res[self.classes_reverse_index_[class_idx]] = (
                                precision_score(y_true=y_true_, y_pred=y_pred_),
                                recall_score(y_true=y_true_, y_pred=y_pred_),
                                f1_score(y_true=y_true_, y_pred=y_pred_)
                            )
        return res

    def get_params(self, deep=True) -> dict:
        return {'bert_hub_module_handle': self.bert_hub_module_handle, 'batch_size': self.batch_size,
                'max_epochs': self.max_epochs, 'patience': self.patience, 'filters_for_conv1': self.filters_for_conv1,
                'filters_for_conv2': self.filters_for_conv2, 'filters_for_conv3': self.filters_for_conv3,
                'filters_for_conv4': self.filters_for_conv4, 'filters_for_conv5': self.filters_for_conv5,
                'validation_fraction': self.validation_fraction, 'gpu_memory_frac': self.gpu_memory_frac,
                'verbose': self.verbose, 'random_seed': self.random_seed, 'num_monte_carlo': self.num_monte_carlo,
                'multioutput': self.multioutput, 'bayesian': self.bayesian, 'hidden_layer_size': self.hidden_layer_size,
                'n_hidden_layers': self.n_hidden_layers, 'kl_weight_fin': self.kl_weight_fin,
                'kl_weight_init': self.kl_weight_init}

    def set_params(self, **params):
        for parameter, value in params.items():
            self.__setattr__(parameter, value)
        return self

    def build_model(self, n_train_samples: int):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_frac
        self.sess_ = tf.compat.v1.Session(config=config)
        input_ids = tf.compat.v1.placeholder(shape=(self.batch_size, self.MAX_SEQ_LENGTH), dtype=tf.int32,
                                             name='input_ids')
        input_mask = tf.compat.v1.placeholder(shape=(self.batch_size, self.MAX_SEQ_LENGTH), dtype=tf.int32,
                                              name='input_mask')
        segment_ids = tf.compat.v1.placeholder(shape=(self.batch_size, self.MAX_SEQ_LENGTH), dtype=tf.int32,
                                               name='segment_ids')
        if self.multioutput:
            y_ph = tf.compat.v1.placeholder(shape=(self.batch_size, len(self.classes_)), dtype=tf.float32, name='y_ph')
        else:
            y_ph = tf.compat.v1.placeholder(shape=(self.batch_size,), dtype=tf.int32, name='y_ph')
        bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids
        )
        if self.bert_hub_module_handle is None:
            if self.PATH_TO_BERT is None:
                raise ValueError('Path to the BERT model is not defined!')
            path_to_bert = os.path.normpath(self.PATH_TO_BERT)
            if not self.check_path_to_bert(path_to_bert):
                raise ValueError('`path_to_bert` is wrong! There are no BERT files into the directory `{0}`.'.format(
                    self.PATH_TO_BERT))
            bert_config = BertConfig.from_json_file(os.path.join(path_to_bert, 'bert_config.json'))
            bert_model = BertModel(config=bert_config, is_training=False, input_ids=input_ids, input_mask=input_mask,
                                   token_type_ids=segment_ids, use_one_hot_embeddings=False)
            sequence_output = tf.stop_gradient(bert_model.sequence_output, name='BERT_SequenceOutput')
            pooled_output = tf.stop_gradient(bert_model.pooled_output, name='BERT_PooledOutput')
            tvars = tf.trainable_variables()
            init_checkpoint = os.path.join(self.PATH_TO_BERT, 'bert_model.ckpt')
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if self.verbose:
                print('The BERT model has been loaded from a local drive.')
        else:
            bert_module = tfhub.Module(self.bert_hub_module_handle, trainable=True, name='BERT_module')
            bert_outputs = bert_module(bert_inputs, signature='tokens', as_dict=True)
            sequence_output = tf.stop_gradient(bert_outputs['sequence_output'], name='BERT_SequenceOutput')
            pooled_output = tf.stop_gradient(bert_outputs['pooled_output'], name='BERT_PooledOutput')
            if self.verbose:
                print('The BERT model has been loaded from the TF-Hub.')
        conv_layers = []
        feature_vector_size = sequence_output.shape[-1].value
        if self.bayesian:
            input_sequence_layer = tf.keras.Input((self.MAX_SEQ_LENGTH, feature_vector_size), name='InputForConv')
            input_mask_layer = tf.keras.Input((self.MAX_SEQ_LENGTH,), name='InputMaskForConv', dtype='int32')
            input_pooled_layer = tf.keras.Input((feature_vector_size,), name='InputForDense')
            mask_layer = MaskingMultiplicationLayer(
                feature_vector_size=self.filters_for_conv1 + self.filters_for_conv2 + self.filters_for_conv3 +
                                    self.filters_for_conv4 + self.filters_for_conv5,
                name='MaskingMultiplicationLayer'
            )(input_mask_layer)
            if self.filters_for_conv1 > 0:
                conv_layer_1 = tfp.layers.Convolution1DFlipout(
                    filters=self.filters_for_conv1, kernel_size=1, name='Conv1', padding='same',
                    activation=tf.nn.elu, seed=self.random_seed
                )(input_sequence_layer)
                conv_layers.append(conv_layer_1)
            if self.filters_for_conv2 > 0:
                conv_layer_2 = tfp.layers.Convolution1DFlipout(
                    filters=self.filters_for_conv2, kernel_size=2, name='Conv2', padding='same',
                    activation=tf.nn.elu, seed=self.random_seed
                )(input_sequence_layer)
                conv_layers.append(conv_layer_2)
            if self.filters_for_conv3 > 0:
                conv_layer_3 = tfp.layers.Convolution1DFlipout(
                    filters=self.filters_for_conv3, kernel_size=3, name='Conv3', padding='same',
                    activation=tf.nn.elu, seed=self.random_seed
                )(input_sequence_layer)
                conv_layers.append(conv_layer_3)
            if self.filters_for_conv4 > 0:
                conv_layer_4 = tfp.layers.Convolution1DFlipout(
                    filters=self.filters_for_conv4, kernel_size=4, name='Conv4', padding='same',
                    activation=tf.nn.elu, seed=self.random_seed
                )(input_sequence_layer)
                conv_layers.append(conv_layer_4)
            if self.filters_for_conv5 > 0:
                conv_layer_5 = tfp.layers.Convolution1DFlipout(
                    filters=self.filters_for_conv5, kernel_size=5, name='Conv5', padding='same',
                    activation=tf.nn.elu, seed=self.random_seed
                )(input_sequence_layer)
                conv_layers.append(conv_layer_5)
            if len(conv_layers) > 1:
                pooling_layer = tf.keras.layers.Concatenate(name='Concat1')(conv_layers)
                pooling_layer = tf.keras.layers.Multiply(name='ConvZeroPaddedOutput')([pooling_layer, mask_layer])
                pooling_layer = tf.keras.layers.Masking(name='ConvMaskedOutput')(pooling_layer)
                pooling_layer = tf.keras.layers.GlobalAveragePooling1D(name='AvePooling')(pooling_layer)
                concat_layer = tf.keras.layers.Concatenate(name='Concat2')([pooling_layer, input_pooled_layer])
            else:
                pooling_layer = tf.keras.layers.Multiply(name='ConvZeroPaddedOutput')([conv_layers[0], mask_layer])
                pooling_layer = tf.keras.layers.Masking(name='ConvMaskedOutput')(pooling_layer)
                pooling_layer = tf.keras.layers.GlobalAveragePooling1D(name='AvePooling')(pooling_layer)
                concat_layer = tf.keras.layers.Concatenate(name='Concat')([pooling_layer, input_pooled_layer])
            if (self.hidden_layer_size > 0) and (self.n_hidden_layers > 0):
                if self.n_hidden_layers > 1:
                    hidden_layer = tfp.layers.DenseFlipout(self.hidden_layer_size, seed=self.random_seed,
                                                           name='HiddenLayer1', activation=tf.nn.elu)(concat_layer)
                    for layer_idx in range(1, self.n_hidden_layers):
                        hidden_layer = tfp.layers.DenseFlipout(self.hidden_layer_size, seed=self.random_seed,
                                                               name='HiddenLayer{0}'.format(layer_idx + 1),
                                                               activation=tf.nn.elu)(hidden_layer)
                else:
                    hidden_layer = tfp.layers.DenseFlipout(self.hidden_layer_size, seed=self.random_seed,
                                                           name='HiddenLayer', activation=tf.nn.elu)(concat_layer)
                output_layer = tfp.layers.DenseFlipout(len(self.classes_), seed=self.random_seed, name='OutputLayer')(
                    hidden_layer)
            else:
                output_layer = tfp.layers.DenseFlipout(len(self.classes_), seed=self.random_seed, name='OutputLayer')(
                    concat_layer)
            model = tf.keras.Model([input_sequence_layer, input_mask_layer, input_pooled_layer], output_layer,
                                   name='BayesianNetworkModel')
            logits = model([sequence_output, input_mask, pooled_output])
            if self.multioutput:
                labels_distribution = tfp.distributions.Bernoulli(logits=logits, name='LabelsDistribution')
            else:
                labels_distribution = tfp.distributions.Categorical(logits=logits, name='LabelsDistribution')
            neg_log_likelihood = -tf.reduce_mean(input_tensor=labels_distribution.log_prob(y_ph))
            kl = sum(model.losses) / float(n_train_samples)
            if abs(self.kl_weight_init - self.kl_weight_fin) > self.EPSILON:
                kl_weight = tf.Variable(self.kl_weight_init, trainable=False, name='KL_weight', dtype=tf.float32)
                elbo_loss = neg_log_likelihood + kl_weight * kl
            else:
                elbo_loss = neg_log_likelihood + self.kl_weight_init * kl
                kl_weight = None
            with tf.name_scope('train'):
                optimizer = tf.contrib.opt.AdamWOptimizer(learning_rate=3e-4, weight_decay=1e-5)
                train_op = optimizer.minimize(elbo_loss)
            return train_op, elbo_loss, neg_log_likelihood, kl_weight
        mask_layer = MaskingMultiplicationLayer(
            feature_vector_size=self.filters_for_conv1 + self.filters_for_conv2 + self.filters_for_conv3 +
                                self.filters_for_conv4 + self.filters_for_conv5,
            name='MaskingMultiplicationLayer'
        )(input_mask)
        spatial_dropout = tf.keras.layers.SpatialDropout1D(rate=0.15, seed=self.random_seed,
                                                           name='SpatialDropout')(sequence_output)
        if self.filters_for_conv1 > 0:
            conv_layer_1 = tf.keras.layers.Conv1D(
                filters=self.filters_for_conv1, kernel_size=1, name='Conv1', padding='same', activation=None,
                kernel_initializer=tf.keras.initializers.he_uniform(seed=self.random_seed)
            )(spatial_dropout)
            conv_layer_1 = tf.keras.layers.BatchNormalization(name='BatchNormConv1')(conv_layer_1)
            conv_layer_1 = tf.keras.layers.Activation(name='ActivationConv1', activation=tf.nn.elu)(conv_layer_1)
            conv_layers.append(conv_layer_1)
        if self.filters_for_conv2 > 0:
            conv_layer_2 = tf.keras.layers.Conv1D(
                filters=self.filters_for_conv2, kernel_size=2, name='Conv2', padding='same', activation=None,
                kernel_initializer=tf.keras.initializers.he_uniform(seed=self.random_seed)
            )(spatial_dropout)
            conv_layer_2 = tf.keras.layers.BatchNormalization(name='BatchNormConv2')(conv_layer_2)
            conv_layer_2 = tf.keras.layers.Activation(name='ActivationConv2', activation=tf.nn.elu)(conv_layer_2)
            conv_layers.append(conv_layer_2)
        if self.filters_for_conv3 > 0:
            conv_layer_3 = tf.keras.layers.Conv1D(
                filters=self.filters_for_conv3, kernel_size=3, name='Conv3', padding='same', activation=None,
                kernel_initializer=tf.keras.initializers.he_uniform(seed=self.random_seed)
            )(spatial_dropout)
            conv_layer_3 = tf.keras.layers.BatchNormalization(name='BatchNormConv3')(conv_layer_3)
            conv_layer_3 = tf.keras.layers.Activation(name='ActivationConv3', activation=tf.nn.elu)(conv_layer_3)
            conv_layers.append(conv_layer_3)
        if self.filters_for_conv4 > 0:
            conv_layer_4 = tf.keras.layers.Conv1D(
                filters=self.filters_for_conv4, kernel_size=4, name='Conv4', padding='same', activation=None,
                kernel_initializer=tf.keras.initializers.he_uniform(seed=self.random_seed)
            )(spatial_dropout)
            conv_layer_4 = tf.keras.layers.BatchNormalization(name='BatchNormConv4')(conv_layer_4)
            conv_layer_4 = tf.keras.layers.Activation(name='ActivationConv4', activation=tf.nn.elu)(conv_layer_4)
            conv_layers.append(conv_layer_4)
        if self.filters_for_conv5 > 0:
            conv_layer_5 = tf.keras.layers.Conv1D(
                filters=self.filters_for_conv5, kernel_size=5, name='Conv5', padding='same', activation=None,
                kernel_initializer=tf.keras.initializers.he_uniform(seed=self.random_seed)
            )(spatial_dropout)
            conv_layer_5 = tf.keras.layers.BatchNormalization(name='BatchNormConv5')(conv_layer_5)
            conv_layer_5 = tf.keras.layers.Activation(name='ActivationConv5', activation=tf.nn.elu)(conv_layer_5)
            conv_layers.append(conv_layer_5)
        if len(conv_layers) > 1:
            pooling_layer = tf.keras.layers.Concatenate(name='Concat1')(conv_layers)
            pooling_layer = tf.keras.layers.Multiply(name='ConvZeroPaddedOutput')([pooling_layer, mask_layer])
            pooling_layer = tf.keras.layers.Masking(name='ConvMaskedOutput')(pooling_layer)
            pooling_layer = tf.keras.layers.GlobalAveragePooling1D(name='AvePooling')(pooling_layer)
            concat_layer = tf.keras.layers.Concatenate(name='Concat2')([pooling_layer, pooled_output])
        else:
            pooling_layer = tf.keras.layers.Multiply(name='ConvZeroPaddedOutput')([conv_layers[0], mask_layer])
            pooling_layer = tf.keras.layers.Masking(name='ConvMaskedOutput')(pooling_layer)
            pooling_layer = tf.keras.layers.GlobalAveragePooling1D(name='AvePooling')(pooling_layer)
            concat_layer = tf.keras.layers.Concatenate(name='Concat')([pooling_layer, pooled_output])
        glorot_init = tf.keras.initializers.glorot_uniform(seed=self.random_seed)
        if (self.hidden_layer_size > 0) and (self.n_hidden_layers > 0):
            if self.n_hidden_layers > 1:
                hidden = tf.keras.layers.Dropout(rate=0.5, seed=self.random_seed, name='Dropout1')(concat_layer)
                hidden = tf.keras.layers.Dense(
                    units=self.hidden_layer_size, activation=None, name='HiddenLayer1',
                    kernel_initializer=tf.keras.initializers.he_uniform(seed=self.random_seed)
                )(hidden)
                hidden = tf.keras.layers.BatchNormalization(name='BatchNormLayer1')(hidden)
                hidden = tf.keras.layers.Activation(name='Activation1', activation=tf.nn.elu)(hidden)
                for layer_idx in range(1, self.n_hidden_layers):
                    hidden = tf.keras.layers.Dropout(rate=0.5, seed=self.random_seed,
                                                     name='Dropout{0}'.format(layer_idx + 1))(hidden)
                    hidden = tf.keras.layers.Dense(
                        units=self.hidden_layer_size, activation=None, name='HiddenLayer{0}'.format(layer_idx + 1),
                        kernel_initializer=tf.keras.initializers.he_uniform(seed=self.random_seed)
                    )(hidden)
                    hidden = tf.keras.layers.BatchNormalization(name='BatchNormLayer{0}'.format(layer_idx + 1))(hidden)
                    hidden = tf.keras.layers.Activation(name='Activation{0}'.format(layer_idx + 1),
                                                        activation=tf.nn.elu)(hidden)
            else:
                hidden = tf.keras.layers.Dropout(rate=0.5, seed=self.random_seed, name='Dropout')(concat_layer)
                hidden = tf.keras.layers.Dense(
                    units=self.hidden_layer_size, activation=None, name='HiddenLayer',
                    kernel_initializer=tf.keras.initializers.he_uniform(seed=self.random_seed)
                )(hidden)
                hidden = tf.keras.layers.BatchNormalization(name='BatchNormLayer')(hidden)
                hidden = tf.keras.layers.Activation(name='Activation', activation=tf.nn.elu)(hidden)
            output_dropout = tf.keras.layers.Dropout(rate=0.5, seed=self.random_seed, name='OutputDropout')(hidden)
            logits = tf.layers.dense(output_dropout, units=len(self.classes_), kernel_initializer=glorot_init,
                                     name='Logits', activation=(tf.nn.sigmoid if self.multioutput else tf.nn.softmax),
                                     reuse=False)
        else:
            output_dropout = tf.keras.layers.Dropout(rate=0.5, seed=self.random_seed,
                                                     name='OutputDropout')(concat_layer)
            logits = tf.layers.dense(output_dropout, units=len(self.classes_), kernel_initializer=glorot_init,
                                     name='Logits', activation=(tf.nn.sigmoid if self.multioutput else tf.nn.softmax),
                                     reuse=False)
        with tf.name_scope('loss'):
            if self.multioutput:
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_ph, logits=logits)
            else:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ph, logits=logits)
            loss = tf.reduce_mean(loss, name='loss')
        with tf.name_scope('train'):
            optimizer = tf.contrib.opt.AdamWOptimizer(learning_rate=3e-4, weight_decay=1e-5)
            train_op = optimizer.minimize(loss)
        return train_op, loss, None, None

    def finalize_model(self):
        if hasattr(self, 'sess_'):
            for k in list(self.sess_.graph.get_all_collection_keys()):
                self.sess_.graph.clear_collection(k)
            self.sess_.close()
            del self.sess_
            tf.compat.v1.reset_default_graph()

    def save_model(self, file_name: str):
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess_, file_name)

    def load_model(self, file_name: str):
        if not hasattr(self, 'sess_'):
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_frac
            self.sess_ = tf.compat.v1.Session(config=config)
        saver = tf.train.import_meta_graph(file_name + '.meta', clear_devices=True)
        saver.restore(self.sess_, file_name)

    def initialize_bert_tokenizer(self) -> FullTokenizer:
        if self.bert_hub_module_handle is None:
            if self.PATH_TO_BERT is None:
                raise ValueError('Path to the BERT model is not defined!')
            path_to_bert = os.path.normpath(self.PATH_TO_BERT)
            if not self.check_path_to_bert(path_to_bert):
                raise ValueError('`path_to_bert` is wrong! There are no BERT files into the directory `{0}`.'.format(
                    self.PATH_TO_BERT))
            if (os.path.basename(path_to_bert).find('_uncased_') >= 0) or \
                    (os.path.basename(path_to_bert).find('uncased_') >= 0):
                do_lower_case = True
            else:
                if os.path.basename(path_to_bert).find('_cased_') >= 0 or \
                        os.path.basename(path_to_bert).startswith('cased_'):
                    do_lower_case = False
                else:
                    do_lower_case = None
            if do_lower_case is None:
                raise ValueError('`{0}` is bad path to the BERT model, because a tokenization mode (lower case or no) '
                                 'cannot be detected.'.format(path_to_bert))
            tokenizer_ = FullTokenizer(vocab_file=os.path.join(path_to_bert, 'vocab.txt'), do_lower_case=do_lower_case)
        else:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_frac
            self.sess_ = tf.compat.v1.Session(config=config)
            bert_module = tfhub.Module(self.bert_hub_module_handle, trainable=True)
            tokenization_info = bert_module(signature='tokenization_info', as_dict=True)
            vocab_file, do_lower_case = self.sess_.run([tokenization_info['vocab_file'],
                                                        tokenization_info['do_lower_case']])
            tokenizer_ = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
            if hasattr(self, 'sess_'):
                for k in list(self.sess_.graph.get_all_collection_keys()):
                    self.sess_.graph.clear_collection(k)
                self.sess_.close()
                del self.sess_
            tf.compat.v1.reset_default_graph()
        return tokenizer_

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.set_params(
            bert_hub_module_handle=self.bert_hub_module_handle, filters_for_conv1=self.filters_for_conv1,
            filters_for_conv2=self.filters_for_conv2, filters_for_conv3=self.filters_for_conv3,
            filters_for_conv4=self.filters_for_conv4, filters_for_conv5=self.filters_for_conv5,
            num_monte_carlo=self.num_monte_carlo, batch_size=self.batch_size, multioutput=self.multioutput,
            validation_fraction=self.validation_fraction, max_epochs=self.max_epochs, patience=self.patience,
            gpu_memory_frac=self.gpu_memory_frac, verbose=self.verbose, random_seed=self.random_seed,
            bayesian=self.bayesian, hidden_layer_size=self.hidden_layer_size, n_hidden_layers=self.n_hidden_layers,
            kl_weight_init=self.kl_weight_init, kl_weight_fin=self.kl_weight_fin
        )
        try:
            self.is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.certainty_threshold_ = self.certainty_threshold_
            result.classes_ = copy.copy(self.classes_)
            result.classes_reverse_index_ = copy.copy(self.classes_reverse_index_)
            result.tokenizer_ = copy.copy(self.tokenizer_)
            result.sess_ = self.sess_
        return result

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.set_params(
            bert_hub_module_handle=self.bert_hub_module_handle, filters_for_conv1=self.filters_for_conv1,
            filters_for_conv2=self.filters_for_conv2, filters_for_conv3=self.filters_for_conv3,
            filters_for_conv4=self.filters_for_conv4, filters_for_conv5=self.filters_for_conv5,
            num_monte_carlo=self.num_monte_carlo, batch_size=self.batch_size, multioutput=self.multioutput,
            validation_fraction=self.validation_fraction, max_epochs=self.max_epochs, patience=self.patience,
            gpu_memory_frac=self.gpu_memory_frac, verbose=self.verbose, random_seed=self.random_seed,
            bayesian=self.bayesian, hidden_layer_size=self.hidden_layer_size, n_hidden_layers=self.n_hidden_layers,
            kl_weight_init=self.kl_weight_init, kl_weight_fin=self.kl_weight_fin
        )
        try:
            self.is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.certainty_threshold_ = self.certainty_threshold_
            result.classes_ = copy.deepcopy(self.classes_)
            result.classes_reverse_index_ = copy.copy(self.classes_reverse_index_)
            result.tokenizer_ = copy.deepcopy(self.tokenizer_)
            result.sess_ = self.sess_
        return result

    def __getstate__(self):
        return self.dump_all()

    def __setstate__(self, state: dict):
        self.load_all(state)

    def dump_all(self):
        try:
            self.is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        params = self.get_params(True)
        if is_fitted:
            params['certainty_threshold_'] = self.certainty_threshold_
            params['classes_'] = copy.deepcopy(self.classes_)
            params['classes_reverse_index_'] = copy.copy(self.classes_reverse_index_)
            params['tokenizer_'] = copy.deepcopy(self.tokenizer_)
            model_file_name = self.get_temp_model_name()
            try:
                params['model_name_'] = os.path.basename(model_file_name)
                self.save_model(model_file_name)
                for cur_name in self.find_all_model_files(model_file_name):
                    with open(cur_name, 'rb') as fp:
                        model_data = fp.read()
                    params['model.' + os.path.basename(cur_name)] = model_data
                    del model_data
            finally:
                for cur_name in self.find_all_model_files(model_file_name):
                    os.remove(cur_name)
        return params

    def load_all(self, new_params: dict):
        if not isinstance(new_params, dict):
            raise ValueError('`new_params` is wrong! Expected `{0}`, got `{1}`.'.format(type({0: 1}), type(new_params)))
        self.check_params(**new_params)
        if hasattr(self, 'tokenizer_'):
            del self.tokenizer_
        self.finalize_model()
        is_fitted = ('classes_' in new_params) and ('classes_reverse_index_' in new_params) and \
                    ('tokenizer_' in new_params) and ('model_name_' in new_params)
        model_files = list(
            filter(
                lambda it3: len(it3) > 0,
                map(
                    lambda it2: it2[len('model.'):].strip(),
                    filter(
                        lambda it1: it1.startswith('model.') and (len(it1) > len('model.')),
                        new_params.keys()
                    )
                )
            )
        )
        if is_fitted and (len(model_files) == 0):
            is_fitted = False
        if is_fitted:
            tmp_dir_name = tempfile.gettempdir()
            tmp_file_names = [os.path.join(tmp_dir_name, cur) for cur in model_files]
            for cur in tmp_file_names:
                if os.path.isfile(cur):
                    raise ValueError('File `{0}` exists, and so it cannot be used for data transmission!'.format(cur))
            self.set_params(**new_params)
            self.classes_ = copy.deepcopy(new_params['classes_'])
            self.classes_reverse_index_ = copy.copy(new_params['classes_reverse_index_'])
            self.certainty_threshold_ = new_params['certainty_threshold_']
            self.tokenizer_ = copy.deepcopy(new_params['tokenizer_'])
            self.update_random_seed()
            try:
                for idx in range(len(model_files)):
                    with open(tmp_file_names[idx], 'wb') as fp:
                        fp.write(new_params['model.' + model_files[idx]])
                self.load_model(os.path.join(tmp_dir_name, new_params['model_name_']))
            finally:
                for cur in tmp_file_names:
                    if os.path.isfile(cur):
                        os.remove(cur)
        else:
            self.set_params(**new_params)
        return self

    @staticmethod
    def get_temp_model_name() -> str:
        return tempfile.NamedTemporaryFile(mode='w', suffix='bert_cls.ckpt').name

    @staticmethod
    def find_all_model_files(model_name: str) -> List[str]:
        model_files = []
        if os.path.isfile(model_name):
            model_files.append(model_name)
        dir_name = os.path.dirname(model_name)
        base_name = os.path.basename(model_name)
        for cur in filter(lambda it: it.lower().find(base_name.lower()) >= 0, os.listdir(dir_name)):
            model_files.append(os.path.join(dir_name, cur))
        return sorted(model_files)

    @staticmethod
    def check_params(**kwargs):
        if 'batch_size' not in kwargs:
            raise ValueError('`batch_size` is not specified!')
        if (not isinstance(kwargs['batch_size'], int)) and (not isinstance(kwargs['batch_size'], np.int32)) and \
                (not isinstance(kwargs['batch_size'], np.uint32)):
            raise ValueError('`batch_size` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['batch_size'])))
        if kwargs['batch_size'] < 1:
            raise ValueError('`batch_size` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['batch_size']))
        if 'bert_hub_module_handle' not in kwargs:
            raise ValueError('`bert_hub_module_handle` is not specified!')
        if kwargs['bert_hub_module_handle'] is not None:
            if not isinstance(kwargs['bert_hub_module_handle'], str):
                raise ValueError('`bert_hub_module_handle` is wrong! Expected `{0}`, got `{1}`.'.format(
                    type('abc'), type(kwargs['bert_hub_module_handle'])))
            if len(kwargs['bert_hub_module_handle']) < 1:
                raise ValueError('`bert_hub_module_handle` is wrong! Expected a nonepty string.')
        if 'max_epochs' not in kwargs:
            raise ValueError('`max_epochs` is not specified!')
        if (not isinstance(kwargs['max_epochs'], int)) and (not isinstance(kwargs['max_epochs'], np.int32)) and \
                (not isinstance(kwargs['max_epochs'], np.uint32)):
            raise ValueError('`max_epochs` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['max_epochs'])))
        if kwargs['max_epochs'] < 1:
            raise ValueError('`max_epochs` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['max_epochs']))
        if 'num_monte_carlo' not in kwargs:
            raise ValueError('`num_monte_carlo` is not specified!')
        if (not isinstance(kwargs['num_monte_carlo'], int)) and (not isinstance(kwargs['max_epochs'], np.int32)) and \
                (not isinstance(kwargs['num_monte_carlo'], np.uint32)):
            raise ValueError('`num_monte_carlo` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['num_monte_carlo'])))
        if kwargs['num_monte_carlo'] < 1:
            raise ValueError('`num_monte_carlo` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['num_monte_carlo']))
        if 'patience' not in kwargs:
            raise ValueError('`patience` is not specified!')
        if (not isinstance(kwargs['patience'], int)) and (not isinstance(kwargs['patience'], np.int32)) and \
                (not isinstance(kwargs['patience'], np.uint32)):
            raise ValueError('`patience` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['patience'])))
        if kwargs['patience'] < 1:
            raise ValueError('`patience` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['patience']))
        if 'hidden_layer_size' not in kwargs:
            raise ValueError('`hidden_layer_size` is not specified!')
        if (not isinstance(kwargs['hidden_layer_size'], int)) and \
                (not isinstance(kwargs['hidden_layer_size'], np.int32)) and \
                (not isinstance(kwargs['hidden_layer_size'], np.uint32)):
            raise ValueError('`hidden_layer_size` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['hidden_layer_size'])))
        if kwargs['hidden_layer_size'] < 0:
            raise ValueError('`hidden_layer_size` is wrong! Expected a positive integer value or zero, '
                             'but {0} is negative.'.format(kwargs['hidden_layer_size']))
        if 'n_hidden_layers' not in kwargs:
            raise ValueError('`n_hidden_layers` is not specified!')
        if (not isinstance(kwargs['n_hidden_layers'], int)) and \
                (not isinstance(kwargs['n_hidden_layers'], np.int32)) and \
                (not isinstance(kwargs['n_hidden_layers'], np.uint32)):
            raise ValueError('`n_hidden_layers` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['n_hidden_layers'])))
        if kwargs['n_hidden_layers'] < 0:
            raise ValueError('`n_hidden_layers` is wrong! Expected a positive integer value or zero, '
                             'but {0} is negative.'.format(kwargs['n_hidden_layers']))
        if 'random_seed' not in kwargs:
            raise ValueError('`random_seed` is not specified!')
        if kwargs['random_seed'] is not None:
            if (not isinstance(kwargs['random_seed'], int)) and (not isinstance(kwargs['random_seed'], np.int32)) and \
                    (not isinstance(kwargs['random_seed'], np.uint32)):
                raise ValueError('`random_seed` is wrong! Expected `{0}`, got `{1}`.'.format(
                    type(3), type(kwargs['random_seed'])))
        if 'gpu_memory_frac' not in kwargs:
            raise ValueError('`gpu_memory_frac` is not specified!')
        if (not isinstance(kwargs['gpu_memory_frac'], float)) and \
                (not isinstance(kwargs['gpu_memory_frac'], np.float32)) and \
                (not isinstance(kwargs['gpu_memory_frac'], np.float64)) and \
                (not isinstance(kwargs['gpu_memory_frac'], np.float)):
            raise ValueError('`gpu_memory_frac` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3.5), type(kwargs['gpu_memory_frac'])))
        if (kwargs['gpu_memory_frac'] <= 0.0) or (kwargs['gpu_memory_frac'] > 1.0):
            raise ValueError('`gpu_memory_frac` is wrong! Expected a floating-point value in the (0.0, 1.0], '
                             'but {0} is not proper.'.format(kwargs['gpu_memory_frac']))
        if 'validation_fraction' not in kwargs:
            raise ValueError('`validation_fraction` is not specified!')
        if (not isinstance(kwargs['validation_fraction'], float)) and \
                (not isinstance(kwargs['validation_fraction'], np.float32)) and \
                (not isinstance(kwargs['validation_fraction'], np.float64)) and \
                (not isinstance(kwargs['validation_fraction'], np.float)):
            raise ValueError('`validation_fraction` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3.5), type(kwargs['validation_fraction'])))
        if kwargs['validation_fraction'] < 0.0:
            raise ValueError('`validation_fraction` is wrong! Expected a positive floating-point value greater than '
                             'or equal to 0.0, but {0} is not positive.'.format(kwargs['validation_fraction']))
        if kwargs['validation_fraction'] >= 1.0:
            raise ValueError('`validation_fraction` is wrong! Expected a positive floating-point value less than 1.0, '
                             'but {0} is not less than 1.0.'.format(kwargs['validation_fraction']))
        if 'verbose' not in kwargs:
            raise ValueError('`verbose` is not specified!')
        if (not isinstance(kwargs['verbose'], int)) and (not isinstance(kwargs['verbose'], np.int32)) and \
                (not isinstance(kwargs['verbose'], np.uint32)) and \
                (not isinstance(kwargs['verbose'], bool)) and (not isinstance(kwargs['verbose'], np.bool)):
            raise ValueError('`verbose` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(True), type(kwargs['verbose'])))
        if 'multioutput' not in kwargs:
            raise ValueError('`multioutput` is not specified!')
        if (not isinstance(kwargs['multioutput'], int)) and (not isinstance(kwargs['multioutput'], np.int32)) and \
                (not isinstance(kwargs['multioutput'], np.uint32)) and \
                (not isinstance(kwargs['multioutput'], bool)) and (not isinstance(kwargs['multioutput'], np.bool)):
            raise ValueError('`multioutput` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(True), type(kwargs['multioutput'])))
        if 'bayesian' not in kwargs:
            raise ValueError('`bayesian` is not specified!')
        if (not isinstance(kwargs['bayesian'], int)) and (not isinstance(kwargs['bayesian'], np.int32)) and \
                (not isinstance(kwargs['bayesian'], np.uint32)) and \
                (not isinstance(kwargs['bayesian'], bool)) and (not isinstance(kwargs['bayesian'], np.bool)):
            raise ValueError('`bayesian` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(True), type(kwargs['bayesian'])))
        if 'kl_weight_init' not in kwargs:
            raise ValueError('`kl_weight_init` is not specified!')
        if (not isinstance(kwargs['kl_weight_init'], float)) and \
                (not isinstance(kwargs['kl_weight_init'], np.float32)) and \
                (not isinstance(kwargs['kl_weight_init'], np.float64)) and \
                (not isinstance(kwargs['kl_weight_init'], np.float)):
            raise ValueError('`kl_weight_init` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3.5), type(kwargs['kl_weight_init'])))
        if kwargs['kl_weight_init'] <= 0.0:
            raise ValueError('`kl_weight_init` is wrong! Expected a non-negative floating-point value, '
                             'but {0} is {1}.'.format(
                kwargs['kl_weight_init'],
                'zero' if abs(kwargs['kl_weight_init']) <= ImpartialTextClassifier.EPSILON else 'negative'
            ))
        if 'kl_weight_fin' not in kwargs:
            raise ValueError('`kl_weight_fin` is not specified!')
        if (not isinstance(kwargs['kl_weight_fin'], float)) and \
                (not isinstance(kwargs['kl_weight_fin'], np.float32)) and \
                (not isinstance(kwargs['kl_weight_fin'], np.float64)) and \
                (not isinstance(kwargs['kl_weight_fin'], np.float)):
            raise ValueError('`kl_weight_fin` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3.5), type(kwargs['kl_weight_fin'])))
        if kwargs['kl_weight_fin'] <= 0.0:
            raise ValueError('`kl_weight_fin` is wrong! Expected a non-negative floating-point value, '
                             'but {0} is {1}.'.format(
                kwargs['kl_weight_fin'],
                'zero' if abs(kwargs['kl_weight_fin']) <= ImpartialTextClassifier.EPSILON else 'negative'
            ))
        if 'filters_for_conv1' not in kwargs:
            raise ValueError('`filters_for_conv1` is not specified!')
        if (not isinstance(kwargs['filters_for_conv1'], int)) and \
                (not isinstance(kwargs['filters_for_conv1'], np.int32)) and \
                (not isinstance(kwargs['filters_for_conv1'], np.uint32)):
            raise ValueError('`filters_for_conv1` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['filters_for_conv1'])))
        if kwargs['filters_for_conv1'] < 0:
            raise ValueError('`filters_for_conv1` is wrong! Expected a non-negative integer value, '
                             'but {0} is not positive.'.format(kwargs['filters_for_conv1']))
        if 'filters_for_conv2' not in kwargs:
            raise ValueError('`filters_for_conv2` is not specified!')
        if (not isinstance(kwargs['filters_for_conv2'], int)) and \
                (not isinstance(kwargs['filters_for_conv2'], np.int32)) and \
                (not isinstance(kwargs['filters_for_conv2'], np.uint32)):
            raise ValueError('`filters_for_conv2` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['filters_for_conv2'])))
        if kwargs['filters_for_conv2'] < 0:
            raise ValueError('`filters_for_conv2` is wrong! Expected a non-negative integer value, '
                             'but {0} is not positive.'.format(kwargs['filters_for_conv2']))
        if 'filters_for_conv3' not in kwargs:
            raise ValueError('`filters_for_conv3` is not specified!')
        if (not isinstance(kwargs['filters_for_conv3'], int)) and \
                (not isinstance(kwargs['filters_for_conv3'], np.int32)) and \
                (not isinstance(kwargs['filters_for_conv3'], np.uint32)):
            raise ValueError('`filters_for_conv3` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['filters_for_conv3'])))
        if kwargs['filters_for_conv3'] < 0:
            raise ValueError('`filters_for_conv3` is wrong! Expected a non-negative integer value, '
                             'but {0} is not positive.'.format(kwargs['filters_for_conv3']))
        if 'filters_for_conv4' not in kwargs:
            raise ValueError('`filters_for_conv4` is not specified!')
        if (not isinstance(kwargs['filters_for_conv4'], int)) and \
                (not isinstance(kwargs['filters_for_conv4'], np.int32)) and \
                (not isinstance(kwargs['filters_for_conv4'], np.uint32)):
            raise ValueError('`filters_for_conv4` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['filters_for_conv4'])))
        if kwargs['filters_for_conv4'] < 0:
            raise ValueError('`filters_for_conv4` is wrong! Expected a non-negative integer value, '
                             'but {0} is not positive.'.format(kwargs['filters_for_conv4']))
        if 'filters_for_conv5' not in kwargs:
            raise ValueError('`filters_for_conv5` is not specified!')
        if (not isinstance(kwargs['filters_for_conv5'], int)) and \
                (not isinstance(kwargs['filters_for_conv5'], np.int32)) and \
                (not isinstance(kwargs['filters_for_conv5'], np.uint32)):
            raise ValueError('`filters_for_conv5` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['filters_for_conv5'])))
        if kwargs['filters_for_conv5'] < 0:
            raise ValueError('`filters_for_conv5` is wrong! Expected a non-negative integer value, '
                             'but {0} is not positive.'.format(kwargs['filters_for_conv5']))
        if (kwargs['filters_for_conv1'] == 0) and (kwargs['filters_for_conv2'] == 0) and \
                (kwargs['filters_for_conv3'] == 0) and (kwargs['filters_for_conv4'] == 0) and \
                (kwargs['filters_for_conv5'] == 0):
            raise ValueError('Number of convolution filters for all kernel sizes is zero!')

    @staticmethod
    def check_X(X: Union[list, tuple, np.ndarray], X_name: str):
        if (not hasattr(X, '__len__')) or (not hasattr(X, '__getitem__')):
            raise ValueError('`{0}` is wrong, because it is not a list-like object!'.format(X_name))
        if isinstance(X, np.ndarray):
            if len(X.shape) != 1:
                raise ValueError('`{0}` is wrong, because it is not 1-D list!'.format(X_name))
        n = len(X)
        for idx in range(n):
            if (not hasattr(X[idx], '__len__')) or (not hasattr(X[idx], '__getitem__')) or \
                    (not hasattr(X[idx], 'strip')) or (not hasattr(X[idx], 'split')):
                raise ValueError('Item {0} of `{1}` is wrong, because it is not string-like object!'.format(
                    idx, X_name))

    @staticmethod
    def check_Xy(X: Union[list, tuple, np.ndarray], X_name: str,
                 y: Union[list, tuple, np.ndarray], y_name: str, multioutput: bool=False) -> Tuple[dict, list]:
        ImpartialTextClassifier.check_X(X, X_name)
        if (not hasattr(y, '__len__')) or (not hasattr(y, '__getitem__')):
            raise ValueError('`{0}` is wrong, because it is not a list-like object!'.format(y_name))
        if isinstance(y, np.ndarray):
            if len(y.shape) != 1:
                raise ValueError('`{0}` is wrong, because it is not 1-D list!'.format(y_name))
        n = len(y)
        if n != len(X):
            raise ValueError('Length of `{0}` does not correspond to length of `{1}`! {2} != {3}'.format(
                X_name, y_name, len(X), len(y)))
        classed_dict = dict()
        classes_dict_reverse = list()
        for idx in range(n):
            if y[idx] is None:
                raise ValueError('Item {0} of `{1}` is wrong, because it is `None`.'.format(idx, y_name))
            if isinstance(y[idx], set) and multioutput:
                for class_value_ in y[idx]:
                    try:
                        class_value = int(class_value_)
                        if (not hasattr(class_value_, 'split')) or (not hasattr(class_value_, 'strip')):
                            if class_value != class_value_:
                                class_value = None
                    except:
                        if (not hasattr(class_value_, 'split')) or (not hasattr(class_value_, 'strip')):
                            class_value = None
                        else:
                            class_value = str(class_value_).strip()
                    if class_value is None:
                        raise ValueError('Item {0} of `{1}` is wrong, because {2} is inadmissible value for class '
                                         'label.'.format(idx, y_name, class_value_))
                    if isinstance(class_value, int) and (class_value < 0):
                        raise ValueError('Item {0} of `{1}` is wrong, because set of labels cannot contains undefined '
                                         '(negative) class labels.'.format(idx, y_name))
                    elif isinstance(class_value, str) and (len(class_value) < 1):
                        raise ValueError('Item {0} of `{1}` is wrong, because set of labels cannot contains undefined '
                                         '(negative) class labels.'.format(idx, y_name))
                    if class_value not in classed_dict:
                        classed_dict[class_value] = len(classes_dict_reverse)
                        classes_dict_reverse.append(class_value)
            else:
                try:
                    class_value = int(y[idx])
                    if (not hasattr(y[idx], 'split')) or (not hasattr(y[idx], 'strip')):
                        if class_value != y[idx]:
                            class_value = None
                except:
                    if (not hasattr(y[idx], 'split')) or (not hasattr(y[idx], 'strip')):
                        class_value = None
                    else:
                        class_value = str(y[idx]).strip()
                if class_value is None:
                    raise ValueError('Item {0} of `{1}` is wrong, because {2} is inadmissible value for class '
                                     'label.'.format(idx, y_name, y[idx]))
                if isinstance(class_value, int):
                    if class_value >= 0:
                        if class_value not in classed_dict:
                            classed_dict[class_value] = len(classes_dict_reverse)
                            classes_dict_reverse.append(class_value)
                else:
                    if len(class_value) > 0:
                        if class_value not in classed_dict:
                            classed_dict[class_value] = len(classes_dict_reverse)
                            classes_dict_reverse.append(class_value)
        if len(classed_dict) < 2:
            raise ValueError('`{0}` is wrong! There are too few classes in the `{0}`.'.format(y_name))
        return classed_dict, classes_dict_reverse

    @staticmethod
    def train_test_split(y: Union[list, tuple, np.ndarray], test_part: float) -> Tuple[np.ndarray, np.ndarray]:
        y_prep = ImpartialTextClassifier.prepare_y(y)
        n = len(y_prep)
        n_test = int(round(n * test_part))
        if n_test < 1:
            raise ValueError('{0} is too small for `test_part`!'.format(test_part))
        if n_test >= n:
            raise ValueError('{0} is too large for `test_part`!'.format(test_part))
        indices = np.arange(0, n, 1, dtype=np.int32)
        np.random.shuffle(indices)
        classes_for_training = set()
        classes_for_testing = set()
        for idx in indices[:n_test]:
            if isinstance(y_prep[idx], set):
                classes_for_testing |= y_prep[idx]
            else:
                classes_for_testing.add(y_prep[idx])
        for idx in indices[n_test:]:
            if isinstance(y_prep[idx], set):
                classes_for_training |= y_prep[idx]
            else:
                classes_for_training.add(y_prep[idx])
        for restart in range(10):
            if classes_for_training == classes_for_testing:
                break
            np.random.shuffle(indices)
            classes_for_training = set()
            classes_for_testing = set()
            for idx in indices[:n_test]:
                if isinstance(y_prep[idx], set):
                    classes_for_testing |= y_prep[idx]
                else:
                    classes_for_testing.add(y_prep[idx])
            for idx in indices[n_test:]:
                if isinstance(y_prep[idx], set):
                    classes_for_training |= y_prep[idx]
                else:
                    classes_for_training.add(y_prep[idx])
        if classes_for_training != classes_for_testing:
            warnings.warn('Source data cannot be splitted by train and test parts!')
        if not (classes_for_testing <= classes_for_training):
            np.random.shuffle(indices)
            classes_for_training = set()
            classes_for_testing = set()
            for idx in indices[:n_test]:
                if isinstance(y_prep[idx], set):
                    classes_for_testing |= y_prep[idx]
                else:
                    classes_for_testing.add(y_prep[idx])
            for idx in indices[n_test:]:
                if isinstance(y_prep[idx], set):
                    classes_for_training |= y_prep[idx]
                else:
                    classes_for_training.add(y_prep[idx])
            for restart in range(10):
                if classes_for_testing <= classes_for_training:
                    break
                np.random.shuffle(indices)
                classes_for_training = set()
                classes_for_testing = set()
                for idx in indices[:n_test]:
                    if isinstance(y_prep[idx], set):
                        classes_for_testing |= y_prep[idx]
                    else:
                        classes_for_testing.add(y_prep[idx])
                for idx in indices[n_test:]:
                    if isinstance(y_prep[idx], set):
                        classes_for_training |= y_prep[idx]
                    else:
                        classes_for_training.add(y_prep[idx])
            if not (classes_for_testing <= classes_for_training):
                raise ValueError('Source data cannot be splitted by train and test parts!')
        return indices[n_test:], indices[:n_test]

    @staticmethod
    def cv_split(y: Union[list, tuple, np.ndarray], cv: int,
                 random_state: int=None) -> List[Tuple[np.ndarray, np.ndarray]]:
        if cv < 2:
            raise ValueError('{0} is too small for the CV parameter!'.format(cv))
        y_prep = ImpartialTextClassifier.prepare_y(y)
        all_classes_list = set()
        is_multioutput = False
        for cur in y_prep:
            if isinstance(cur, set):
                all_classes_list |= cur
                is_multioutput = True
            else:
                all_classes_list.add(-1 if (hasattr(cur, 'split') and hasattr(cur, 'strip') and
                                            (len(cur) == 0)) else cur)
        if is_multioutput:
            if random_state is not None:
                np.random.seed(random_state)
            n = len(y_prep)
            n_test = n // cv
            if n_test < 1:
                raise ValueError('{0} is too large for the CV parameter! Dataset size is {1}.'.format(cv, n))
            indices = np.arange(0, n, 1, dtype=np.int32)
            np.random.shuffle(indices)
            bounds = [(idx * n_test, (idx + 1) * n_test) for idx in range(cv - 1)]
            bounds.append(((cv - 1) * n_test, n))
            classes_distr = [set() for _ in range(cv)]
            for cv_idx in range(cv):
                for idx in indices[bounds[cv_idx][0]:bounds[cv_idx][1]]:
                    if isinstance(y_prep[idx], set):
                        classes_distr[cv_idx] |= y_prep[idx]
                    else:
                        classes_distr[cv_idx].add(y_prep[idx])
            for restart in range(10):
                if all(map(lambda it: it == classes_distr[0], classes_distr[1:])):
                    break
                np.random.shuffle(indices)
                del classes_distr
                classes_distr = [set() for _ in range(cv)]
                for cv_idx in range(cv):
                    for idx in indices[bounds[cv_idx][0]:bounds[cv_idx][1]]:
                        if isinstance(y_prep[idx], set):
                            classes_distr[cv_idx] |= y_prep[idx]
                        else:
                            classes_distr[cv_idx].add(y_prep[idx])
            if not all(map(lambda it: it == classes_distr[0], classes_distr[1:])):
                warnings.warn('Source data cannot be splitted by {0} parts!'.format(cv))
            cv_indices = []
            for cv_idx in range(cv):
                test_index = indices[bounds[cv_idx][0]:bounds[cv_idx][1]]
                train_index = np.array(sorted(list(set(indices.tolist()) - set(test_index.tolist()))), dtype=np.int32)
                cv_indices.append((train_index, test_index))
        else:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            X = np.random.uniform(0.0, 1.0, (len(y_prep), 3))
            cv_indices = [(train_index, test_index) for train_index, test_index in skf.split(X, y_prep)]
            del X, skf
        return cv_indices

    @staticmethod
    def prepare_y(y: Union[list, tuple, np.ndarray]) -> Union[list, tuple, np.ndarray]:
        prep_y = []
        for cur in y:
            if isinstance(cur, set):
                new_set = set()
                for subitem in cur:
                    if hasattr(subitem, 'split') and hasattr(subitem, 'strip'):
                        if len(subitem) == 0:
                            new_set.add(-1)
                        else:
                            try:
                                new_val = int(subitem)
                            except:
                                new_val = subitem
                            new_set.add(new_val)
                    else:
                        new_set.add(subitem)
                prep_y.append(new_set)
            else:
                if hasattr(cur, 'split') and hasattr(cur, 'strip'):
                    if len(cur) == 0:
                        prep_y.append(-1)
                    else:
                        try:
                            new_val = int(cur)
                        except:
                            new_val = cur
                        prep_y.append(new_val)
                else:
                    prep_y.append(cur)
        if isinstance(y, np.ndarray):
            return np.array(prep_y, dtype=object)
        if isinstance(y, tuple):
            return tuple(prep_y)
        return prep_y

    @staticmethod
    def calculate_kl_weight(epoch: int, n_epochs: int, init_kl_weight: float, fin_kl_weight: float) -> float:
        if n_epochs < 2:
            return init_kl_weight
        if abs(init_kl_weight - fin_kl_weight) <= ImpartialTextClassifier.EPSILON:
            return init_kl_weight
        a = abs(init_kl_weight - fin_kl_weight) / (float(n_epochs - 1) * float(n_epochs - 1))
        if init_kl_weight > fin_kl_weight:
            cur_kl_weight = a * float(epoch - (n_epochs - 1)) * float(epoch - (n_epochs - 1)) + fin_kl_weight
        else:
            cur_kl_weight = a * float(epoch) * float(epoch) + init_kl_weight
        return cur_kl_weight

    @staticmethod
    def check_path_to_bert(dir_name: str) -> bool:
        if not os.path.isdir(dir_name):
            return False
        if not os.path.isfile(os.path.join(dir_name, 'vocab.txt')):
            return False
        if not os.path.isfile(os.path.join(dir_name, 'bert_model.ckpt.data-00000-of-00001')):
            return False
        if not os.path.isfile(os.path.join(dir_name, 'bert_model.ckpt.index')):
            return False
        if not os.path.isfile(os.path.join(dir_name, 'bert_model.ckpt.meta')):
            return False
        if not os.path.isfile(os.path.join(dir_name, 'bert_config.json')):
            return False
        return True

    @staticmethod
    def shuffle_train_data(X: List[np.ndarray], y: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        indices = np.arange(0, X[0].shape[0], 1, dtype=np.int32)
        np.random.shuffle(indices)
        return [X[channel_idx][indices] for channel_idx in range(len(X))], y[indices]


class MaskingMultiplicationLayer(tf.keras.layers.Layer):

    def __init__(self, feature_vector_size, **kwargs):
        self.feature_vector_size = feature_vector_size
        super(MaskingMultiplicationLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        new_masking_shape = tf.keras.backend.concatenate(
            [tf.keras.backend.shape(inputs), tf.keras.backend.ones(shape=(1,), dtype='int32')],
            axis=-1
        )
        mask = tf.keras.backend.reshape(tf.keras.backend.cast(inputs, 'float32'), shape=new_masking_shape)
        return tf.keras.backend.repeat_elements(mask, rep=self.feature_vector_size, axis=-1)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        return tuple(shape + [self.feature_vector_size])
