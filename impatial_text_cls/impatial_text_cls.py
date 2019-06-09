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
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_probability as tfp
from bert.tokenization import FullTokenizer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ImpatialTextClassifier(BaseEstimator, ClassifierMixin):
    MAX_SEQ_LENGTH = 512

    def __init__(self, hidden_layer_sizes: Union[tuple, List[int]]=(100,),
                 bert_hub_module_handle: Union[str, None]='https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
                 batch_size: int=32, validation_fraction: float=0.1, max_epochs: int=10, patience: int=3,
                 num_monte_carlo: int=50, gpu_memory_frac: float=1.0, verbose: bool=False, multioutput: bool=False,
                 random_seed: Union[int, None]=None):
        self.batch_size = batch_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.bert_hub_module_handle = bert_hub_module_handle
        self.max_epochs = max_epochs
        self.num_monte_carlo = num_monte_carlo
        self.patience = patience
        self.random_seed = random_seed
        self.gpu_memory_frac = gpu_memory_frac
        self.validation_fraction = validation_fraction
        self.verbose = verbose
        self.multioutput = multioutput

    def __del__(self):
        if hasattr(self, 'tokenizer_'):
            del self.tokenizer_
        self.finalize_model()

    def fit(self, X: Union[list, tuple, np.array], y: Union[list, tuple, np.array],
            validation_data: Union[None, Tuple[Union[list, tuple, np.array], Union[list, tuple, np.array]]]=None):
        classes_in_dataset = self.check_Xy(X, 'X', y, 'y', self.multioutput)
        if (classes_in_dataset[0] != 0) or (classes_in_dataset[-1] != (len(classes_in_dataset) - 1)):
            raise ValueError('`y` is wrong! Labels of classes are not ordered. '
                             'Expected a `{0}`, but got a `{1}`.'.format(
                list(range(len(classes_in_dataset))), classes_in_dataset))
        self.n_classes_ = len(classes_in_dataset)
        if hasattr(self, 'tokenizer_'):
            del self.tokenizer_
        self.finalize_model()
        self.update_random_seed()
        if validation_data is None:
            if self.validation_fraction > 0.0:
                train_index, test_index = self.train_test_split(y, self.validation_fraction)
                X_train_ = [X[idx] for idx in train_index]
                y_train_ = [y[idx] for idx in train_index]
                X_val_ = [X[idx] for idx in test_index]
                y_val_ = [y[idx] for idx in test_index]
                del train_index, test_index
            else:
                X_train_ = X
                y_train_ = y
                X_val_ = None
                y_val_ = None
        else:
            if (not isinstance(validation_data, tuple)) and (not isinstance(validation_data, list)):
                raise ValueError('')
            if len(validation_data) != 2:
                raise ValueError('')
            classes_for_validation = self.check_Xy(validation_data[0], 'X_val', validation_data[1], 'y_val',
                                                   self.multioutput)
            if not (set(classes_for_validation) <= set(range(self.n_classes_))):
                unknown_classes = sorted(list(set(classes_for_validation) - set(range(self.n_classes_))))
                if len(unknown_classes) == 1:
                    raise ValueError('`y_val` is wrong. Class {0} is unknown.'.format(unknown_classes[0]))
                else:
                    raise ValueError('`y_val` is wrong. Classes {0} are unknown.'.format(unknown_classes))
            X_train_ = X
            y_train_ = y
            X_val_ = validation_data[0]
            y_val_ = validation_data[1]
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
        X_train_tokenized, y_train_tokenized = self.extend_Xy(X_train_tokenized, y_train_tokenized, shuffle=True)
        if (X_val_ is not None) and (y_val_ is not None):
            X_val_tokenized, y_val_tokenized, X_unlabeled_tokenized_ = self.tokenize_all(X_val_, y_val_)
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_op, loss_ = self.build_model()
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
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        init.run(session=self.sess_)
        tmp_model_name = self.get_temp_model_name()
        if self.verbose:
            if X_val_tokenized is None:
                print('Epoch   ELBO loss   Duration (secs)')
        n_epochs_without_improving = 0
        try:
            best_acc = None
            for epoch in range(self.max_epochs):
                start_time = time.time()
                random.shuffle(bounds_of_batches_for_training)
                feed_dict_for_batch = None
                train_loss = 0.0
                for cur_batch in bounds_of_batches_for_training:
                    X_batch = [X_train_tokenized[channel_idx][cur_batch[0]:cur_batch[1]]
                               for channel_idx in range(len(X_train_tokenized))]
                    y_batch = y_train_tokenized[cur_batch[0]:cur_batch[1]]
                    if feed_dict_for_batch is not None:
                        del feed_dict_for_batch
                    feed_dict_for_batch = self.fill_feed_dict(X_batch, y_batch)
                    _, train_loss_ = self.sess_.run([train_op, loss_], feed_dict=feed_dict_for_batch)
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
                        test_loss_ = self.sess_.run(loss_, feed_dict=feed_dict_for_batch)
                        test_loss += test_loss_ * self.batch_size
                        probs = np.asarray([self.sess_.run(self.labels_distribution_.probs,
                                                           feed_dict=feed_dict_for_batch)
                                            for _ in range(self.num_monte_carlo)])
                        del feed_dict_for_batch
                        mean_probs = np.mean(probs, axis=0)
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
                        del probs, mean_probs
                    test_loss /= float(X_val_tokenized[0].shape[0])
                    if self.verbose:
                        print('Epoch {0}'.format(epoch))
                        print('  Duration is {0:.3f} seconds'.format(time.time() - start_time))
                        print('  Train ELBO loss: {0:>9.6f}'.format(train_loss))
                        print('  Val. ELBO loss:  {0:>9.6f}'.format(test_loss))
                    quality_by_classes = self.calculate_quality(y_val_tokenized, y_pred[0:len(y_val_tokenized)])
                    quality_test = 0.0
                    if self.multioutput:
                        for class_idx in quality_by_classes.keys():
                            quality_test += quality_by_classes[class_idx]
                        quality_test /= float(len(quality_by_classes))
                        if self.verbose:
                            print('  Val. ROC-AUC for all entities: {0:>6.4f}'.format(quality_test))
                            max_text_width = 0
                            for class_idx in quality_by_classes.keys():
                                text_width = len(str(class_idx))
                                if text_width > max_text_width:
                                    max_text_width = text_width
                            for class_idx in sorted(list(quality_by_classes.keys())):
                                print('    ROC-AUC for {0:>{1}}: {2:>6.4f}'.format(class_idx, max_text_width,
                                                                                   quality_by_classes[class_idx]))
                    else:
                        precision_test = 0.0
                        recall_test = 0.0
                        for class_idx in quality_by_classes.keys():
                            precision_test += quality_by_classes[class_idx][0]
                            recall_test += quality_by_classes[class_idx][1]
                            quality_test += quality_by_classes[class_idx][2]
                        precision_test /= float(len(quality_by_classes))
                        recall_test /= float(len(quality_by_classes))
                        quality_test /= float(len(quality_by_classes))
                        if self.verbose:
                            print('  Val. quality for all entities:')
                            print('    F1={0:>6.4f}, P={1:>6.4f}, R={2:>6.4f}'.format(
                                quality_test, precision_test, recall_test))
                            max_text_width = 0
                            for class_idx in quality_by_classes.keys():
                                text_width = len(str(class_idx))
                                if text_width > max_text_width:
                                    max_text_width = text_width
                            for class_idx in sorted(list(quality_by_classes.keys())):
                                print('      Val. quality for {0:>{1}}:'.format(class_idx, max_text_width))
                                print('        F1={0:>6.4f}, P={1:>6.4f}, R={2:>6.4f}'.format(
                                    quality_by_classes[class_idx][2], quality_by_classes[class_idx][0],
                                    quality_by_classes[class_idx][1])
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
                        print('{0:>5}   {1:>9.6f}   {2:>15.3f}'.format(epoch, train_loss, time.time() - start_time))
                if n_epochs_without_improving >= self.patience:
                    if self.verbose:
                        print('Epoch %05d: early stopping' % (epoch + 1))
                    break
            if best_acc is not None:
                self.finalize_model()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, _ = self.build_model()
                self.load_model(tmp_model_name)
        finally:
            for cur_name in self.find_all_model_files(tmp_model_name):
                os.remove(cur_name)
        self.calculate_certainty_treshold(X_train_tokenized, y_train_tokenized, X_val_tokenized, y_val_tokenized,
                                          X_unlabeled_tokenized)
        return self

    def calculate_certainty_treshold(self, X_train: List[np.ndarray], y_train: np.ndarray,
                                     X_val: Union[List[np.ndarray], None], y_val: Union[np.ndarray, None],
                                     X_unlabeled: Union[List[np.ndarray], None]):
        if self.multioutput:
            if y_val is None:
                n_batches = int(np.ceil(X_train[0].shape[0] / float(self.batch_size)))
                probabilities = np.zeros((X_train[0].shape[0], self.n_classes_), dtype=np.float32)
                for iteration in range(n_batches):
                    batch_start = iteration * self.batch_size
                    batch_end = min(batch_start + self.batch_size, X_train[0].shape[0])
                    X_batch = [X_train[channel_idx][batch_start:batch_end]
                               for channel_idx in range(len(X_train))]
                    feed_dict_for_batch = self.fill_feed_dict(X_batch)
                    probs = np.asarray([self.sess_.run(self.labels_distribution_.probs, feed_dict=feed_dict_for_batch)
                                        for _ in range(self.num_monte_carlo)])
                    del feed_dict_for_batch
                    mean_probs = np.mean(probs, axis=0)
                    probabilities[batch_start:batch_end] = mean_probs[0:(batch_end - batch_start)]
                    del probs, mean_probs
                y_true = np.copy(y_train)
            else:
                probabilities = np.zeros((X_val[0].shape[0], self.n_classes_), dtype=np.float32)
                n_batches = int(np.ceil(X_val[0].shape[0] / float(self.batch_size)))
                for iteration in range(n_batches):
                    batch_start = iteration * self.batch_size
                    batch_end = min(batch_start + self.batch_size, X_val[0].shape[0])
                    X_batch = [X_val[channel_idx][batch_start:batch_end]
                               for channel_idx in range(len(X_val))]
                    feed_dict_for_batch = self.fill_feed_dict(X_batch)
                    probs = np.asarray([self.sess_.run(self.labels_distribution_.probs, feed_dict=feed_dict_for_batch)
                                        for _ in range(self.num_monte_carlo)])
                    del feed_dict_for_batch
                    mean_probs = np.mean(probs, axis=0)
                    probabilities[batch_start:batch_end] = mean_probs[0:(batch_end - batch_start)]
                    del probs, mean_probs
                y_true = np.copy(y_val)
            if X_unlabeled is not None:
                n_batches = int(np.ceil(X_unlabeled[0].shape[0] / float(self.batch_size)))
                probabilities_for_unlabeled = np.empty((X_unlabeled[0].shape[0], self.n_classes_))
                for iteration in range(n_batches):
                    batch_start = iteration * self.batch_size
                    batch_end = min(batch_start + self.batch_size, X_unlabeled[0].shape[0])
                    X_batch = [X_unlabeled[channel_idx][batch_start:batch_end]
                               for channel_idx in range(len(X_unlabeled))]
                    feed_dict_for_batch = self.fill_feed_dict(X_batch)
                    probs = np.asarray([self.sess_.run(self.labels_distribution_.probs, feed_dict=feed_dict_for_batch)
                                        for _ in range(self.num_monte_carlo)])
                    mean_probs = np.mean(probs, axis=0)
                    probabilities_for_unlabeled[batch_start:batch_end] = mean_probs[0:(batch_end - batch_start)]
                probabilities = np.vstack(
                    (
                        probabilities,
                        probabilities_for_unlabeled
                    )
                )
                y_true = np.vstack(
                    (
                        y_true,
                        np.zeros((X_unlabeled[0].shape[0], self.n_classes_))
                    )
                )
            self.certainty_threshold_ = np.full((self.n_classes_,), 1e-3)
            for class_idx in range(self.n_classes_):
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
                n_batches = int(np.ceil(X_train[0].shape[0] / float(self.batch_size)))
                probabilities_for_labeled_samples = np.zeros((X_train[0].shape[0],), dtype=np.float32)
                for iteration in range(n_batches):
                    batch_start = iteration * self.batch_size
                    batch_end = min(batch_start + self.batch_size, X_train[0].shape[0])
                    X_batch = [X_train[channel_idx][batch_start:batch_end]
                               for channel_idx in range(len(X_train))]
                    feed_dict_for_batch = self.fill_feed_dict(X_batch)
                    probs = np.asarray([self.sess_.run(self.labels_distribution_.probs, feed_dict=feed_dict_for_batch)
                                        for _ in range(self.num_monte_carlo)])
                    del feed_dict_for_batch
                    mean_probs = np.mean(probs, axis=0)
                    probabilities_for_labeled_samples[batch_start:batch_end] = mean_probs.max(
                        axis=-1)[0:(batch_end - batch_start)]
                    del probs, mean_probs
            else:
                probabilities_for_labeled_samples = np.zeros((X_val[0].shape[0],), dtype=np.float32)
                n_batches = int(np.ceil(X_val[0].shape[0] / float(self.batch_size)))
                for iteration in range(n_batches):
                    batch_start = iteration * self.batch_size
                    batch_end = min(batch_start + self.batch_size, X_val[0].shape[0])
                    X_batch = [X_val[channel_idx][batch_start:batch_end]
                               for channel_idx in range(len(X_val))]
                    feed_dict_for_batch = self.fill_feed_dict(X_batch)
                    probs = np.asarray([self.sess_.run(self.labels_distribution_.probs, feed_dict=feed_dict_for_batch)
                                        for _ in range(self.num_monte_carlo)])
                    del feed_dict_for_batch
                    mean_probs = np.mean(probs, axis=0)
                    probabilities_for_labeled_samples[batch_start:batch_end] = mean_probs.max(
                        axis=-1)[0:(batch_end - batch_start)]
                    del probs, mean_probs
            if X_unlabeled is None:
                probabilities_for_another_samples = None
            else:
                probabilities_for_another_samples = np.zeros((X_unlabeled[0].shape[0]), dtype=np.float32)
                n_batches = int(np.ceil(X_unlabeled[0].shape[0] / float(self.batch_size)))
                for iteration in range(n_batches):
                    batch_start = iteration * self.batch_size
                    batch_end = min(batch_start + self.batch_size, X_unlabeled[0].shape[0])
                    X_batch = [X_unlabeled[channel_idx][batch_start:batch_end]
                               for channel_idx in range(len(X_unlabeled))]
                    feed_dict_for_batch = self.fill_feed_dict(X_batch)
                    probs = np.asarray([self.sess_.run(self.labels_distribution_.probs, feed_dict=feed_dict_for_batch)
                                        for _ in range(self.num_monte_carlo)])
                    mean_probs = np.mean(probs, axis=0)
                    probabilities_for_another_samples[batch_start:batch_end] = mean_probs.max(
                        axis=-1)[0:(batch_end - batch_start)]
                    del probs, mean_probs
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

    def predict_proba(self, X: Union[list, tuple, np.array]) -> np.ndarray:
        self.check_params(
            bert_hub_module_handle=self.bert_hub_module_handle, batch_size=self.batch_size,
            validation_fraction=self.validation_fraction, max_epochs=self.max_epochs, patience=self.patience,
            gpu_memory_frac=self.gpu_memory_frac, verbose=self.verbose, random_seed=self.random_seed,
            num_monte_carlo=self.num_monte_carlo, hidden_layer_sizes=self.hidden_layer_sizes,
            multioutput=self.multioutput
        )
        self.check_X(X, 'X')
        self.is_fitted()
        X_tokenized = self.tokenize_all(X)
        n_samples = X_tokenized[0].shape[0]
        X_tokenized = self.extend_Xy(X_tokenized)
        n_batches = X_tokenized[0].shape[0] // self.batch_size
        bounds_of_batches = []
        for iteration in range(n_batches):
            batch_start = iteration * self.batch_size
            batch_end = batch_start + self.batch_size
            bounds_of_batches.append((batch_start, batch_end))
        probabilities = np.zeros((X_tokenized[0].shape[0], self.n_classes_), dtype=np.float32)
        for cur_batch in bounds_of_batches:
            feed_dict = self.fill_feed_dict(
                [
                    X_tokenized[channel_idx][cur_batch[0]:cur_batch[1]]
                    for channel_idx in range(len(X_tokenized))
                ]
            )
            probs = np.asarray([self.sess_.run(self.labels_distribution_.probs, feed_dict=feed_dict)
                                for _ in range(self.num_monte_carlo)])
            probabilities[cur_batch[0]:cur_batch[1]] = np.mean(probs, axis=0)
            del probs
        del X_tokenized, bounds_of_batches
        return probabilities[0:n_samples]

    def predict_log_proba(self, X: Union[list, tuple, np.array]) -> np.ndarray:
        return np.log(self.predict_proba(X) + 1e-9)

    def predict(self, X: Union[list, tuple, np.array]) -> np.ndarray:
        probabilities = self.predict_proba(X)
        if self.multioutput:
            recognized_classes = list()
            for sample_idx in range(probabilities.shape[0]):
                set_of_classes = set()
                for class_idx in range(probabilities.shape[1]):
                    if probabilities[sample_idx][class_idx] >= self.certainty_threshold_[class_idx]:
                        set_of_classes.add(class_idx)
                if len(set_of_classes) == 0:
                    recognized_classes.append(-1)
                elif len(set_of_classes) == 1:
                    recognized_classes.append(set_of_classes.pop())
                else:
                    recognized_classes.append(copy.copy(set_of_classes))
                del set_of_classes
            recognized_classes = np.array(recognized_classes, dtype=object)
        else:
            recognized_classes = probabilities.argmax(axis=-1)
            for idx in range(len(recognized_classes)):
                if probabilities[idx][recognized_classes[idx]] < self.certainty_threshold_:
                    recognized_classes[idx] = -1
            del probabilities
        return recognized_classes

    def fit_predict(self, X: Union[list, tuple, np.array], y: Union[list, tuple, np.array], **kwargs):
        return self.fit(X, y).predict(X)

    def score(self, X, y, sample_weight=None) -> float:
        self.check_params(
            bert_hub_module_handle=self.bert_hub_module_handle, batch_size=self.batch_size,
            validation_fraction=self.validation_fraction, max_epochs=self.max_epochs, patience=self.patience,
            gpu_memory_frac=self.gpu_memory_frac, verbose=self.verbose, random_seed=self.random_seed,
            num_monte_carlo=self.num_monte_carlo, hidden_layer_sizes=self.hidden_layer_sizes,
            multioutput=self.multioutput
        )
        self.is_fitted()
        classes_list = self.check_Xy(X, 'X', y, 'y', self.multioutput)
        if not (set(classes_list) <= set(range(self.n_classes_))):
            unknown_classes = sorted(list(set(classes_list) - set(range(self.n_classes_))))
            if len(unknown_classes) == 1:
                raise ValueError('`y` is wrong. Class {0} is unknown.'.format(unknown_classes[0]))
            else:
                raise ValueError('`y` is wrong. Classes {0} are unknown.'.format(unknown_classes))
        y_pred = self.predict(X)
        if self.multioutput:
            quality = 0.0
            for class_idx in range(self.n_classes_):
                y_true_ = list(map(lambda cur: class_idx in cur if isinstance(cur, set) else class_idx == cur, y))
                y_pred_ = list(map(lambda cur: class_idx in cur if isinstance(cur, set) else class_idx == cur, y_pred))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    quality = f1_score(y_true=np.array(y_true_, dtype=np.int32),
                                       y_pred=np.array(y_pred_, dtype=np.int32))
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
        tf.random.set_random_seed(self.random_seed)

    def is_fitted(self):
        check_is_fitted(self, ['n_classes_', 'logits_', 'tokenizer_', 'input_ids_', 'input_mask_', 'segment_ids_',
                               'labels_distribution_', 'y_ph_', 'sess_', 'certainty_threshold_'])

    def fill_feed_dict(self, X: List[np.array], y: np.array = None) -> dict:
        assert len(X) == 3
        assert len(X[0]) == self.batch_size
        feed_dict = {
            ph: x for ph, x in zip([self.input_ids_, self.input_mask_, self.segment_ids_], X)
        }
        if y is not None:
            feed_dict[self.y_ph_] = y
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
                    np.full(shape=(n_extend, self.n_classes_), fill_value=y[-1], dtype=y.dtype)
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

    def tokenize_all(self, X: Union[list, tuple, np.array], y: Union[list, tuple, np.array] = None) -> \
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
                if isinstance(y[sample_idx], set) or (y[sample_idx] >= 0):
                    indices_of_labeled_samples.append(sample_idx)
                else:
                    indices_of_unlabeled_samples.append(sample_idx)
            if len(indices_of_labeled_samples) == 0:
                raise ValueError('There are no labeled data samples!')
            if self.multioutput:
                y_tokenized = np.zeros((len(indices_of_labeled_samples), self.n_classes_), dtype=np.int32)
                for idx, sample_idx in enumerate(indices_of_labeled_samples):
                    if isinstance(y[sample_idx], set):
                        for class_idx in y[sample_idx]:
                            y_tokenized[idx, class_idx] = 1
                    else:
                        class_idx = y[sample_idx]
                        y_tokenized[idx, class_idx] = 1
            else:
                y_tokenized = np.empty((len(indices_of_labeled_samples),), dtype=np.int32)
                for idx, sample_idx in enumerate(indices_of_labeled_samples):
                    y_tokenized[idx] = y[sample_idx]
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

    def calculate_quality(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, Union[float,
                                                                                           Tuple[float, float, float]]]:
        res = dict()
        for class_idx in range(self.n_classes_):
            if self.multioutput:
                y_true_ = np.asarray(y_true[:, class_idx] > 0, dtype=np.int32)
                if any(map(lambda it: it > 0, y_true_)):
                    y_pred_ = y_pred[:, class_idx]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res[class_idx] = roc_auc_score(y_true_, y_pred_)
            else:
                y_true_ = np.asarray(y_true == class_idx, dtype=np.int32)
                if any(map(lambda it: it > 0, y_true_)):
                    y_pred_ = np.asarray(y_pred == class_idx, dtype=np.int32)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res[class_idx] = (
                            precision_score(y_true=y_true_, y_pred=y_pred_),
                            recall_score(y_true=y_true_, y_pred=y_pred_),
                            f1_score(y_true=y_true_, y_pred=y_pred_)
                        )
        return res

    def get_params(self, deep=True) -> dict:
        return {'bert_hub_module_handle': self.bert_hub_module_handle, 'batch_size': self.batch_size,
                'max_epochs': self.max_epochs, 'patience': self.patience, 'hidden_layer_sizes': self.hidden_layer_sizes,
                'validation_fraction': self.validation_fraction, 'gpu_memory_frac': self.gpu_memory_frac,
                'verbose': self.verbose, 'random_seed': self.random_seed, 'num_monte_carlo': self.num_monte_carlo,
                'multioutput': self.multioutput}

    def set_params(self, **params):
        for parameter, value in params.items():
            self.__setattr__(parameter, value)
        return self

    def build_model(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_frac
        self.sess_ = tf.Session(config=config)
        self.input_ids_ = tf.placeholder(shape=(self.batch_size, self.MAX_SEQ_LENGTH), dtype=tf.int32,
                                         name='input_ids')
        self.input_mask_ = tf.placeholder(shape=(self.batch_size, self.MAX_SEQ_LENGTH), dtype=tf.int32,
                                          name='input_mask')
        self.segment_ids_ = tf.placeholder(shape=(self.batch_size, self.MAX_SEQ_LENGTH), dtype=tf.int32,
                                           name='segment_ids')
        if self.multioutput:
            self.y_ph_ = tf.placeholder(shape=(self.batch_size, self.n_classes_), dtype=tf.int32, name='y_ph')
        else:
            self.y_ph_ = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32, name='y_ph')
        bert_inputs = dict(
            input_ids=self.input_ids_,
            input_mask=self.input_mask_,
            segment_ids=self.segment_ids_
        )
        with tf.name_scope('BERT_base'):
            bert_module = tfhub.Module(self.bert_hub_module_handle, trainable=True)
            bert_outputs = bert_module(bert_inputs, signature='tokens', as_dict=True)
            pooled_output = tf.stop_gradient(bert_outputs['pooled_output'])
        if self.verbose:
            print('The BERT model has been loaded from the TF-Hub.')
        layers = []
        for hidden_layer_idx in range(len(self.hidden_layer_sizes)):
            layers.append(
                tfp.layers.DenseFlipout(self.hidden_layer_sizes[hidden_layer_idx], seed=self.random_seed,
                                        activation=tf.nn.relu, name='HiddenLayer{0}'.format(hidden_layer_idx + 1))
            )
        layers.append(tfp.layers.DenseFlipout(self.n_classes_, seed=self.random_seed, name='OutputLayer'))
        model = tf.keras.Sequential(layers)
        self.logits_ = model(pooled_output)
        if self.multioutput:
            self.labels_distribution_ = tfp.distributions.Bernoulli(logits=self.logits_)
        else:
            self.labels_distribution_ = tfp.distributions.Categorical(logits=self.logits_)
        neg_log_likelihood = -tf.reduce_mean(input_tensor=self.labels_distribution_.log_prob(self.y_ph_))
        kl = sum(model.losses) / float(self.batch_size)
        elbo_loss = neg_log_likelihood + kl
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(elbo_loss)
        return train_op, elbo_loss

    def finalize_model(self):
        if hasattr(self, 'input_ids_'):
            del self.input_ids_
        if hasattr(self, 'input_mask_'):
            del self.input_mask_
        if hasattr(self, 'segment_ids_'):
            del self.segment_ids_
        if hasattr(self, 'y_ph_'):
            del self.y_ph_
        if hasattr(self, 'logits_'):
            del self.logits_
        if hasattr(self, 'labels_distribution_'):
            del self.labels_distribution_
        if hasattr(self, 'sess_'):
            for k in list(self.sess_.graph.get_all_collection_keys()):
                self.sess_.graph.clear_collection(k)
            self.sess_.close()
            del self.sess_
        tf.reset_default_graph()

    def save_model(self, file_name: str):
        saver = tf.train.Saver(allow_empty=True)
        saver.save(self.sess_, file_name)

    def load_model(self, file_name: str):
        saver = tf.train.Saver(allow_empty=True)
        saver.restore(self.sess_, file_name)

    def initialize_bert_tokenizer(self) -> FullTokenizer:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_frac
        self.sess_ = tf.Session(config=config)
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
        tf.reset_default_graph()
        return tokenizer_

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.set_params(
            bert_hub_module_handle=self.bert_hub_module_handle, hidden_layer_sizes=copy.copy(self.hidden_layer_sizes),
            num_monte_carlo=self.num_monte_carlo, batch_size=self.batch_size, multioutput=self.multioutput,
            validation_fraction=self.validation_fraction, max_epochs=self.max_epochs, patience=self.patience,
            gpu_memory_frac=self.gpu_memory_frac, verbose=self.verbose, random_seed=self.random_seed
        )
        try:
            self.is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.certainty_threshold_ = copy.copy(self.certainty_threshold_)
            result.n_classes_ = self.n_classes_
            result.logits_ = self.logits_
            result.tokenizer_ = self.tokenizer_
            result.input_ids_ = self.input_ids_
            result.input_mask_ = self.input_mask_
            result.segment_ids_ = self.segment_ids_
            result.y_ph_ = self.y_ph_
            result.sess_ = self.sess_
            result.labels_distribution_ = self.labels_distribution_
        return result

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.set_params(
            bert_hub_module_handle=self.bert_hub_module_handle, hidden_layer_sizes=copy.copy(self.hidden_layer_sizes),
            num_monte_carlo=self.num_monte_carlo, batch_size=self.batch_size, multioutput=self.multioutput,
            validation_fraction=self.validation_fraction, max_epochs=self.max_epochs, patience=self.patience,
            gpu_memory_frac=self.gpu_memory_frac, verbose=self.verbose, random_seed=self.random_seed
        )
        try:
            self.is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.certainty_threshold_ = copy.copy(self.certainty_threshold_)
            result.n_classes_ = self.n_classes_
            result.logits_ = self.logits_
            result.tokenizer_ = self.tokenizer_
            result.input_ids_ = self.input_ids_
            result.input_mask_ = self.input_mask_
            result.segment_ids_ = self.segment_ids_
            result.y_ph_ = self.y_ph_
            result.sess_ = self.sess_
            result.labels_distribution_ = self.labels_distribution_
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
            params['certainty_threshold_'] = copy.copy(self.certainty_threshold_)
            params['n_classes_'] = self.n_classes_
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
        is_fitted = ('n_classes_' in new_params) and  ('tokenizer_' in new_params) and ('model_name_' in new_params)
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
            self.n_classes_ = new_params['n_classes_']
            self.certainty_threshold_ = copy.copy(new_params['certainty_threshold_'])
            self.tokenizer_ = copy.deepcopy(new_params['tokenizer_'])
            self.update_random_seed()
            try:
                for idx in range(len(model_files)):
                    with open(tmp_file_names[idx], 'wb') as fp:
                        fp.write(new_params['model.' + model_files[idx]])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, _ = self.build_model()
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
                (not isinstance(kwargs['gpu_memory_frac'], np.float64)):
            raise ValueError('`gpu_memory_frac` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3.5), type(kwargs['gpu_memory_frac'])))
        if (kwargs['gpu_memory_frac'] <= 0.0) or (kwargs['gpu_memory_frac'] > 1.0):
            raise ValueError('`gpu_memory_frac` is wrong! Expected a floating-point value in the (0.0, 1.0], '
                             'but {0} is not proper.'.format(kwargs['gpu_memory_frac']))
        if 'validation_fraction' not in kwargs:
            raise ValueError('`validation_fraction` is not specified!')
        if (not isinstance(kwargs['validation_fraction'], float)) and \
                (not isinstance(kwargs['validation_fraction'], np.float32)) and \
                (not isinstance(kwargs['validation_fraction'], np.float64)):
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
        if 'hidden_layer_sizes' not in kwargs:
            raise ValueError('`hidden_layer_sizes` is not specified!')
        if (not isinstance(kwargs['hidden_layer_sizes'], list)) and \
                (not isinstance(kwargs['hidden_layer_sizes'], tuple)) and \
                (not isinstance(kwargs['hidden_layer_sizes'], np.ndarray)):
            raise ValueError('`hidden_layer_sizes` is wrong! Expected `{0}`, got `{1}`.'.format(
                type((1, 2, 3)), type(kwargs['hidden_layer_sizes'])))
        if isinstance(kwargs['hidden_layer_sizes'], np.ndarray):
            if len(kwargs['hidden_layer_sizes'].shape) != 1:
                raise ValueError('`hidden_layer_sizes` is wrong! Expected 1d array, but got {0}d one.'.format(
                    len(kwargs['hidden_layer_sizes'].shape)))
        if len(kwargs['hidden_layer_sizes']) < 1:
            raise ValueError('`hidden_layer_sizes` is wrong! It is empty.')
        for layer_idx, layer_size in enumerate(kwargs['hidden_layer_sizes']):
            if (not isinstance(layer_size, int)) and (not isinstance(layer_size, np.int32)) and \
                    (not isinstance(layer_size, np.int64)) and (not isinstance(layer_size, np.int)) and \
                    (not isinstance(layer_size, np.int16)) and (not isinstance(layer_size, np.int8)) and \
                    (not isinstance(layer_size, np.uint32)) and (not isinstance(layer_size, np.uint64)) and \
                    (not isinstance(layer_size, np.uint16)) and (not isinstance(layer_size, np.uint8)) and \
                    (not isinstance(layer_size, np.uint)):
                raise ValueError('Item {0} of `hidden_layer_sizes` is wrong! Expected `{1}`, got `{2}`.'.format(
                    layer_idx, type(1), type(layer_size)))
            if layer_size <= 0:
                raise ValueError('Item {0} of `hidden_layer_sizes` is wrong! Expected a positive value, '
                                 'got {1}.'.format(layer_idx, layer_size))

    @staticmethod
    def check_X(X: Union[list, tuple, np.array], X_name: str):
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
    def check_Xy(X: Union[list, tuple, np.array], X_name: str,
                 y: Union[list, tuple, np.array], y_name: str, multioutput: bool=False) -> List[int]:
        ImpatialTextClassifier.check_X(X, X_name)
        if (not hasattr(y, '__len__')) or (not hasattr(y, '__getitem__')):
            raise ValueError('`{0}` is wrong, because it is not a list-like object!'.format(y_name))
        if isinstance(y, np.ndarray):
            if len(y.shape) != 1:
                raise ValueError('`{0}` is wrong, because it is not 1-D list!'.format(y_name))
        n = len(y)
        if n != len(X):
            raise ValueError('Length of `{0}` does not correspond to length of `{1}`! {2} != {3}'.format(
                X_name, y_name, len(X), len(y)))
        classes_list = set()
        for idx in range(n):
            if y[idx] is None:
                raise ValueError('Item {0} of `{1}` is wrong, because it is `None`.'.format(idx, y_name))
            if isinstance(y[idx], set) and multioutput:
                for class_idx_ in y[idx]:
                    try:
                        class_idx = int(class_idx_)
                        if class_idx != class_idx_:
                            class_idx = None
                    except:
                        class_idx = None
                    if class_idx is None:
                        raise ValueError('Item {0} of `{1}` is wrong, because `{2}` is inadmissible type for class '
                                         'label.'.format(idx, y_name, type(class_idx_)))
                    if class_idx < 0:
                        raise ValueError('Item {0} of `{1}` is wrong, because set of labels cannot contains undefined '
                                         '(negative) class labels.'.format(idx, y_name))
            else:
                try:
                    class_idx = int(y[idx])
                    if class_idx != y[idx]:
                        class_idx = None
                except:
                    class_idx = None
                if class_idx is None:
                    raise ValueError('Item {0} of `{1}` is wrong, because `{2}` is inadmissible type for class '
                                     'label.'.format(idx, y_name, type(y[idx])))
                if class_idx >= 0:
                    classes_list.add(class_idx)
        if len(classes_list) < 2:
            raise ValueError('`{0}` is wrong! There are too few classes in the `{0}`.'.format(y_name))
        return sorted(list(classes_list))

    @staticmethod
    def train_test_split(y: Union[list, tuple, np.array], test_part: float) -> Tuple[np.ndarray, np.ndarray]:
        n = len(y)
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
            if isinstance(y[idx], set):
                classes_for_testing |= y[idx]
            else:
                classes_for_testing.add(y[idx])
        for idx in indices[n_test:]:
            if isinstance(y[idx], set):
                classes_for_training |= y[idx]
            else:
                classes_for_training.add(y[idx])
        for restart in range(10):
            if classes_for_training == classes_for_testing:
                break
            np.random.shuffle(indices)
            classes_for_training = set()
            classes_for_testing = set()
            for idx in indices[:n_test]:
                if isinstance(y[idx], set):
                    classes_for_testing |= y[idx]
                else:
                    classes_for_testing.add(y[idx])
            for idx in indices[n_test:]:
                if isinstance(y[idx], set):
                    classes_for_training |= y[idx]
                else:
                    classes_for_training.add(y[idx])
        if classes_for_training != classes_for_testing:
            warnings.warn('Source data cannot be splitted by train and test parts!')
        if not (classes_for_testing <= classes_for_training):
            np.random.shuffle(indices)
            classes_for_training = set()
            classes_for_testing = set()
            for idx in indices[:n_test]:
                if isinstance(y[idx], set):
                    classes_for_testing |= y[idx]
                else:
                    classes_for_testing.add(y[idx])
            for idx in indices[n_test:]:
                if isinstance(y[idx], set):
                    classes_for_training |= y[idx]
                else:
                    classes_for_training.add(y[idx])
            for restart in range(10):
                if classes_for_testing <= classes_for_training:
                    break
                np.random.shuffle(indices)
                classes_for_training = set()
                classes_for_testing = set()
                for idx in indices[:n_test]:
                    if isinstance(y[idx], set):
                        classes_for_testing |= y[idx]
                    else:
                        classes_for_testing.add(y[idx])
                for idx in indices[n_test:]:
                    if isinstance(y[idx], set):
                        classes_for_training |= y[idx]
                    else:
                        classes_for_training.add(y[idx])
            if not (classes_for_testing <= classes_for_training):
                raise ValueError('Source data cannot be splitted by train and test parts!')
        return indices[n_test:], indices[:n_test]
