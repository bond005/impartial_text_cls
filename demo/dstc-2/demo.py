from argparse import ArgumentParser
import os
import pickle
import sys


try:
    from impatial_text_cls.impatial_text_cls import ImpatialTextClassifier
    from impatial_text_cls.utils import read_dstc2_data
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from impatial_text_cls.impatial_text_cls import ImpatialTextClassifier
    from impatial_text_cls.utils import read_dstc2_data


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The binary file with the text classifier.')
    parser.add_argument('-t', '--train', dest='train_file_name', type=str, required=True,
                        help='Path to the archive with DSTC-2 training data.')
    parser.add_argument('-e', '--test', dest='test_file_name', type=str, required=True,
                        help='Path to the archive with DSTC-2 data for final testing.')
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
    parser.add_argument('--nn_type', dest='nn_type', type=str, choices=['bayesian', 'usual'],
                        required=False, default='bayesian', help='Neural network type: `bayesian` or `usual`.')
    args = parser.parse_args()

    model_name = os.path.normpath(args.model_name)
    train_file_name = os.path.normpath(args.train_file_name)
    test_file_name = os.path.normpath(args.test_file_name)

    if os.path.isfile(model_name):
        with open(model_name, 'rb') as fp:
            nn, train_classes = pickle.load(fp)
        print('Classes list: {0}'.format(train_classes))
        print('')
    else:
        train_texts, train_labels, train_classes = read_dstc2_data(train_file_name)
        print('Classes list: {0}'.format(train_classes))
        print('Number of samples for training is {0}.'.format(len(train_texts)))
        nn = ImpatialTextClassifier(filters_for_conv1=args.size_of_conv1, filters_for_conv2=args.size_of_conv2,
                                    filters_for_conv3=args.size_of_conv3, filters_for_conv4=args.size_of_conv4,
                                    filters_for_conv5=args.size_of_conv5, batch_size=args.batch_size,
                                    num_monte_carlo=args.num_monte_carlo, gpu_memory_frac=args.gpu_memory_frac,
                                    verbose=True, multioutput=True, random_seed=42, validation_fraction=0.15,
                                    max_epochs=100, patience=5, bayesian=(args.nn_type == 'bayesian'))
        nn.fit(train_texts, train_labels)
        print('')
        with open(model_name, 'wb') as fp:
            pickle.dump((nn, train_classes), fp)
    test_texts, test_labels, test_classes = read_dstc2_data(test_file_name, train_classes)
    assert test_classes == train_classes, 'Classes in the test set do not correspond to classes in the train set! ' \
                                          '{0}'.format(test_classes)
    print('')
    print('Number of samples for final testing is {0}.'.format(len(test_texts)))
    y_pred = nn.predict(test_texts)
    accuracy_by_classes = dict()
    for class_idx in range(nn.n_classes_):
        n_total = 0
        n_correct = 0
        for sample_idx in range(len(test_texts)):
            if isinstance(test_labels[sample_idx], set):
                if class_idx in test_labels[sample_idx]:
                    y_true_ = 1
                else:
                    y_true_ = 0
            else:
                if class_idx == test_labels[sample_idx]:
                    y_true_ = 1
                else:
                    y_true_ = 0
            if isinstance(y_pred[sample_idx], set):
                if class_idx in y_pred[sample_idx]:
                    y_pred_ = 1
                else:
                    y_pred_ = 0
            else:
                if class_idx == y_pred[sample_idx]:
                    y_pred_ = 1
                else:
                    y_pred_ = 0
            if y_true_ == y_pred_:
                n_correct += 1
            if y_true_ > 0:
                n_total += 1
        if n_total > 0:
            accuracy_by_classes[class_idx] = float(n_correct) / float(len(test_texts))
    total_accuracy = 0.0
    name_width = 0
    for class_idx in accuracy_by_classes.keys():
        total_accuracy += accuracy_by_classes[class_idx]
        if len(test_classes[class_idx]) > name_width:
            name_width = len(test_classes[class_idx])
    total_accuracy /= float(len(accuracy_by_classes))
    print('Total accuracy: {0:6.2%}'.format(total_accuracy))
    print('By classes:')
    for class_idx in sorted(list(accuracy_by_classes.keys())):
        print('  {0:<{1}} {2:6.2%}'.format(test_classes[class_idx], name_width, accuracy_by_classes[class_idx]))


if __name__ == '__main__':
    main()
