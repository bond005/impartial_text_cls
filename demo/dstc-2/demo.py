from argparse import ArgumentParser
import os
import pickle
import sys


try:
    from impatial_text_cls.impatial_text_cls import ImpatialTextClassifier
    from impatial_text_cls.utils import read_dstc2_data, str_to_layers
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from impatial_text_cls.impatial_text_cls import ImpatialTextClassifier
    from impatial_text_cls.utils import read_dstc2_data, str_to_layers


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The binary file with the text classifier.')
    parser.add_argument('-t', '--train', dest='train_file_name', type=str, required=True,
                        help='Path to the archive with DSTC-2 training data.')
    parser.add_argument('-e', '--test', dest='test_file_name', type=str, required=True,
                        help='Path to the archive with DSTC-2 data for final testing.')
    parser.add_argument('--layers', dest='sizes_of_layers', type=str, required=False, default='300-100',
                        help='Sizes of the Bayesian neural network layers.')
    parser.add_argument('--num_monte_carlo', dest='num_monte_carlo', type=int, required=False, default=10,
                        help='Number of generated Monte Carlo samples for each data sample.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, required=False, default=16,
                        help='Size of mini-batch.')
    parser.add_argument('--gpu_frac', dest='gpu_memory_frac', type=float, required=False, default=0.9,
                        help='Allocable part of the GPU memory for the classifier.')
    args = parser.parse_args()

    model_name = os.path.normpath(args.model_name)
    train_file_name = os.path.normpath(args.train_file_name)
    test_file_name = os.path.normpath(args.test_file_name)
    layers = str_to_layers(args.sizes_of_layers)

    if os.path.isfile(model_name):
        with open(model_name, 'rb') as fp:
            nn, train_classes = pickle.load(fp)
        print('Classes list: {0}'.format(train_classes))
        print('')
    else:
        train_texts, train_labels, train_classes = read_dstc2_data(train_file_name)
        print('Classes list: {0}'.format(train_classes))
        print('Number of samples for training is {0}.'.format(len(train_texts)))
        nn = ImpatialTextClassifier(hidden_layer_sizes=layers, batch_size=args.batch_size,
                                    num_monte_carlo=args.num_monte_carlo, gpu_memory_frac=args.gpu_memory_frac,
                                    verbose=True, multioutput=True, random_seed=42, validation_fraction=0.15,
                                    max_epochs=100, patience=5)
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
    n_errors = 0
    n_total = 0
    for sample_idx in range(len(test_labels)):
        if isinstance(test_labels[sample_idx], set):
            true_classes = test_labels[sample_idx]
        else:
            true_classes = {test_labels[sample_idx]}
        if isinstance(y_pred[sample_idx], set):
            pred_classes = y_pred[sample_idx]
        else:
            pred_classes = {y_pred[sample_idx]}
        n_errors += len((true_classes | pred_classes) - (true_classes & pred_classes))
        n_total += len(true_classes)
    accuracy = 1.0 - float(n_errors) / float(n_total)
    print('Test accuracy is {0:.2%}.'.format(accuracy))


if __name__ == '__main__':
    main()
