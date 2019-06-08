import os
import re
import sys
import unittest

import numpy as np

try:
    from impatial_text_cls.utils import read_dstc2_data, str_to_layers
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from impatial_text_cls.utils import read_dstc2_data, str_to_layers


class TestUtils(unittest.TestCase):
    def test_read_dstc2_data(self):
        file_name = os.path.join(os.path.dirname(__file__), 'test_dataset.tar.gz')
        loaded_texts, loaded_labels, loaded_classes_list = read_dstc2_data(file_name)
        true_texts = np.array(
            [
                'moderately priced north part of town',
                'yes',
                'what is the address and phone number',
                'thank you good bye',
                'expensive',
                'south',
                'dont care',
                'what is the address',
                'thank you good bye',
                'hello welcome',
                'south',
                'would you like something',
                'steak house',
                'indian',
                'and whats the phone number',
                'thank you good bye',
                'i need a cheap restaurant serving italian food',
                'i dont care',
                'could i get the address',
                'thank you bye'
            ],
            dtype=object
        )
        true_labels = np.array(
            [
                {5, 3},
                0,
                {7, 8},
                {9, 1},
                5,
                3,
                6,
                7,
                {9, 1},
                2,
                3,
                -1,
                4,
                4,
                8,
                {9, 1},
                {4, 5},
                6,
                7,
                {9, 1}
            ],
            dtype=object
        )
        true_classes_list = ['affirm', 'bye', 'hello', 'inform_area', 'inform_food', 'inform_pricerange', 'inform_this',
                             'request_addr', 'request_phone', 'thankyou']
        self.assertIsInstance(loaded_classes_list, list)
        self.assertEqual(true_classes_list, loaded_classes_list)
        self.assertIsInstance(loaded_texts, np.ndarray)
        self.assertIsInstance(loaded_labels, np.ndarray)
        self.assertEqual(true_texts.shape, loaded_texts.shape)
        self.assertEqual(true_labels.shape, loaded_labels.shape)
        self.assertEqual(true_texts.tolist(), loaded_texts.tolist())
        self.assertEqual(true_labels.tolist(), loaded_labels.tolist())

    def test_str_to_layers_positive01(self):
        src = '100-50'
        true_res = [100, 50]
        calc_res = str_to_layers(src)
        self.assertEqual(true_res, calc_res)

    def test_str_to_layers_positive02(self):
        src = '100'
        true_res = [100]
        calc_res = str_to_layers(src)
        self.assertEqual(true_res, calc_res)

    def test_str_to_layers_negative01(self):
        src = '100-a-50'
        true_err_msg = re.escape('`100-a-50` is wrong description of layer sizes!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = str_to_layers(src)

    def test_str_to_layers_negative02(self):
        src = ''
        true_err_msg = re.escape('`` is wrong description of layer sizes!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = str_to_layers(src)

    def test_str_to_layers_negative03(self):
        src = '100-0-50'
        true_err_msg = re.escape('`100-0-50` is wrong description of layer sizes!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = str_to_layers(src)


if __name__ == '__main__':
    unittest.main(verbosity=2)
