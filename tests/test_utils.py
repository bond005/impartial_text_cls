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

import os
import re
import sys
import unittest

import numpy as np

try:
    from impartial_text_cls.utils import read_dstc2_data, read_snips2017_data, str_to_layers, read_csv
    from impartial_text_cls.utils import parse_hidden_layers_description
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from impartial_text_cls.utils import read_dstc2_data, read_snips2017_data, str_to_layers, read_csv
    from impartial_text_cls.utils import parse_hidden_layers_description


class TestUtils(unittest.TestCase):
    def test_read_dstc2_data_positive01(self):
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

    def test_read_dstc2_data_positive02(self):
        file_name = os.path.join(os.path.dirname(__file__), 'test_dataset.tar.gz')
        true_classes_list = ['affirm', 'hello', 'inform_area', 'inform_food', 'inform_pricerange',  'request_addr',
                             'request_phone', 'thankyou']
        loaded_texts, loaded_labels, loaded_classes_list = read_dstc2_data(file_name, true_classes_list)
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
                {4, 2},
                0,
                {5, 6},
                7,
                4,
                2,
                -1,
                5,
                7,
                1,
                2,
                -1,
                3,
                3,
                6,
                7,
                {3, 4},
                -1,
                5,
                7
            ],
            dtype=object
        )
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

    def test_read_snips2017_data(self):
        true_train_texts = [
            'Add another song to the Cita Romántica playlist.',
            'add clem burke in my playlist Pre-Party R&B Jams',
            'Add Live from Aragon Ballroom to Trapeo',
            'book The Middle East restaurant in IN for noon',
            'Book a table at T-Rex distant from Halsey St.',
            'I\'d like to eat at a taverna that serves chili con carne for a party of 10',
            'What will the weather be this year in Horseshoe Lake State Fish and Wildlife Area?',
            'Will it be sunny one hundred thirty five days from now in Monterey Bay National Marine Sanctuary',
            'Is it supposed to rain nearby my current location at 0 o\'clock?',
            'I need to hear the song Aspro Mavro from Bill Szymczyk on Youtube',
            'play Yo Ho from the new york pops on Youtube',
            'Play some seventies music by Janne Puurtinen on Youtube.',
            'rate The Lotus and the Storm zero of 6',
            'Rate The Fall-Down Artist 5 stars.',
            'Rate the current novel one points',
            'find the soundtrack titled This Side of Paradise',
            'find a book called The Mad Magician',
            'find the picture Louder Than Bombs',
            'What are the movie schedule at Malco Theatres',
            'I want to get the movie schedule',
            'Show me movie time for I Am Sorry at my movie house'
        ]
        true_train_labels = [
            'AddToPlaylist',
            'AddToPlaylist',
            'AddToPlaylist',
            'BookRestaurant',
            'BookRestaurant',
            'BookRestaurant',
            'GetWeather',
            'GetWeather',
            'GetWeather',
            'PlayMusic',
            'PlayMusic',
            'PlayMusic',
            'RateBook',
            'RateBook',
            'RateBook',
            'SearchCreativeWork',
            'SearchCreativeWork',
            'SearchCreativeWork',
            'SearchScreeningEvent',
            'SearchScreeningEvent',
            'SearchScreeningEvent'
        ]
        true_val_texts = [
            'add Stani, stani Ibar vodo songs in my playlist música libre',
            'add this album to my Blues playlist',
            'Book a reservation for seven people at a bakery in Osage City',
            'Book spot for three at Maid-Rite Sandwich Shop in Antigua and Barbuda',
            'How\'s the weather in Munchique National Natural Park',
            'Tell me the weather forecast for France',
            'I\'d like to hear music that\'s popular from Trick-trick on the Slacker service',
            'Play Making Out by Alexander Rosenbaum off Google Music.',
            'Rate All That Remains a five',
            'Give this album 4 points',
            'Please help me find the Bloom: Remix Album song.',
            'Find me the soundtrack called Enter the Chicken',
            'Find movie times for Landmark Theatres.',
            'What are the movie times for Amco Entertainment'
        ]
        true_val_labels = [
            'AddToPlaylist',
            'AddToPlaylist',
            'BookRestaurant',
            'BookRestaurant',
            'GetWeather',
            'GetWeather',
            'PlayMusic',
            'PlayMusic',
            'RateBook',
            'RateBook',
            'SearchCreativeWork',
            'SearchCreativeWork',
            'SearchScreeningEvent',
            'SearchScreeningEvent'
        ]
        true_test_texts = [
            'I\'d like to have this track onto my Classical Relaxations playlist.',
            'Book a reservation for my babies and I',
            'What will the weather be faraway from here?',
            'can you put on Like A Hurricane by Paul Landers',
            'rate this album four out of 6 stars',
            'Wish to find the movie the Heart Beat',
            'Is Babar: King of the Elephants playing'
        ]
        true_test_labels = [
            'AddToPlaylist',
            'BookRestaurant',
            'GetWeather',
            'PlayMusic',
            'RateBook',
            'SearchCreativeWork',
            'SearchScreeningEvent'
        ]
        loaded_train_data, loaded_val_data, loaded_test_data = read_snips2017_data(
            os.path.join(os.path.dirname(__file__), 'test_snips2017')
        )
        self.assertIsInstance(loaded_train_data, tuple)
        self.assertIsInstance(loaded_val_data, tuple)
        self.assertIsInstance(loaded_test_data, tuple)
        self.assertEqual(len(loaded_train_data), 2)
        self.assertEqual(len(loaded_val_data), 2)
        self.assertEqual(len(loaded_test_data), 2)
        self.assertIsInstance(loaded_train_data[0], list)
        self.assertIsInstance(loaded_train_data[1], list)
        self.assertEqual(len(loaded_train_data[0]), len(loaded_train_data[1]))
        self.assertIsInstance(loaded_val_data[0], list)
        self.assertIsInstance(loaded_val_data[1], list)
        self.assertEqual(len(loaded_val_data[0]), len(loaded_val_data[1]))
        self.assertIsInstance(loaded_test_data[0], list)
        self.assertIsInstance(loaded_test_data[1], list)
        self.assertEqual(len(loaded_test_data[0]), len(loaded_test_data[1]))
        self.assertEqual(true_train_texts, loaded_train_data[0])
        self.assertEqual(true_train_labels, loaded_train_data[1])
        self.assertEqual(true_val_texts, loaded_val_data[0])
        self.assertEqual(true_val_labels, loaded_val_data[1])
        self.assertEqual(true_test_texts, loaded_test_data[0])
        self.assertEqual(true_test_labels, loaded_test_data[1])

    def test_read_csv_positive01(self):
        file_name = os.path.join(os.path.dirname(__file__), 'csv_for_testing1.csv')
        true_classes = ['Криминал', 'Медицина', 'Политика', 'Спорт']
        true_texts = [
            'Названы регионы России с самой высокой смертностью от рака.',
            'Испанские клубы открестились от Неймара.',
            'Семилетняя беженка погибла после задержания на границе США.',
            'Главная реформа Обамы признана неконституционной.',
            'Бывший чемпион UFC не выдержал кровопролития и сдался.',
            'Охранника магазина зарезали из-за трех бутылок водки.',
            'Лукашенко пожаловался Путину на украинских «отмороженных нацменов».'
        ]
        true_labels = [1, 3, 0, 2, 3, 0, 2]
        loaded_texts, loaded_labels, loaded_classes = read_csv(file_name)
        self.assertIsInstance(loaded_classes, list)
        self.assertEqual(true_classes, loaded_classes)
        self.assertIsInstance(loaded_texts, np.ndarray)
        self.assertIsInstance(loaded_labels, np.ndarray)
        self.assertEqual(loaded_texts.shape, (len(true_texts),))
        self.assertEqual(loaded_labels.shape, (len(true_labels),))
        self.assertEqual(true_texts, loaded_texts.tolist())
        self.assertEqual(true_labels, loaded_labels.tolist())

    def test_read_csv_positive02(self):
        file_name = os.path.join(os.path.dirname(__file__), 'csv_for_testing2.csv')
        true_classes = ['Криминал', 'Культура', 'Наука', 'Общество', 'Политика', 'Экономика']
        true_texts = [
            'Германия разрешила третий пол.',
            'Лукашенко объяснил поставки оружия Азербайджану.',
            'Шакиру уличили в неуплате налогов.',
            'Люди в масках ворвались на собрание врачей в Киеве и отказались их выпускать.',
            'Подсчитаны расходы на российскую сверхтяжелую ракету для освоения Луны.',
            'Популярные актеры дебютируют в короткометражных хоррорах.',
            'Россия ответила на высылку дипломата из Словакии.',
            'Российские нефтяники заработали триллионы на фоне бензинового кризиса.',
            'Google выбрал своего «Человека года».'
        ]
        true_labels = [3, {4, 5}, {0, 1}, 0, {2, 5}, 1, 4, 5, 3]
        loaded_texts, loaded_labels, loaded_classes = read_csv(file_name)
        self.assertIsInstance(loaded_classes, list)
        self.assertEqual(true_classes, loaded_classes)
        self.assertIsInstance(loaded_texts, np.ndarray)
        self.assertIsInstance(loaded_labels, np.ndarray)
        self.assertEqual(loaded_texts.shape, (len(true_texts),))
        self.assertEqual(loaded_labels.shape, (len(true_labels),))
        self.assertEqual(true_texts, loaded_texts.tolist())
        self.assertEqual(true_labels, loaded_labels.tolist())

    def test_read_csv_positive03(self):
        file_name = os.path.join(os.path.dirname(__file__), 'csv_for_testing2.csv')
        true_classes = ['Криминал', 'Культура', 'Общество', 'Политика', 'Экономика']
        true_texts = [
            'Германия разрешила третий пол.',
            'Лукашенко объяснил поставки оружия Азербайджану.',
            'Шакиру уличили в неуплате налогов.',
            'Люди в масках ворвались на собрание врачей в Киеве и отказались их выпускать.',
            'Подсчитаны расходы на российскую сверхтяжелую ракету для освоения Луны.',
            'Популярные актеры дебютируют в короткометражных хоррорах.',
            'Россия ответила на высылку дипломата из Словакии.',
            'Российские нефтяники заработали триллионы на фоне бензинового кризиса.',
            'Google выбрал своего «Человека года».'
        ]
        true_labels = [2, {3, 4}, {0, 1}, 0, 4, 1, 3, 4, 2]
        loaded_texts, loaded_labels, loaded_classes = read_csv(file_name, 1)
        self.assertIsInstance(loaded_classes, list)
        self.assertEqual(true_classes, loaded_classes)
        self.assertIsInstance(loaded_texts, np.ndarray)
        self.assertIsInstance(loaded_labels, np.ndarray)
        self.assertEqual(loaded_texts.shape, (len(true_texts),))
        self.assertEqual(loaded_labels.shape, (len(true_labels),))
        self.assertEqual(true_texts, loaded_texts.tolist())
        self.assertEqual(true_labels, loaded_labels.tolist())

    def test_read_csv_positive04(self):
        file_name = os.path.join(os.path.dirname(__file__), 'csv_for_testing1.csv')
        true_classes = ['Криминал', 'Политика', 'Спорт']
        true_texts = [
            'Испанские клубы открестились от Неймара.',
            'Семилетняя беженка погибла после задержания на границе США.',
            'Главная реформа Обамы признана неконституционной.',
            'Бывший чемпион UFC не выдержал кровопролития и сдался.',
            'Охранника магазина зарезали из-за трех бутылок водки.',
            'Лукашенко пожаловался Путину на украинских «отмороженных нацменов».'
        ]
        true_labels = [2, 0, 1, 2, 0, 1]
        loaded_texts, loaded_labels, loaded_classes = read_csv(file_name, 1)
        self.assertIsInstance(loaded_classes, list)
        self.assertEqual(true_classes, loaded_classes)
        self.assertIsInstance(loaded_texts, np.ndarray)
        self.assertIsInstance(loaded_labels, np.ndarray)
        self.assertEqual(loaded_texts.shape, (len(true_texts),))
        self.assertEqual(loaded_labels.shape, (len(true_labels),))
        self.assertEqual(true_texts, loaded_texts.tolist())
        self.assertEqual(true_labels, loaded_labels.tolist())

    def test_parse_hidden_layers_description_positive01(self):
        src = '100'
        true_res = (100, 1)
        self.assertEqual(true_res, parse_hidden_layers_description(src))

    def test_parse_hidden_layers_description_positive02(self):
        src = '100:3'
        true_res = (100, 3)
        self.assertEqual(true_res, parse_hidden_layers_description(src))

    def test_parse_hidden_layers_description_positive03(self):
        src = '100:0'
        true_res = (0, 0)
        self.assertEqual(true_res, parse_hidden_layers_description(src))

    def test_parse_hidden_layers_description_positive04(self):
        src = '0:2'
        true_res = (0, 0)
        self.assertEqual(true_res, parse_hidden_layers_description(src))

    def test_parse_hidden_layers_description_positive05(self):
        src = '0:0'
        true_res = (0, 0)
        self.assertEqual(true_res, parse_hidden_layers_description(src))

    def test_parse_hidden_layers_description_positive06(self):
        src = ''
        true_res = (0, 0)
        self.assertEqual(true_res, parse_hidden_layers_description(src))

    def test_parse_hidden_layers_description_positive07(self):
        src = None
        true_res = (0, 0)
        self.assertEqual(true_res, parse_hidden_layers_description(src))

    def test_parse_hidden_layers_description_negative01(self):
        src = ':'
        true_err_msg = re.escape('Description of hidden layers is empty!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = parse_hidden_layers_description(src)

    def test_parse_hidden_layers_description_negative02(self):
        src = '100:3:2'
        true_err_msg = re.escape('`100:3:2` is wrong description of hidden layers!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = parse_hidden_layers_description(src)

    def test_parse_hidden_layers_description_negative03(self):
        src = '100:2a'
        true_err_msg = re.escape('`100:2a` is wrong description of hidden layers!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = parse_hidden_layers_description(src)

    def test_parse_hidden_layers_description_negative04(self):
        src = '100:-1'
        true_err_msg = re.escape('`100:-1` is wrong description of hidden layers!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = parse_hidden_layers_description(src)


if __name__ == '__main__':
    unittest.main(verbosity=2)
