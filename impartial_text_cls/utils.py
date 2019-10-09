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

import codecs
import csv
import json
import os
import tarfile
from typing import List, Tuple, Union

import numpy as np


def str_to_layers(descr: str) -> List[int]:
    err_msg = '`{0}` is wrong description of layer sizes!'.format(descr)
    parts = list(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), descr.split('-'))))
    if len(parts) == 0:
        raise ValueError(err_msg)
    try:
        sizes = [int(cur) for cur in parts]
    except:
        sizes = []
    if len(sizes) == 0:
        raise ValueError(err_msg)
    if any(map(lambda it: it <= 0, sizes)):
        raise ValueError(err_msg)
    return sizes


def parse_hidden_layers_description(hidden_layer: Union[str, None]) -> Tuple[int, int]:
    if hidden_layer is None:
        return (0, 0)
    if len(hidden_layer.strip()) == 0:
        return (0, 0)
    parts = list(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), hidden_layer.split(':'))))
    if len(parts) < 1:
        raise ValueError('Description of hidden layers is empty!')
    if len(parts) > 2:
        raise ValueError('`{0}` is wrong description of hidden layers!'.format(hidden_layer))
    if not parts[0].isdigit():
        raise ValueError('`{0}` is wrong description of hidden layers!'.format(hidden_layer))
    hidden_layer_size = int(parts[0])
    if hidden_layer_size < 0:
        raise ValueError('`{0}` is wrong description of hidden layers!'.format(hidden_layer))
    if len(parts) > 1:
        if not parts[1].isdigit():
            raise ValueError('`{0}` is wrong description of hidden layers!'.format(hidden_layer))
        number_of_hidden_layers = int(parts[1])
        if number_of_hidden_layers < 0:
            raise ValueError('`{0}` is wrong description of hidden layers!'.format(hidden_layer))
        if number_of_hidden_layers == 0:
            hidden_layer_size = 0
    else:
        number_of_hidden_layers = 1
    if hidden_layer_size == 0:
        number_of_hidden_layers = 0
    return (hidden_layer_size, number_of_hidden_layers)


def read_dstc2_data(archive_name: str, classes_list: Union[None, List[str]]=None) -> \
        Tuple[np.ndarray, np.ndarray, List[str]]:
    texts = []
    labels = []
    if classes_list is None:
        classes = set()
    else:
        classes = set(classes_list)
    with tarfile.open(archive_name, "r:gz") as tar_fp:
        for member in filter(lambda it2: it2.lower().endswith('label.json'),
                             sorted(list(map(lambda it1: it1.name, tar_fp.getmembers())))):
            data_fp = tar_fp.extractfile(member)
            if data_fp is not None:
                data = json.load(data_fp)
                if not isinstance(data, dict):
                    raise ValueError('Archive `{0}`: file `{1}` is wrong! This file must contain a dictionary in a '
                                     'JSON format!'.format(archive_name, member))
                if 'turns' not in data:
                    raise ValueError('Archive `{0}`: file `{1}` is wrong! A dictionary from this file must contain the '
                                     '`turns` key!'.format(archive_name, member))
                samples = data['turns']
                if not isinstance(samples, list):
                    raise ValueError('Archive `{0}`: file `{1}` is wrong! Value for the `turns` key must be a '
                                     'list!'.format(archive_name, member))
                for idx in range(len(samples)):
                    err_msg = 'Archive `{0}`, file `{1}`: item {2} of the `turns` data list is ' \
                              'wrong!'.format(archive_name, member, idx)
                    cur_sample = samples[idx]
                    if not isinstance(cur_sample, dict):
                        raise ValueError(err_msg)
                    if 'transcription' not in cur_sample:
                        raise ValueError(err_msg)
                    if 'semantics' not in cur_sample:
                        raise ValueError(err_msg)
                    if not isinstance(cur_sample['semantics'], dict):
                        raise ValueError(err_msg)
                    if 'json' not in cur_sample['semantics']:
                        raise ValueError(err_msg)
                    semantics = cur_sample['semantics']['json']
                    if not isinstance(semantics, list):
                        raise ValueError(err_msg)
                    text = cur_sample['transcription']
                    intents = set()
                    for act_idx in range(len(semantics)):
                        err_msg = 'Archive `{0}`, file `{1}`: semantic act {2} of the item {3} is wrong!'.format(
                            archive_name, member, act_idx, idx)
                        if not isinstance(semantics[act_idx], dict):
                            raise ValueError(err_msg)
                        if ('act' not in semantics[act_idx]) or ('slots' not in semantics[act_idx]):
                            raise ValueError(err_msg)
                        if not isinstance(semantics[act_idx]['slots'], list):
                            raise ValueError(err_msg)
                        new_intent = semantics[act_idx]['act']
                        if len(semantics[act_idx]['slots']) > 0:
                            if len(semantics[act_idx]['slots']) != 1:
                                raise ValueError(err_msg)
                            slots = semantics[act_idx]['slots'][0]
                            if not isinstance(slots, list):
                                raise ValueError(err_msg)
                            if len(slots) != 2:
                                raise ValueError(err_msg)
                            if slots[0] == 'slot':
                                new_intent += ('_' + slots[1])
                            else:
                                new_intent += ('_' + slots[0])
                        if classes_list is None:
                            intents.add(new_intent)
                            classes.add(new_intent)
                        else:
                            if new_intent in classes:
                                intents.add(new_intent)
                    texts.append(text)
                    labels.append(intents)
    if classes_list is None:
        classes = sorted(list(classes))
    else:
        classes = classes_list
    IDs = []
    for cur_label in labels:
        if len(cur_label) > 1:
            IDs.append(set(map(lambda class_name: classes.index(class_name), cur_label)))
        elif len(cur_label) == 1:
            IDs.append(classes.index(list(cur_label)[0]))
        else:
            IDs.append(-1)
    del labels
    return np.array(texts, dtype=object), np.array(IDs, dtype=object), classes


def read_snips2017_file(file_name: str) -> List[str]:
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        source_str = ' '.join(
            list(filter(
                lambda it: len(it.strip()) > 0,
                fp.read().replace('\u2060', ' ').replace('\uFEFF', '').split()
            ))
        )
        source_data = json.loads(source_str)
    if not isinstance(source_data, dict):
        raise ValueError('Data in the file `{0}` are wrong! Expected `{1}`, got `{2}`'.format(
            file_name, type({'a': 1, 'b': 2}), type(source_data)))
    intent_name = os.path.basename(os.path.dirname(file_name))
    if intent_name not in source_data:
        raise ValueError('Data in the file `{0}` are wrong! The key `{1}` is expected in the dictionary'.format(
            file_name, intent_name))
    samples = source_data[intent_name]
    if not isinstance(samples, list):
        raise ValueError('Data in the file `{0}` are wrong! Samples list must be a `{1}`, but it is a `{2}`.'.format(
            file_name, type([1, 2]), type(samples)))
    all_texts = []
    for sample_idx in range(len(samples)):
        cur_sample = samples[sample_idx]
        err_msg = 'Data in the file `{0}` are wrong! Sample {1} contains incorrect information!'.format(
            file_name, sample_idx)
        if not isinstance(cur_sample, dict):
            raise ValueError(err_msg)
        if 'data' not in cur_sample:
            raise ValueError(err_msg)
        subtexts = cur_sample['data']
        if not isinstance(subtexts, list):
            raise ValueError(err_msg)
        new_text = ''
        for cur_subtext in subtexts:
            if not isinstance(cur_subtext, dict):
                raise ValueError(err_msg)
            if not 'text' in cur_subtext:
                raise ValueError(err_msg)
            if not isinstance(cur_subtext['text'], str):
                raise ValueError(err_msg)
            new_text += cur_subtext['text']
        all_texts.append(' '.join(new_text.strip().split()))
    return all_texts


def read_snips2017_data(dir_name: str) -> Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]],
                                                Tuple[List[str], List[str]]]:
    true_intents = {'AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork',
                    'SearchScreeningEvent'}
    intents = list(filter(lambda it: it in true_intents, os.listdir(dir_name)))
    if len(intents) != len(true_intents):
        raise ValueError('The directory `{0}` does not contain the SNIPS-2017 dataset!'.format(dir_name))
    subdirs_with_intents = sorted([os.path.join(dir_name, cur) for cur in intents])
    data_for_training = []
    labels_for_training = []
    data_for_validation = []
    labels_for_validation = []
    data_for_final_testing = []
    labels_for_final_testing = []
    true_intents = sorted(list(true_intents))
    for cur_subdir in subdirs_with_intents:
        base_intent_dir = os.path.basename(cur_subdir)
        intent_name = os.path.basename(cur_subdir)
        if not os.path.isfile(os.path.join(cur_subdir, 'train_{0}.json'.format(base_intent_dir))):
            raise ValueError('The file `{0}` does not exist!'.format(
                os.path.join(cur_subdir, 'train_{0}.json'.format(base_intent_dir))))
        if not os.path.isfile(os.path.join(cur_subdir, 'train_{0}_full.json'.format(base_intent_dir))):
            raise ValueError('The file `{0}` does not exist!'.format(
                os.path.join(cur_subdir, 'train_{0}_full.json'.format(base_intent_dir))))
        if not os.path.isfile(os.path.join(cur_subdir, 'validate_{0}.json'.format(base_intent_dir))):
            raise ValueError('The file `{0}` does not exist!'.format(
                os.path.join(cur_subdir, 'validate_{0}.json'.format(base_intent_dir))))
        texts = read_snips2017_file(os.path.join(cur_subdir, 'train_{0}_full.json'.format(base_intent_dir)))
        labels = [intent_name for _ in range(len(texts))]
        data_for_training += texts
        labels_for_training += labels
        texts = read_snips2017_file(os.path.join(cur_subdir, 'train_{0}.json'.format(base_intent_dir)))
        labels = [intent_name for _ in range(len(texts))]
        data_for_validation += texts
        labels_for_validation += labels
        texts = read_snips2017_file(os.path.join(cur_subdir, 'validate_{0}.json'.format(base_intent_dir)))
        labels = [intent_name for _ in range(len(texts))]
        data_for_final_testing += texts
        labels_for_final_testing += labels
    return (data_for_training, labels_for_training), (data_for_validation, labels_for_validation), \
           (data_for_final_testing, labels_for_final_testing)


def read_csv(file_name: str, min_freq: int=0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    texts = []
    labels = []
    line_idx = 1
    set_of_classes = set()
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        reader = csv.reader(fp, quotechar='"', delimiter=',')
        for row in reader:
            if len(row) > 0:
                err_msg = 'File `{0}`: line {1} is wrong!'.format(file_name, line_idx)
                if len(row) != 2:
                    raise ValueError(err_msg)
                new_text = row[0].strip()
                new_label = row[1].strip()
                if len(new_label) == 0:
                    raise ValueError(err_msg)
                texts.append(new_text)
                new_label = set(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), new_label.split('::'))))
                if len(new_label) == 0:
                    raise ValueError(err_msg)
                set_of_classes |= new_label
                if len(new_label) == 1:
                    labels.append(new_label.pop())
                else:
                    labels.append(new_label)
            line_idx += 1
    set_of_classes = sorted(list(set_of_classes))
    if len(set_of_classes) < 2:
        raise ValueError('Only single class is represented in the file `{0}`!'.format(file_name))
    classes_distr = dict()
    for cur in labels:
        if isinstance(cur, set):
            for class_name in cur:
                classes_distr[class_name] = classes_distr.get(class_name, 0) + 1
        else:
            classes_distr[cur] = classes_distr.get(cur, 0) + 1
    set_of_classes = sorted(list(filter(lambda class_name: classes_distr.get(class_name, 0) > min_freq,
                                        classes_distr.keys())))
    if len(set_of_classes) < 2:
        raise ValueError('Only single class is represented in the file `{0}`!'.format(file_name))
    label_indices = []
    multioutput = False
    filtered_texts = []
    for idx, cur in enumerate(labels):
        if isinstance(cur, set):
            new_ = set(map(lambda it2: set_of_classes.index(it2), filter(lambda it1: it1 in set_of_classes, cur)))
            if len(new_) > 0:
                if len(new_) > 1:
                    label_indices.append(new_)
                    multioutput = True
                else:
                    label_indices.append(new_.pop())
                filtered_texts.append(texts[idx])
        else:
            if cur in set_of_classes:
                label_indices.append(set_of_classes.index(cur))
                filtered_texts.append(texts[idx])
    return np.array(filtered_texts, dtype=object), np.array(label_indices, dtype=object if multioutput else np.int32), \
           set_of_classes
