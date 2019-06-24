import codecs
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


def read_snips2017_data(dir_name: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray],
                                                Tuple[np.ndarray, np.ndarray], List[str]]:
    true_intents = {'addtoplaylist', 'bookrestaurant', 'getweather', 'playmusic', 'ratebook', 'searchcreativework',
                    'searchscreeningevent'}
    intents = list(filter(lambda it: it.lower() in true_intents, os.listdir(dir_name)))
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
        intent_idx = true_intents.index(os.path.basename(cur_subdir).lower())
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
        labels = [intent_idx for _ in range(len(texts))]
        data_for_training += texts
        labels_for_training += labels
        texts = read_snips2017_file(os.path.join(cur_subdir, 'train_{0}.json'.format(base_intent_dir)))
        labels = [intent_idx for _ in range(len(texts))]
        data_for_validation += texts
        labels_for_validation += labels
        texts = read_snips2017_file(os.path.join(cur_subdir, 'validate_{0}.json'.format(base_intent_dir)))
        labels = [intent_idx for _ in range(len(texts))]
        data_for_final_testing += texts
        labels_for_final_testing += labels
    return (np.array(data_for_training, dtype=object), np.array(labels_for_training, dtype=np.int32)), \
           (np.array(data_for_validation, dtype=object), np.array(labels_for_validation, dtype=np.int32)), \
           (np.array(data_for_final_testing, dtype=object), np.array(labels_for_final_testing, dtype=np.int32)), \
           true_intents
