import json
import tarfile
from typing import List, Tuple

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


def read_dstc2_data(archive_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    texts = []
    labels = []
    classes = set()
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
                        if 'act' not in semantics[act_idx]:
                            raise ValueError(err_msg)
                        intents.add(semantics[act_idx]['act'])
                        classes.add(semantics[act_idx]['act'])
                    texts.append(text)
                    labels.append(intents)
    classes = sorted(list(classes))
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
