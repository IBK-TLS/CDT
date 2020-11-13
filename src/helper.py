from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from CDT_labelisation import labelisation_automatic


def convert_label_RCIS(label):
    dict_label = {
        "100.0": "H",
        "75.0": "MH",
        "50.0": "ML",
        "25.0": "L",
  		    "0": "0"
    }
    # ex label = ["PN_100.0_50.0", "Cst"]
    list_lab = label.split("_")  # ["PN", "100.0", "50.0"]

    #new_list_lab = [list_lab[0], dict_label[list_lab[1]], dict_label[list_lab[2]] ]
    new_list_lab = [list_lab[0]]

    if label == "Cst":
        return label
    if list_lab[0] in ["PN", "SCN", "VN"]:
        new_list_lab.append("-"+dict_label[list_lab[1]])
    else:
        new_list_lab.append(dict_label[list_lab[1]])

    if list_lab[0] in ["PN", "ECN", "VN"]:
        new_list_lab.append("-"+dict_label[list_lab[2]])
    else:
        new_list_lab.append(dict_label[list_lab[2]])

    return new_list_lab[0]+"("+new_list_lab[1]+","+new_list_lab[2]+")"


def prepare_dataset(dirdataset, kernel_size=1, kernel_stride=1, nbsplitlabel=4, window=5):

    list_file = [f for f in listdir(dirdataset) if isfile(join(dirdataset, f))]

    step = 1

    features = []
    classes = []
    fclasses = []
    values = []
    labels = []
    nb_observation = 0

    for i, f in enumerate(list_file):
        dataset = pd.read_csv(dirdataset+"/"+f)
        _fclasses = list(dataset["Class"])
        _values = list(dataset["Value"])

        _values = [np.mean(_values[x:x+kernel_size])
                   for x in np.arange(0, len(_values), kernel_stride)]
        _fclasses = [1 if 1 in _fclasses[x:x+kernel_size]
                     else 0 for x in np.arange(0, len(_fclasses), kernel_stride)]

        min_values = min(_values)
        max_values = max(_values)
        _values = [(v-min_values)/(max_values - min_values) for v in _values]
        _features = labelisation_automatic(
            _values, div=nbsplitlabel, epsilon=0.05)
        _ = _features.pop(0)
        _ = _fclasses.pop(0)
        _ = _values.pop(0)
        _ = _features.pop(-1)
        _ = _fclasses.pop(-1)
        _ = _values.pop(-1)

        nb_observation += len(list(dataset["Class"]))

        _labels = [x for i, x in enumerate(
            _features) if i == _features.index(x)]
        _uclasses = [x for i, x in enumerate(
            _fclasses) if i == _fclasses.index(x)]
        _nlabels = len(_labels)

        _features = [_features[x:x+window]
                     for x in np.arange(0, len(_features)-window+1, step)]
        _fclasses = [_fclasses[x:x+window]
                     for x in np.arange(0, len(_fclasses)-window+1, step)]
        _values = [_values[x:x+window]
                   for x in np.arange(0, len(_values)-window+1, step)]

        _classes = [0 for _ in range(len(_fclasses))]

        for i, f in enumerate(_fclasses):

            uniqueclasses = [x for i, x in enumerate(f) if x != 0]
            uniqueclasses = [x for i, x in enumerate(
                uniqueclasses) if i == uniqueclasses.index(x)]

            if not len(uniqueclasses) == 0:
                if len(uniqueclasses) == 1:
                    _classes[i] = uniqueclasses[0]
                else:
                    _classes[i] = uniqueclasses

        features_to_keep = _features
        fclasses_to_keep = _fclasses
        classes_to_keep = _classes
        labels_to_keep = _labels
        values_to_keep = _values

        features = features + features_to_keep
        fclasses = fclasses + fclasses_to_keep
        classes = classes + classes_to_keep
        values = values + values_to_keep
        labels = labels + labels_to_keep

    labels = [x for i, x in enumerate(labels) if i == labels.index(x)]

    return features, classes
