"""
Composition-based decision tree for  anomaly detection
-------------------------------
CDT detector.

:authors: Ines Ben Kraiem & Geoffrey Roman-Jimenez

:copyright:
    Copyright 2020 SIG Research Group, IRIT, Toulouse-France.
    

"""

from os import listdir
from os.path import isfile, join
import time
import numpy as np
import pandas as pd
import argparse
import random
import uuid
import pickle
import itertools

from labelisation import labelisation_automatic
from CompositionTree_RCIS import composition_tree

def cdt_compute_result(dirdataset,cdt, window, step, nbsplitlabel, kernel_size, kernel_stride):
	""" Test the newly labeled time-series on CDT and calculate metrics evaluation.

    Parameters
    ----------
    
    dirdataset : CSV files with two columns column (Class: the class of anomaly, Value: the measure of sensors)
        time-series data set for testing.
    cdt : object
        the classifier
    window : int
        Fixed window size. 
    step : int
        The step of sliding window.
    nbsplitlabel : int
        The number of splits to identify different magnitudes of the variety of patterns.
	kernel_size : int
		Size of moving average for downsampling
	kernel_stride : int 
		The step of moving average for downsampling
		
	Returns TP, TN, FP, FN, acc, recall, precision, f1
        -------
        TP : float
            True positive: anomalous observation identified as anomalous (i.e., true alarms)
		TN : float
			True negative : normal observation identified as normal (i.e., successful detections)
		FP : float 
			False positive: normal observation identified as anomalous(i.e., false alarms)
		FN : float
			False negative : anomalous observation identified as normal (i.e., missed detections)
		acc : float
			Accuracy : the ratio of the correctly labeled observations to the whole pool of observations.
		recall : float
			quantifies the number of positive class predictions made out of all positive examples in the dataset.
		precision : float
			quantifies the number of positive class predictions that actually belong to the positive class.
		f1 : float
			 the harmonic mean of precision and recall.
        	
    """
    list_file = [_f for _f in listdir(dirdataset) if isfile(join(dirdataset, _f))]
    mind = (window-1)/2

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

        _values = [np.mean(_values[x:x+kernel_size]) for x in np.arange(0,len(_values), kernel_stride)]
        _fclasses = [1 if 1 in _fclasses[x:x+kernel_size] else 0 for x in np.arange(0,len(_fclasses), kernel_stride)]

        min_values = min(_values)
        max_values = max(_values)
        _values = [(v-min_values)/(max_values - min_values) for v in _values]
        _features = labelisation_automatic(_values, div=nbsplitlabel, epsilon=0.05)
        _ = _features.pop(0)
        _ = _fclasses.pop(0)
        _ = _values.pop(0)
        _ = _features.pop(-1)
        _ = _fclasses.pop(-1)
        _ = _values.pop(-1)

        nb_observation += len( list(dataset["Class"]) )

        _labels = [x for i, x in enumerate(_features) if i == _features.index(x)]
        _uclasses = [x for i, x in enumerate(_fclasses) if i == _fclasses.index(x)]
        _nlabels = len(_labels)

        _features = [_features[x:x+window] for x in np.arange(0,len(_features)-window+1, step)]
        _fclasses = [_fclasses[x:x+window] for x in np.arange(0,len(_fclasses)-window+1, step)]
        _values = [_values[x:x+window] for x in np.arange(0,len(_values)-window+1, step)]

        _classes = [0 for _ in range(len(_fclasses))]

        for i, f in enumerate(_fclasses):

            uniqueclasses = [x for i, x in enumerate(f) if x != 0]
            uniqueclasses = [x for i, x in enumerate(uniqueclasses) if i==uniqueclasses.index(x)]

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
    nclasses = len([x for i, x in enumerate(classes) if i == classes.index(x)])
    uclasses = [x for i, x in enumerate(classes) if i == classes.index(x)]

    
    TP, TN, FP, FN= 0, 0, 0, 0

    for observation, true_class in zip(features, classes):
        _, pred_class = cdt.which_leaf(observation)
        if true_class == 0 and pred_class == 0: TN +=1
        if true_class == 0 and pred_class == 1: FP +=1
        if true_class == 1 and pred_class == 0: FN +=1
        if true_class == 1 and pred_class == 1: TP +=1
    acc= (TP+TN)/(TP+TN+FP+FN)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    f1 = (2*recall*precision ) / (recall+precision)
    return TP, TN, FP, FN, acc, recall, precision, f1