"""
Composition-based decision tree for anomaly detection
-------------------------------
CDT detector.

:authors: Ines Ben Kraiem & Geoffrey Roman-Jimenez

:copyright:
    Copyright 2020 SIG Research Group, IRIT, Toulouse-France.
    

"""

import numpy as np
import pandas as pd
import argparse
import random
import uuid 
import pickle
import itertools

from CompositionConditionTree import composition_condition_tree

from os import listdir
from os.path import isfile, join



def labelisation_automatic(values, div=4, epsilon=1):
	""" Labelization method using patterns.

    Parameters
    ----------
    values : list
        List of continuous time -series data.
    div : int
        number of split for patterns.
    epsilon: float
		The margin of error to consider a remarkable point comm constant (cst)
		
	Returns
        -------
        labels : np.array
            labeled time-series data.
    """

    divstep = 100/div 
    l_tuple = []
    for i in range(div):
        l_tuple.append( (max(100*epsilon,i*divstep), (i+1)*divstep) )
    
    labels = [ "NL" for _ in values]
    for i in np.arange(1, len(values)-1,1):
        y= values[i]
        y_m = values[i-1]
        y_p = values[i+1]
        dym = y-y_m
        dyp = y-y_p
		""" 
		We define nine general patterns used for liberalization: 
		- PP (Positive Peak)
		- PN (Negative Peak)
		- SCP (Start Constant Positive)
		- SCN (Start Constant Negative)
		- ECP (End Constant Positive)
		- ECN (End Con-stant Negative
		- CST (Constant
		- VP (Variation Positive
		- VN (VariationNegative)
		
		"""
        for (lsm_min, lsm_max), (lsp_min, lsp_max) in list(itertools.product(l_tuple, repeat=2)): 
            sm_max = lsm_max/100
            sm_min = lsm_min/100
            sp_max = lsp_max/100
            sp_min = lsp_min/100

            if sm_min < dym <= sm_max and sp_min < dyp <= sp_max :
                labels[i] = "PP"+"_"+str(lsm_max)+"_"+str(lsp_max)
            elif -sm_max <= dym < -sm_min and -sp_max <= dyp < -sp_min :
                labels[i] = "PN"+"_"+str(lsm_max)+"_"+str(lsp_max)
            elif -epsilon <= dym <= epsilon and sp_min < dyp <= sp_max :
                labels[i] = "FCstN"+"_"+str(0)+"_"+str(lsp_max)
            elif -epsilon <= dym <= epsilon and -sp_max <= dyp < -sp_min :
                labels[i] = "FCstP"+"_"+str(0)+"_"+str(lsp_max)
            elif sm_min < dym <= sm_max and -epsilon <= dyp <= epsilon :
                labels[i] = "DCstP"+"_"+str(lsm_max)+"_"+str(0)
            elif -sm_max <= dym < -sm_min and -epsilon <= dyp <= epsilon :
                labels[i] = "DCstN"+"_"+str(lsm_max)+"_"+str(0)
            elif sm_min < dym <= sm_max and -sp_max <= dyp < -sp_min :
                labels[i] = "VP"+"_"+str(lsm_max)+"_"+str(lsp_max)
            elif -sm_max <= dym < -sm_min and sp_min < dyp <= sp_max :
                labels[i] = "VN"+"_"+str(lsm_max)+"_"+str(lsp_max)
            elif -epsilon <= dym <= epsilon and -epsilon <= dyp <= epsilon :
                labels[i] = "Cst"
        #print("results:", dym, dyp, labels[i])
    return labels




