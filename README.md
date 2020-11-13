## Composition-based Decision Tree
Implementation of *Composition-based Decision Tree for anomaly detection in uni-variate Time Series*, by Ines Ben Kraiem and Geoffrey Roman-Jimenez. 

Paper published at 14th International Conference on Research Challengesin Information Science (RCIS) (2020).

>Anomaly detection in time-series is an important issue in many applications. It is particularly hard to accurately detect multiple anomalies in time-series. Pattern discovery and rule extraction are effective solutions for allowing multiple anomaly detection. 
>In this paper, we define a Composition-based Decision Tree algorithm that automatically discovers and generates human-understandable classification rules for multiple anomaly >detection in time-series. To evaluate our solution, our algorithm is compared to other anomaly detection algorithms on real datasets and benchmarks.

[Full paper](https://link.springer.com/chapter/10.1007/978-3-030-50316-1_19)

### Summary
**CDT** uses sequences of patterns to identify remarkable points corresponding to multiple anomalies. The compositions of patterns existing into time-series are learned through an internally generated decision tree and then simplified using Boolean algebra to produce intelligible rules.
CDT consist of 4 major steps:
1. Time-series are pre-processed by performing a normalization of values.
2. The pre-defined patterns are used to automatically create labeled time-series.
3. A modified decision tree is created to construct an anomaly detection classifier and to generate decision rules.
4. Rule simplification and generation. 

### Installation
* Code is implemented in Python
* Clone the repository

### Usage 
CDT consists of *labialization* for labeling time-series using patterns and *CompositionTree_RCIS* for detecting multiple anomalies using modified decision-tree.

Parameters for *labialization* are:

* `nbsplitlabel`: The number of splits to identify different magnitudes of the variety of patterns. 

Parameters for *Preprocessing* are:
* `window and step`: control the creation of fixed sized sliding windows in continous time series.
* `kernel_size and kernel_stride`: can be used for downsampling a continuous timeseries using a moving average.

Parameters for CompositionTree_RCIS are:
* CompositionTree_RCIS.composition_tree
    * `nclasses`: is the number of classes ( in our approach  2 classes: 0=normal, 1=anomaly) 

* CompositionTree_RCIS.fit are:
    * `features`: is the observations calculated by sliding window
    * `classes`: is the list of classes corresponding to each observation (window)

We illustrate theses classes in the notebook `/src/experiments_RCIS_CompositionTree.ipynb`.

Datasets are provided in `/data`:
* SGE data sets: Univariate Time series Calorie data consists of readings from 25 meters deployed in different buildings.
They contain anomalies of different types such as  positives peaks (PP), negatives peaks (PN), and sudden variations (VN, VP)
* Yahoo’s S5 Webscope Dataset: which is publicly available on [this link](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70), consists of 371 files divided into four categories, named A1/A2/A3 and A4. 

### Contributors
* Ines Ben Kraiem, IRIT, University of Toulouse 2, France.
* Geoffrey Roman-Jimenez, IRIT, CNRS , France.


### Licence
Copyright (c) 2020 Ines Ben Kraiem and Geoffrey Roman-Jimenez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
