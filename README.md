## Composition-based Decision Tree
Implementation of *Composition-based Decision Tree for anomaly detection in uni-variate Time Series*, by Geoffrey Roman-Jimenez and Ines Ben Kraiem. 

Paper published at 14th International Conference on Research Challengesin Information Science (RCIS) (2020).

>Anomaly detection in time-series is an important issue in many applications. It is particularly hard to accurately detect multiple anomalies in time-series. Pattern discovery >and rule extraction are effective solutions for allowing multiple anomaly detection. 
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
CDT consists of ..... for pre-processing and methods.CDT for predicting multiple anomalies.

Parameters for methods.PreProcessor are:

window_size and step control the creation of fixed sized sliding windows in continous time series.
dow can be used for downsampling a continuous timeseries using a moving average.
....

Parameters for methods.PBAD are:
....

Datasets are provided in /data:
* SGE data sets: Univariate Time series Calorie data consists of readings from 25 meters deployed in different buildings.
They contain anomalies of different types such as  positives peaks (PP), negatives peaks (PN), and sudden variations (VN, VP)
* Yahoo’s S5 Webscope Dataset: which is publicly available on [this link](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70), consists of 371 files divided into four categories, named A1/A2/A3 and A4. 

### Contributors
* Geoffrey Roman-Jimenez, IRIT, CNRS , France.
* Ines Ben Kraiem, IRIT, University of Toulouse 2, France.

### Licence
