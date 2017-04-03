.. hcc-encoding documentation master file, created by
   sphinx-quickstart on Sun Apr 02 21:31:44 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://travis-ci.org/MacHu-GWU/elementary_math-project.svg?branch=master

.. image:: https://img.shields.io/pypi/v/hccEncoding.svg

.. image:: https://img.shields.io/pypi/l/hccEncoding.svg

.. image:: https://img.shields.io/pypi/pyversions/hccEncoding.svg




Welcome to hccEncoding's documentation!
========================================

Categorical data fields characterized by a large number of distinct values represent a serious challenge for many classification and regression algorithms that require numerical inputs. On the other hand, these types of data fields are quite common in real-world data mining applications and often contain potentially relevant information that is difficult to represent for modeling purposes.

This python package implements some preprocessing strategies for high-cardinality categorical data that allows this class of attributes to be used in predictive models. Currently there are two major methods, whihc are based on Daniele Micci-Barreca 's empirical Bayes method [ref1] and Owen Zhang's leave-one-out encoding[ref2].

.. toctree::
   :maxdepth: 2

Functions
------------------
* BayesEncoding(train,test,target,feature,k=5,f=1,noise=0.01,drop_origin_feature=False)
* BayesEncodingKfold(train,test,target,feature,k=5,f=1,noise=0.01,drop_origin_feature=False,fold=5)
* LOOEncoding(train,test,target,feature,noise=0.01,drop_origin_feature=False)
* LOOEncodingKfold(train,test,target,feature,noise=0.01,drop_origin_feature=False,fold=5)

**Please see example for detailed explanation**
- `Example <https://github.com/Robin888/hccEncoding-project/blob/master/example/Example.ipynb>`_


General Parameters
------------------
* train 
  - train dataset, datatype: pandas dataframe
* test 
  - test dataset,  datatype: pandas dataframe
* target 
  - name of target for prediction, datatype: string
* feature 
  - name of features that need to be encoded, datatype: string
* k [default=5]
  - parameter for BayesEncoding and BayesEncodingKfold, determines half of the minimal sample size of which we completely 'trust' the estimate of transition between the cell's posterior probability and the prior probability, datatype: int
* f [default=1]
  - parameter for BayesEncoding and BayesEncodingKfold,controls how quickly the weight changes from the prior to the posterior as the size of the group increases, to further understand k and f's meaning  datatype: int  
* noise [default=0.01]
  - a manually added noise after encoding. For classification problems, a random uniform-distributed noise in the range of [-noise,noise]*data is added. For regression problem, a random normal-distributed noise in the range of norm(0,noise) is added, datatype: double
* drop_origin_feature [default=False]
  - whether dropping the original feature or not, datatype: boolean
* kfold [default=5]
  - parameter for LOOEncodingKfold and BayesEncodingKfold, represent the number of folds that the train dataset will be splitted into. datatype: int


References
==================
* ref1: Daniele Micci-Barreca. 2001. A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. SIGKDD Explor. Newsl. 3, 1 (July 2001), 27-32.
* ref2: - `https://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions <https://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

