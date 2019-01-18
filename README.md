# Project Guideline

In this project, you will explore how to detect malware using machine learning (ML) algorithms trained on the basis of a voluminous dataset provided by Microsoft.
The concrete goal is to predict if a Windows machine will get infected by various families of malware, based on different properties of that machine.
This document will help you get a better understanding of each steps/component you need to accomplish in order to build a ML pipeline shown as follows.

![alt text](MalwareDetectionPipeline.png "General Machine Learning Pipeline")

This project will be graded from 0~5 points, with 1 bonus points.

## ML Tools (0 Point)
If you haven't take CS 583 (machine learning), you should first get familiar with several tools heavily used in this project.

* [Python 3](https://www.python.org/downloads/): Your programming language for this project is Python 3.
* [Numpy](http://www.numpy.org/): NumPy is the fundamental package for scientific computing with Python, but we will mostly use it as an efficient multi-dimensional container of generic data.
* [Pandas](https://pandas.pydata.org/): Pandas is a Python library providing high-performance, easy-to-use data structures and data analysis tools. For this project, you will definitely use its convenient `read_csv()` API, among many other useful functions.
* [Scikit-Learn](https://scikit-learn.org/stable/index.html): A simple and efficient toolset for many ML algorithms. We will use several ML models implemented in this package, e.g. RandomForestClassifier and GradientBoostingClassifier.
* (Optional) [XGBoost](https://xgboost.readthedocs.io/en/latest/index.html): XGBoost is an more advanced package for GradientBoosting, comparing to sklearn's GradientBoosting. It is not required unless you want to tune high performance gradient boosting models.
* [Keras](https://keras.io/): a convenient deep neural network package written in Python and capable of running on top of TensorFlow.


## Dataset Processing (2 Points)
Our raw dataset is available at [Kaggle](https://www.kaggle.com/c/microsoft-malware-prediction/data). You can either download manually from the website pointed by the link, or pull using kaggle’s command line API:
```
kaggle competitions download -c microsoft-malware-prediction
```
It’s around 2.5 GB compressed, and 7.8 GB after unzipped.
Read the data description to get a rough view of the structure of the data.
Simply put, it is a table where each row depicts a machine.
In the training dataset (train.csv), we know if the machine is infected already by some malware;
while in the testing dataset (test.csv), we don’t and need to make a prediction for each machine.
As ML algorithms usually take numerical data as input,
our first step will be convert raw data into a numeric matrix X whose dimension is #samples by #features and a label vector Y of size #samples. You need to consider how to handle the following issues when you do so:

* Split **features** (many machine properties) from **labels** (if machine is infected).
* How to handle missing values, or the special “NA” (not available).
* Convert categorical features (features whose values are string) to numeric. Hint: use sklearn’s [OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder).
* Testing dataset won’t have labels.
* Is is necessary to cache/pickle the data you processed?
* Do you need to scale the features? Hint: e.g. [min-max scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) or [zero-mean scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)

Technically, your task in this step is to implement several functions defined in `data_process.py`.
Firstly the major function that process the .csv data file:
```python
def processDataset(filename, hasLabel=True, portions=[0.1]):
    """Properly process the .csv file named @filename.
    Args:
        filename: name of the .csv file, NOT including the '.csv' extension.
        hashLabel: true if the file has label (training data), otherwise false.
        portions: list of floats, at the end of the processing,
                  for each x in portions, pickling 100*x% of the data.
    Returns:
        X, a numpy matrix of size #samples x #features
        Y, if @hasLabel is True, an numpy array of size #samples, otherwise None.
        featureNames, the corresponding name of X's each column.
    """
    raise NotImplementedError()
```
The number of samples in train.csv is huge, 8,921,483 in total.
We will only use 1% of the data (89,215) to play around several ML algorithms.
Implement the following function to randomly sample 1% of the entire dataset and cache the result into a pickle file.
```python
def pklDataPortion(X, Y, featureNames, pklname, portion=0.1):
    """Given entire dataset (X, Y), randomly pick @portion% of the data from it,
    and pickles the sampled data into the file named @pklname.
    Args:
        X, a numpy matrix of size #samples x #features.
        Y, an numpy array of size #samples.
        featureNames, the corresponding name of X's each column.
        pklname, where the pickled datas should be written to.
        portion, pickle only part of the dataset.
    """
    raise NotImplementedError()
```
There is an data sample file even smaller than 1% of the provided dataset, `train_samples.csv`. It contains only 4,000 rows and you can use it during your coding for debugging.

## Feature Engineering (Optional, 0 Point)
There are many ways to engineer features in the dataset before fitting a model with the dataset.
In this project however we will just use an informal way to get the importance of each raw features from decision-tree-based models.
You can get a score of a feature's importance from most decision-tree-based ensemble models (e.g. RandomForestClassifier and GradientBoostingClassifier we will use later).
Implement the following function in `cross_valid.py` to rank features using the information provided by an trained classifier:
```python
def rankFeatures(clf, featureNames, top=10):
    """Given a trained classifier that is able to rank features,
    sort the feature names in @featureNames, and print several top features.
    Args:
        cls: a fitted classifier having the scores of feature importances.
        featureNames: names of the features.
        top: how many top features to print to the stdout.
    Returns:
        sortedFeatures, feature names sorted by its importance.
        sortedFeatureImportances, the scores for each item in sortedFeatures.
    """
    raise NotImplementedError()
```


## Model Training (3 Points)
Before diving into ML models, we need a way to evaluate multiple model's performance.
The candidate metrics are accuracy, precision, recall, F1 score, and AUC score.
It should be easy for you to calculate them once you google their math definitions, or even just use sklearn's simple [APIs](https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation).
A slightly harder question for you is which metrics we should use.
I will just use accuracy (accu) as an example hereafter.

### Cross Validation (CV)
We will adopt the cross-validation technique for evaluating models.
By splitting the dataset into `k` folds, we can run `k` rounds of train-valid procedures where the validation set in each round is unique.
In the train-valid procedure, we fit the model with `k-1` folds of data and evaluate the model using the rest 1 fold of data.
Your first task is to implement the cross validation procedure in `cross_valid.py` as follows:
```python
def crossValidate(clf, X, Y, featureNames, trainFunc, fold=5):
    """Cross validate using stratified k-fold split.
    Args:
        clf: a sklearn classifier of any type.
        (X, Y): feature matrix and label vector from the dataset.
        featureNames: column names of @X.
        tranFunc: a function object that implements how to fit the @clf.
        fold: number of fold to run for cross validation.
    Returns:
        Metrics on each validation fold as a dictionary, e.g.
            result['ValidAccuList'] = [accu_1, accu_2, ..., accu_k]
            result['ValidLossList'] = [loss_1, loss_2, ..., loss_k]
            ...
    """
    raise NotImplementedError()
```
You may find [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) useful in implementing cross validation.

### Ensemble Learning Based Models
Ensemble learning based models has achieved good results in many classification problems.
In this project we will try two of them.
The first model, defined in `cv_rf.py`, is [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
Your task is to implement the following function used as the trainFunc function object in `crossValidate`:
```python
def trainRandomForest(clf, trainX, trainY, validX, validY,
                      featureNames, k=0, top=10):
    """How to train a random forest classifier.
    Args:
        clf: expect to be a sklearn RandomForestClassifier object.
        (trainX, trainY, validX, validY): splitted train/valid dataset.
        featureNames: column names of @trainX and @validX.
        k: which round of cross validation.
        top: print top-x features.
    Returns:
        clf: fitted classifier.
        trainAccu: final training accuracy.
        validAccu: final validation accuracy.
    """
    raise NotImplementedError()
```
The parameter `top` in `trainRandomForest` is meaningless if you choose not to do feature engineering.

Once done, it should be straightforward how to measure the cross-validation performance of [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) with the following to-be-implemented-by-you function in `cv_gbt.py`, :
```python
def trainGradientBoost(clf, trainX, trainY, validX, validY,
                       featureNames, k=0, top=5):
    """How to train a gradient boosting classifier.
    Args:
        clf: expect to be a sklearn GradientBoostingClassifier object.
        (trainX, trainY, validX, validY): splitted train/valid dataset.
        featureNames: column names of @trainX and @validX.
        k: which round of cross validation.
        top: print top-x features.
    Returns:
        clf: fitted classifier.
        trainAccu: final training accuracy.
        validAccu: final validation accuracy.
    """
    raise NotImplementedError()
```

Try to improve the average performance of your validation metric(s) for different models, and compare their best performance.

Now be creative: pick a model implemented by sklearn and try it out on this malware detection dataset.
Explain what is the main reason you pick the model, and report its cross validation score (e.g. accuracy, F1 score, AUC score).

### Deep Neural Networks
I will provide detailed instructions here when the class finishes ensemble learning.


## Model Evaluation (Optional, 1 Bonus Points)
If you like chanllenges, train your best model with the entire training dataset, or a large portion of it.
Then do a prediction over the testing dataset (of course you need to process testing set first).
Submit your prediction results to [Kaggle](https://www.kaggle.com/c/malware-classification).
Let me know if you are really interested in participate in the Kaggle contest and/or get into any trouble.
If most of the class want to do this part, I will provide further details.
Otherwise, I will only speak to the volunteers, e.g. provide them with more software tools and/or powerful GPU server.
Finishing this part (after my review) will give you one **honorable** bonus point.
