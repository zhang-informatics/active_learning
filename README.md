# Evaluating Active Learning Methods for Annotating Semantic Predications Extracted from MEDLINE
The precision of the semantic predications extracted by [SemRep](https://semrep.nlm.nih.gov/) is low.
The precision can be improved by filtering out incorrect predications using a machine learning classifier. However,
training a classification model requires obtaining expert-annotated data. This study evaluates the use of active learning,
a method for reducing the annotation cost for machine learning tasks, on training a classifier to filter incorrect SemRep output.

Link: **TBD**

## Getting Started
These instructions will show you how to get the code running on your system in order to recreate the results from the paper.

### Prerequisites
* [Python 3](www.python.org/downloads)
* [NumPy](www.numpy.org)
* [SciPy](www.scipy.org)
* [scikit-learn](www.scikit-learn.org)

### Installing
Get the code: `git clone https://github.com/zhang-informatics/active_learning`

Download the data from: **TBD**

To see usage instructions:
```
python3 al.py -h
```

### Running Experiments
To run an active learning experiment, e.g.
```
python al.py --nfolds 10 --cvdir cv_data --resultsdir results_random random data/tfidf_n11_cui.csv.gz
```
`--nfolds 10`: Evaluated using 10-fold cross validation.

`--cvdir`: Save the cross validation splits in the `cv_data` directory.

`--resultsdir`: Save the results of the evaluations in the `results_random` directory.

`random`: Run the passive learning baseline query strategy.

`data/tfidf_n11_cui.csv.gz`: Use the features specified in this file (1gram tf-idf and subject/object CUI features).
By default the system uses a linear SVM as the machine-learning model. This can be changed by modifying `al.py`.

#### Supported query strategies
- Random (passive learning) `random`
- Uncertainty Sampling
  * Entropy Sampling `entropy`
  * Least Confidence `least_confidence`
  * Least Confidence with Bias `lcb`
  * Least Confidence with Dynamic Bias `lcb2`
  * Margin Sampling `margin`
  * Simple Margin Sampling `simple_margin`
- Representative Sampling
  * Density Sampling `density`
  * Distance to Center `d2c`
  * MinMax Sampling `minmax`
- Combined Sampling Methods
  * Beta-weighted Combined Sampling `combined`

Parameters for the chosen query strategy are passed via the `--qs_kwargs` flag.

#### Supported query strategy parameters:
- Uncertainty Sampling: `model_change={True,False}`
- Representative Sampling: `metric={'distance_metric'}`. `distance_metric` can be any metric from the scipy.spatial.distance package.
  * MinMax Sampling: `LDS={True,False}, LDS_k=int, LDS_threshold={int,'auto'}`
- Combined Sampling Methods
  * Beta-weighted Combined Sampling: `qs1='query_strategy', qs2='query_strategy', beta={int,'dynamic'}, alpha={float,'auto'}`

## Running Tests
Run tests with
```
python -m unittest discover
```
From the project directory.

## Authors
* **Jake Vasilakes** - Study design, programming, data collection, and annotation.
* **Rui Zhang** - Study design.
* **Rubina Rizvi** - Data annotation.
See also the list of authors on the associated paper.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements
This research was supported by National Center for Complementary & Integrative Health Award (#R01AT009457) (Zhang),
the Agency for Healthcare Research & Quality grant (#1R01HS022085) (Melton),
and the National Center for Advancing Translational Science (#U01TR002062) (Liu/Pakhomov/Jiang)
