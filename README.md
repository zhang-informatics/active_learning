# Evaluating Active Learning Methods for Annotating Semantic Predications Extracted from MEDLINE
The precision of the semantic predications extracted by [SemRep](https://semrep.nlm.nih.gov/) is low.
The precision can be improved by filtering out incorrect predications using a machine learning classifier. However,
training a classification model requires obtaining expert-annotated data. This study evaluates the use of active learning,
a method for reducing the annotation cost for machine learning tasks, on training a classifier to filter incorrect SemRep output.

## Citation

Vasilakes J, Rizvi R, Melton GB, Pakhomov S, Zhang R. Evaluating active learning methods for annotating semantic predications. JAMIA Open. 2018;1(2):275-282.

## Getting Started
These instructions will show you how to get the code running on your system in order to recreate the results from the paper.
See also [Example.ipynb](Example.ipynb) for example code.

### Prerequisites
* [Python 3](www.python.org/downloads)
* [NumPy](www.numpy.org)
* [SciPy](www.scipy.org)
* [scikit-learn](www.scikit-learn.org)

### Installing
Get the code: `git clone https://github.com/zhang-informatics/active_learning`

Download the data from: [https://doi.org/10.5061/dryad.k4b688s](https://doi.org/10.5061/dryad.k4b688s)

To see usage instructions:
```
python3 al.py -h
```

### Feature Extraction
To extract the same feature set as was used in the paper, first separate the full data file by PREDICATE_GROUP into the substance interactions
and clinical medicine predications. Then extract and combine the tf-idf and CUI features. For example, for the substance interaction predications:
```
python scripts/python/semmed_2_features.py --feature_type tfidf --keep_percentage 42 data/substance_interactions/annotated_predications.csv \
					   data/substance_interactions/train/tfidf_n11_features.csv
python scripts/python/semmed_2_features.py --feature_type semmed --keep_percentage 100 data/substance_interactions/annotated_predications.csv \
					   data/substance_interactions/train/semmed_features.csv
python scripts/python/svm.py --semmeddb_csv data/substance_interactions/train/semmed_features.csv \
			     --tfidf_csv data/substance_interactions/train/tfidf_n11_features.csv \
			     --tfidf_keep_percentage 12 --run_comb --classifier sgd --loss_func hinge \
			     --features cui_feature --comb_keep_percentage 65 \
			     --save_comb_X data/substance_interactions/train/tfidf_n11_cui_features.csv
```
The last command should output an AUC of **0.835**. 

The commands are the same for the clinical medicine data except for the path to the data files and
for scripts/python/svm.py set `--tfidf_keep_percentage 13` and `--comb_keep_percentage 66`. This command should output an AUC of **0.804**.

### Running Experiments
To run an active learning experiment, e.g.
```
python al.py --nfolds 10 --cvdir cv_data --resultsdir results/ random data/substance_interactions/train/tfidf_n11_cui.csv
```
`--nfolds 10`: Evaluated using 10-fold cross validation.

`--cvdir`: Save the cross validation splits in the `cv_data/` directory.

`--resultsdir`: Save the results of the evaluations in the `results/` directory.

`random`: Run the passive learning baseline query strategy.

`data/substance_interactions/train/tfidf_n11_cui.csv`: Use the features specified in this file (1gram tf-idf and subject/object CUI features).

This command will create the directory `cv_data/` if it does not exist or use the existing CV splits if it does exist.
It will also create the directory `results/` if it does not exist. The script will abort if the specified 
`--resultsdir` already exists.

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
  See the [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html) for more information. 
- Combined Sampling Methods
  * Beta-weighted Combined Sampling: `qs1='query_strategy', qs2='query_strategy', beta={int,'dynamic'}, alpha={float,'auto'}`

## Running Tests
From the project home directory run
```
python -m unittest discover
```

## Authors
* **Jake Vasilakes** - Study design, programming, data collection, and annotation.
* **Rui Zhang** - Study conception and design.
* **Rubina Rizvi** - Data annotation.

See also the list of authors on the associated paper.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements
This research was supported by National Center for Complementary & Integrative Health Award (#R01AT009457) (Zhang),
the Agency for Healthcare Research & Quality grant (#1R01HS022085) (Melton),
and the National Center for Advancing Translational Science (#U01TR002062) (Liu/Pakhomov/Jiang)
