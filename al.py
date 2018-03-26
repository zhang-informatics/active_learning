#! /usr/bin/env python
"""
The active learning model driver script.  Reads the data, runs the active
learning model, and saves the results. The machine learning model is hard-
coded as an SVM using sklearn's SGDClassifier. See run_active_learner() for
details.
"""

import os
import sys
import time
import tracemalloc
import re
import warnings
import argparse
import pickle

import numpy as np
import pandas as pd

from sklearn import __version__
sklearn_ver = __version__
from sklearn import model_selection
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import SelectPercentile, f_classif, chi2

from activelearning import ActiveLearningModel
from activelearning.querystrategies import *


class IllegalArgumentError(ValueError):
    '''
    Illegal command line arguments.
    '''
    pass


def timeit(method):
    '''
    Compute execution time for method.
    :param function method: Function to time.
    :returns: Wrapped input method.
    '''
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        print("{0} ({1}, {2}) {3:2f} sec"
               .format(method.__name__, args, kwargs, te - ts),
              flush=True)
        return result
    return timed

def safe_makedirs(directory):
    '''
    Accounts for a possible race-condition creating the directory.
    :param str directory: Directory to create.
    '''
    try:
        os.makedirs(directory)
    except OSError:
        pass

# TODO: save to binary or compressed format.
def make_cv_files(X, y, feature_names, cv_basepath,
                  nfolds=10, random_state=None):
    '''
    Split X and y data into nfold cross-validation sets.
    Save these sets at 'cv_basepath_<type>_<fold>.csv'.
    :param array-like X: Features.
    :param array-like y: Labels.
    :param str cv_basepath: Path to directory where files will be saved.
                            Includes first part of filename.
    :param int nfolds: Number of cross-validation folds.
    :param int random_state: Sets the random_state parameter of StratifiedKFold.
    '''
    if not os.path.exists(os.path.dirname(cv_basepath)):
        safe_makedirs(os.path.dirname(cv_basepath))
    folds = model_selection.StratifiedKFold(n_splits=nfolds, shuffle=True,
                                            random_state=random_state)
    for (i, (traincv, testcv)) in enumerate(folds.split(X, y)):
        train_x = X[traincv,:]
        test_x = X[testcv,:]
        train_y = y[traincv]
        test_y = y[testcv]
        x_header = ','.join(feature_names)
        y_header = "T/F"
        # E.g. cv_splits/10/01/all_test-x_01.csv
        template = "{0}_{1}_{2:02d}.csv".format(cv_basepath, "{}", i + 1)
        np.savetxt(template.format("train-x"), train_x,
                   delimiter=',', header=x_header, comments='')
        np.savetxt(template.format("test-x"), test_x,
                   delimiter=',', header=x_header, comments='')
        np.savetxt(template.format("train-y"), train_y,
                   delimiter=',', header=y_header, comments='')
        np.savetxt(template.format("test-y"), test_y,
                   delimiter=',', header=y_header, comments='')

def read_cv_files(cv_basepath, cv_fold):
    '''
    Reads cross-validation data with filenames of the format
    <col>_<type>_<i>.csv E.g. cv_splits/10/5/all_train-x_01.csv
    :param str cv_basepath: Where cross-validation files are stored.
                            Includes first part of filename.
    :param int cv_fold: Cross-validation fold. 
    :returns: Training and test data.
    :rtype: tuple(pd.DataFrame * 4)
    '''
    template = "{0}_{1}_{2:02d}.csv".format(cv_basepath, "{}", cv_fold)
    train_x = pd.read_csv(template.format("train-x"), compression="infer")
    test_x = pd.read_csv(template.format("test-x"), compression="infer")
    train_y = pd.read_csv(template.format("train-y"), compression="infer")
    test_y = pd.read_csv(template.format("test-y"), compression="infer")
    # Cast as numpy.ndarray
    train_x = train_x.values.astype('double')
    test_x = test_x.values.astype('double')
    train_y = np.ravel(train_y.values.astype('int'))
    test_y = np.ravel(test_y.values.astype('int'))
    return (train_x, test_x, train_y, test_y)

def select_features(filename, column, percentile, features_outfile=None):
    '''
    Selects the top <percentile> features from the dataset.
    :param str filename: Training data file.
    :param str column: Column in the CSV to use. If 'all' use all columns.
    :param int percentile: Percentile top features to choose.
    :returns: Training data with top percentile features. Labels. Names of selected features.
    :rtype: 3-tuple
    '''
    train = pd.read_csv(filename, sep=',', compression="infer")
    targets = LabelBinarizer().fit_transform(train["T/F"])
    targets = np.ravel(targets)
    train.drop(["T/F"], axis=1, inplace=True)
    # Encase the feature names in quotes to ensure proper parsing later.
    feature_names = np.array(['"{}"'.format(f) for f in np.array(train.columns)])
    train = train.values.astype('double')
    if percentile < 100:
        feature_finder = SelectPercentile(f_classif, percentile=percentile)
        train = feature_finder.fit_transform(train, targets)  
        support = feature_finder.get_support()
        scores = feature_finder.scores_
        pvals = feature_finder.pvalues_
        feature_names = feature_names[support]
        if features_outfile is not None:
            feature_scores = scores[support]
            feature_pvals = pvals[support]
            features = zip(features, feature_scores, feature_pvals)
            rank = sorted(features, key=lambda x: x[1], reverse=True)
            with open(features_outfile, 'w') as outF:
                for feat in rank:
                    outF.write("{} :: {:g} :: {:g}\n".format(*feat))
    return train, targets, feature_names

def get_query_strategy(qs_str, qs_kwargs):
    '''
    Gets the QueryStrategy instance corresponding to the input string, 
    and the arguments to pass to the QueryStrategy.query() method.

    :param str qs_str: String for the query strategy to use.
    :param dict qs_kwargs: Keyword arguments for the QueryStrategy.
    :returns: QueryStrategy instance.
    :rtype: QueryStrategy
    '''
    kwargs_copy = qs_kwargs.copy()  # Don't modify qs_kwargs.
    qs_map = {'random':           Random,
              'simple_margin':    SimpleMargin,
              'margin':           Margin,
              'entropy':          Entropy,
              'least_confidence': LeastConfidence,
              'lcb':              LeastConfidenceBias,
              'lcb2':             LeastConfidenceDynamicBias,
              'd2c':              DistanceToCenter,
              'minmax':           MinMax,
              'density':          Density,
              'combined':         CombinedSampler,
             }
    try:
        QS = qs_map[qs_str]
    except KeyError:
        raise ValueError("Query strategy '{}' is not supported."
                          .format(qs_str))
    if QS == CombinedSampler:
        # Pass in QueryStrategy instances, not strings.
        qs1 = qs_map[qs_kwargs['qs1']]
        qs2 = qs_map[qs_kwargs['qs2']]
        kwargs_copy['qs1'] = qs1()
        kwargs_copy['qs2'] = qs2()
    qs = QS(**kwargs_copy) 
    return qs

def process_qs_kwargs(kwargs):
    kwargs = [arg.split('=') for arg in kwargs]
    bool_map = {"true": True, "false": False}
    kwargs = [(key, bool_map[val.lower()]) if val.lower() in bool_map.keys()
              else (key, val) for (key, val) in kwargs]
    return dict(kwargs)

@timeit
def run_active_learner(cv_basepath, nfolds, ndraws,
                       qs_str, model_change, qs_kwargs,
                       save_profile_to=None):
    '''
    Runs the active learner over nfolds cross-validation, reading
    data from cv_basepath.
    :param str cv_basepath: Where cross-validation files are stored.
                            Includes first part of filename.
    :param int nfolds: Number of cross-validation splits.
    :param str qs_str: String representation of query strategy.
    :param bool model_change: Whether to use model change or not.
    :param dict qs_kwargs: Keyword arguments to pass to query strategy.
    :param str save_profile_to: If not None, where to save tracemalloc snapshot.
    :returns: scores for each CV fold.
    :rtype: list
    '''
    all_scores = []
    choice_orders = []
    for i in range(nfolds):
        if model_change is True:
            qs_kwargs['model_change'] = model_change
        qs = get_query_strategy(qs_str, qs_kwargs)
        if isinstance(qs, UncertaintySampler):
            loss_func = "log"
        else:
            loss_func = "hinge"
        # random_state=i ensures that the algorithm shuffles the data the same
        # way across runs.
        clf_kwargs = {"loss": loss_func, "penalty": "l2",
                      "alpha": 1e-4, "random_state": i}
        if sklearn_ver == "0.19.0":
            clf_kwargs.update({"tol": 1e-3})
        elif sklearn_ver == "0.18.1":
            clf_kwargs.update({"n_iter": 10})
        clf = SGDClassifier(**clf_kwargs)
        # random_state=i ensures that the data is split into L and U sets the
        # the same across runs with the same number of CV splits.
        learner = ActiveLearningModel(clf, qs, random_state=i)
        train_x, test_x, train_y, test_y = read_cv_files(cv_basepath, i + 1)

        if i == 0:
            print("Loss Function: {}".format(loss_func), flush=True)
            print("Train X/y: {}, {}".format(train_x.shape, train_y.shape), flush=True)
            print("Test X/y: {}, {}".format(test_x.shape, test_y.shape), flush=True)

        if save_profile_to is not None:
            snapshot = tracemalloc.take_snapshot()
            outf = "cv_{}.out".format(i)
            snapshot_outfile = os.path.join(save_profile_to, outf)
            snapshot.dump(snapshot_outfile)

        scores, choices = learner.run(train_x, test_x, train_y, test_y, ndraws=ndraws)
        all_scores.append(scores)
        choice_orders.append(choices)
        print('.', end='', flush=True)
    print('', flush=True)
    return all_scores, choice_orders

def check_args(args):
    '''
    Checks that command line arguments are valid.
      - Number of CV folds must be greater than 1.
      - Model change is not compatible with random, minmax, margin,
          simple_margin, or id query strategies.
      - Query strategy keyword arguments must be of the format:
            key1=val1 [[key2=val2] ...]
    :param argparse.Namespace args: Arguments as returned from parse_args().
    '''
    if args.nfolds < 2:
        raise IllegalArgumentError("nfolds must be greater than 1.")
    mc_strategies = ['entropy', 'least_confidence', 'lcb', 'lcb2']

    if args.model_change is True:
        warnings.warn("""Deprecation Warning: the --model_change option is deprecated. Use --qs_kwargs model_change=True instead.""")
        if args.query_strategy not in mc_strategies:
            raise IllegalArgumentError("""Model change only compatible with entropy, least_confidence, lcb, and lcb2""")

    if not all([re.match(r'\w+=[\w\.\']+', arg) for arg in args.qs_kwargs]):
        raise IllegalArgumentError("--qs_kwargs must be of the format arg1=val22 [[arg2=val2] ...]")

    if args.profile is True:
        if args.save_profile_to is None:
            raise IllegalArgumentError("--profile and --save_profile_to must be set together.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_strategy", action="store",
                        choices = ["random", "margin", "simple_margin",
                                   "entropy", "least_confidence", "lcb",
                                   "lcb2", "d2c", "minmax", "density",
                                   "combined"],
                        help="Query strategy.")
    parser.add_argument("trainfile", type=str, help="Training data filename")
    parser.add_argument("--qs_kwargs", type=str, nargs='+', default=[],
                        help="""Kwargs to pass to query strategy.
                                Format: arg1=val22 [[arg2=val2] ...]""")
    parser.add_argument("-i", "--iteration", type=int, default=1,
                        help="""Iteration. Used for multiple runs.
                                (default: 1)""")
    parser.add_argument("-n", "--nfolds", type=int, default=10,
                        help="""Number of cross-validation splits.
                                n > 1. (default: 10)""")
    parser.add_argument("-p", "--percentage", type=int, default=10,
                        help="Top percentage of features to keep. default=10")
    parser.add_argument("--save_ranked_features", type=str, default=None,
                        help="If not None, save ranked features to outfile.")
    parser.add_argument("--ndraws", type=int, default=None,
                        help="""Number of update draws. If None,
                                 ndraws = size of unlabelled set
                                 (default: None)""")
    parser.add_argument("--model_change", action="store_true", default=False,
                        help="""Use model change. Only compatible
                                with least_confidence, lcb, and lcb2.""")
    parser.add_argument("--resultsdir", type=str, default="results", 
                        help="""Base results directory to use.
                                (default: results/)""")
    parser.add_argument("--cvdir", type=str, default="cv_splits",
                        help="""Where to store/read CV files.""")
    parser.add_argument("--score_threshold", type=float, default=0.80,
                        help="Target score for the active learner to acheive.")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="""Whether to save memory profile information. If
                                set, --save_profile_to must also be set.""") 
    parser.add_argument("--save_profile_to", type=str, default=None,
                        help="""Directory in which to save memory profile
                                information. Must be set if --profile is set.""")
    args = parser.parse_args()
    check_args(args)
    return args

def main():
    args = parse_args()

    if args.profile is True:
        safe_makedirs(args.save_profile_to)
        tracemalloc.start()

    # Path format: cv_splits/nfolds/[1...iterations]/[1.csv ... nfolds.csv]
    datadir = os.path.join(args.cvdir, "{0:02d}/{1:02d}"
                                         .format(args.nfolds, args.iteration))
    # Path format: results/qa_strategy/nfolds/[1...iterations]/[<col>.txt]
    query_strategy_str = args.query_strategy
    if args.model_change is True:
        query_strategy_str += "_mc"
    if args.qs_kwargs != []:
        kwargs_str = '_'.join(args.qs_kwargs)
        query_strategy_str += "_{}".format(kwargs_str).replace('=', '-')
    resultsdir = os.path.join(args.resultsdir, "{0}/{1:02d}/{2:02d}"
                                                .format(query_strategy_str,
                                                        args.nfolds,
                                                        args.iteration))
    # If results exist, abort.
    if os.path.exists(resultsdir):
        raise OSError("Results directory '{}' exists. Delete it or use --resultsdir option."
                      .format(resultsdir))
    else:
        safe_makedirs(resultsdir)

    # Create CV data files for this run only if they do not already exist.
    # Column to keep from features file. E.g. column='indicator_type=VERB'
    column="all"
    if not os.path.exists(datadir):
        print("Creating CV data files at {}.".format(datadir), flush=True)
        X, y, feature_names = select_features(args.trainfile, column,
                                              args.percentage,
                                              args.save_ranked_features)
        # Split the data according to n-fold CV and save the splits.
        dirname = os.path.join(datadir, column)
        make_cv_files(X, y, feature_names, dirname, nfolds=args.nfolds,
                      random_state=args.iteration)

    if args.profile is True:
        snapshot = tracemalloc.take_snapshot()
        snapshot_outfile = os.path.join(args.save_profile_to, "main_1.out")
        snapshot.dump(snapshot_outfile)

    # Run the active learner.
    print("Running model: '{}'".format(query_strategy_str), flush=True)
    print("Using column: '{}'".format(column), flush=True)
    print("Iteration {} ".format(args.iteration), flush=True)
    cv_basepath = os.path.join(datadir, column)
    # all.scores.shape = (nfolds, ndraws)
    qs_kwargs = process_qs_kwargs(args.qs_kwargs)
    all_scores, choice_orders = run_active_learner(cv_basepath, args.nfolds,
                                                   args.ndraws, args.query_strategy,
                                                   args.model_change, qs_kwargs,
                                                   save_profile_to=args.save_profile_to)

    if args.profile is True:
        snapshot = tracemalloc.take_snapshot()
        snapshot_outfile = os.path.join(args.save_profile_to, "main_2.out")
        snapshot.dump(snapshot_outfile)

    # In case the number of examples is not divisible by nfolds,
    min_length = np.min([len(scores) for scores in all_scores])
    all_scores = [scores[:min_length] for scores in all_scores]
    # Compute the mean score over CV folds.
    avg_scores = np.mean(np.array(all_scores), axis=0)

    # Find when the learner reached target performance.
    try:
        first = np.where(avg_scores >= args.score_threshold)[0][0]
    except IndexError:
        first = "NEVER"
    print("Acheived {0} AUC at iteration {1}"
           .format(args.score_threshold, first),
          flush=True)

    # Save results
    avg_results_outfile = os.path.join(resultsdir, "avg.txt")
    print("Writing average scores to {}"
           .format(avg_results_outfile),
          flush=True)
    with open(avg_results_outfile, 'w') as outF:
        for score in avg_scores:
            outF.write("{}\n".format(score))

    cv_results_dir = os.path.join(resultsdir, "cv_results")
    safe_makedirs(cv_results_dir)
    print("Writing CV fold scores to {}/"
           .format(cv_results_dir), 
          flush=True)
    for (cvfold, results) in enumerate(all_scores):
        cv_results_outfile = os.path.join(cv_results_dir,
                                          "{0:02d}.txt".format(cvfold + 1))
        with open(cv_results_outfile, 'w') as cv_outF:
            for result in results:
                cv_outF.write("{}\n".format(result))  

    print("Writing choice orders to {}/"
           .format(cv_results_dir), 
          flush=True)
    for (cvfold, order) in enumerate(choice_orders):
        choice_order_outfile = os.path.join(cv_results_dir,
                                          "order-{0:02d}.p".format(cvfold + 1))
        pickle.dump(order, open(choice_order_outfile, "wb"))
    
    if args.profile is True:
        snapshot = tracemalloc.take_snapshot()
        snapshot_outfile = os.path.join(args.save_profile_to, "main_3.out")
        snapshot.dump(snapshot_outfile)


if __name__ == '__main__':
    main()
