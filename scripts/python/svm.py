#! /usr/bin/env python
"""
Implements an SVM classification model as well as functions for extracting
features form the SemMedDb CSV file.
"""

import time
import argparse

import numpy as np
import pandas as pd

from sklearn.svm import SVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import LabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif, chi2


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
        print("{0}(): {1:2f} sec"
               .format(method.__name__, te - ts))
        return result
    return timed

def read_data(csv_file):
    data = pd.read_csv(csv_file, compression='infer')
    y = data["T/F"].astype(np.int64)
    X = data.drop("T/F", axis=1)
    return X, y

@timeit
def select_features(X, y, keep_percentage, features_outfile=None):
    feature_names = X.columns
    feature_finder = SelectPercentile(chi2, percentile=keep_percentage)
    X = feature_finder.fit_transform(X, y)
    support = feature_finder.get_support()
    feature_names = feature_names[support]
    if features_outfile is not None:
        scores = feature_finder.scores_
        pvals = feature_finder.pvalues_
        feature_scores = scores[support]
        feature_pvals = pvals[support]
        feature_data = zip(feature_names, feature_scores, feature_pvals)
        # Sort features by score.
        ranked = sorted(feature_data, key=lambda x: x[1], reverse=True)
        with open(features_outfile, 'w') as outF:
            for feat in ranked:
                outF.write("{} :: {:g} :: {:g}\n".format(*feat))
        print("Saved features to {}".format(features_outfile))
    return X, feature_names

@timeit
def run_classifier(X, y, classifier_str, kernel, loss_func, grid_search=False):
    if grid_search is True:
        print("Grid Search")
        if classifier_str == "svc":
            params = {"C": [0.75, 0.77, 0.8, 0.85, 0.87],
                      "kernel": ["linear"]}
            est = SVC()
        if classifier_str == "nusvc":
            params = {"nu": [0.5, 0.6, 0.7, 0.8, 0.9],
                      "kernel": ["linear", "rbf"]}
            est = NuSVC()
        if classifier_str == "sgd":
            params = {"tol": [1e-3], "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]}
            est = SGDClassifier()
        clf = GridSearchCV(est, params, scoring='roc_auc', cv=5, n_jobs=5)
        clf.fit(X, y)
        print("Best AUC: {:g}".format(clf.best_score_))
        print("Best params: {}".format(clf.best_params_))
    else:
        print("Running classifier.")
        if classifier_str == "nusvc":
            clf = NuSVC(nu=0.5, probability=False, kernel=kernel)
        if classifier_str == "sgd":
            clf = SGDClassifier(loss=loss_func, penalty="l2", tol=1e-3, alpha=0.0001, random_state=8)
        if classifier_str == "svc":
            clf = SVC(kernel=kernel)
        folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        scores = cross_val_score(clf, X, y, cv=folds, scoring='roc_auc')
        clf = clf.fit(X, y)
        print("Num iter: {}".format(clf.n_iter_))
        print("AUC: {:g} (+/- {:g})".format(scores.mean(), scores.std() * 2))

def get_semmed_features(semmed_X, features, y):
    print("Extracting features '{}'".format(features))
    if "all" in features:
        X = semmed_X
    else:
        # Use DataFrame to preserve feature names.
        X = pd.DataFrame(index=range(semmed_X.shape[0]))
        if "cui_feature" in features:
            cui_X = semmed_X.filter(regex=(r'(SU|O)BJECT_CUI=.*'))
            X = pd.concat([X, cui_X], axis=1)
        if "cui2_feature" in features:
            raise NotImplementedError("cui2_feature")
            cui2_feature = pd.DataFrame(semmed_data["SUBJECT_CUI"].str.cat(semmed_data["OBJECT_CUI"], sep='_')).to_dict('records')
            cui2_feature = DictVectorizer(sparse=False).fit_transform(cui2_feature)
            feature_finder = SelectPercentile(chi2, percentile=10)
            cui2_feature = feature_finder.fit_transform(cui2_feature, y)
            X = np.hstack((X, cui2_feature))
        if "dist_feature" in features:
            dist_X = semmed_X.filter(regex=(r'(SU|O)BJECT_DIST'))
            X = pd.concat([X, dist_X], axis=1)
        if "dist2_feature" in features:
            raise NotImplementedError("dist2_feature")
            dist2_feature = np.abs(semmed_data["SUBJECT_START_INDEX"] - semmed_data["OBJECT_START_INDEX"]).values.reshape(-1, 1)
            dist2_feature = OneHotEncoder(sparse=False).fit_transform(dist2_feature)
            feature_finder = SelectPercentile(chi2, percentile=10)
            dist2_feature = feature_finder.fit_transform(dist2_feature, y)
            X = np.hstack((X, dist2_feature))
        if "pred_feature" in features:
            pred_X = semmed_X.filter(regex=(r'PREDICATE=.*'))
            X = pd.concat([X, pred_X], axis=1)
        if "ind_feature" in features:
            ind_X = semmed_X.filter(regex=(r'INDICATOR_TYPE=.*'))
            X = pd.concat([X, ind_X], axis=1)
        if "novelty_feature" in features:
            nov_X = semmed_X.filter(regex=(r'(SU|O)BJECT_NOVELTY'))
            X = pd.concat([X, nov_X], axis=1)
        if "novelty2_feature" in features:
            raise NotImplementedError("novelty2_feature")
            nov2_feature = (semmed_data["SUBJECT_NOVELTY"] + semmed_data["OBJECT_NOVELTY"]).values.reshape(-1, 1)
            X = np.hstack((X, nov2_feature))
    print(X.shape)
    return X
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semmeddb_csv", type=str, required=True,
                        help="CSV file containing SemmedDB features.")
    parser.add_argument("--tfidf_csv", type=str, required=True,
                        help="CSV file containing Tf-idf features.")
    parser.add_argument("--run_semmed", action="store_true", default=False,
                        help="Run SVM with SemmedDB features.")
    parser.add_argument("--run_tfidf", action="store_true", default=False,
                        help="Run SVM with Tf-idf features.")
    parser.add_argument("--run_comb", action="store_true", default=False,
                        help="""Run SVM with SemmedDB and Tf-idf features.
                                Specify SemmedDB feature with --feature.""")
    parser.add_argument("--run_features_only", action="store_true", default=False,
                        help="Run classifier with only features specified by --features")
    parser.add_argument("--classifier", type=str, choices=["svc", "nusvc", "sgd"],
                        default="svc", help="Classifier to use.")
    parser.add_argument("--kernel", type=str, choices=["linear", "rbf", "poly"],
                        default="linear", help="SVC kernel to use.")
    parser.add_argument("--loss_func", type=str, choices=["hinge", "log"],
                        default="hinge", help="SGD loss function.")
    parser.add_argument("--grid_search", action="store_true", default=False,
                        help="Run grid search to optimize parameters.")
    parser.add_argument("--features", type=str, default="all", nargs="*",
                        choices=["len_feature", "cui_feature", "cui2_feature", "cui3_feature",
                                 "dist_feature", "dist2_feature", "pred_feature",
                                 "ind_feature", "novelty_feature",
                                 "novelty2_feature", "all"],
                        help="Features from SemmedDB to use.")
    parser.add_argument("--filter_all_features", action="store_true", default=False,
                        help="Retain only top 30 percent of features specified by --feature.")
    parser.add_argument("--semmed_keep_percentage", type=int, default=10,
                        help="Keep top n percent of SemmedDB features. default: 10")
    parser.add_argument("--tfidf_keep_percentage", type=int, default=10,
                        help="Keep top n percent of Tf-idf features. default: 10")
    parser.add_argument("--comb_keep_percentage", type=int, default=10,
                        help="Keep top n percent of combined features. default: 10")
    parser.add_argument("--semmed_features_outfile", type=str, default=None,
                        help="If not None, save SemMedDB features to outfile.")
    parser.add_argument("--tfidf_features_outfile", type=str, default=None,
                        help="If not None, save tf-idf features to outfile.")
    parser.add_argument("--comb_features_outfile", type=str, default=None,
                        help="If not None, save combination features to outfile.")
    parser.add_argument("--save_comb_X", type=str, default=None,
                        help="If not None, write combined dataset to outfile.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    print("Classifier: {}".format(args.classifier))
    print("Kernel: {}".format(args.kernel if args.classifier == "svc" else "linear"))
    print("Loss: {}".format(args.loss_func))

    # SemMedDB
    semmed_X = None
    if args.run_semmed is True:
        print("Reading SemmedDB Features")
        semmed_X, semmed_y = read_data(args.semmeddb_csv)
        print(semmed_X.shape)
        semmed_X, _ = select_features(semmed_X, semmed_y,
                                   args.semmed_keep_percentage,
                                   features_outfile=args.semmed_features_outfile)
        print(semmed_X.shape)
        run_classifier(semmed_X, semmed_y, args.classifier,
                       args.kernel, args.loss_func,
                       grid_search=args.grid_search)
        print()
    
    # Tf-idf
    tfidf_X = None
    if args.run_tfidf is True:
        print("Reading tf-idf Features")
        tfidf_X, tfidf_y = read_data(args.tfidf_csv)
        print(tfidf_X.shape)
        tfidf_X, tfidf_feats = select_features(tfidf_X, tfidf_y,
                                                args.tfidf_keep_percentage,
                                                features_outfile=args.tfidf_features_outfile)
        print(tfidf_X.shape)
        run_classifier(tfidf_X, tfidf_y, args.classifier,
                       args.kernel, args.loss_func,
                       grid_search=args.grid_search)
        print()

    # Combination
    if args.run_comb is True:
        if semmed_X is None:
            semmed_X, semmed_y = read_data(args.semmeddb_csv)
        if tfidf_X is None:
            print("Reading tf-idf Features")
            tfidf_X, tfidf_y = read_data(args.tfidf_csv)
            print(tfidf_X.shape)
            tfidf_X, tfidf_feats = select_features(tfidf_X, tfidf_y, args.tfidf_keep_percentage,
                                                   features_outfile=args.tfidf_features_outfile)
            print(tfidf_X.shape)
        print("Getting SemmedDB Features")
        comb_y = tfidf_y
        new_semmed_X = get_semmed_features(semmed_X, args.features, comb_y)
        print(new_semmed_X.shape)
        new_semmed_X, semmed_feats = select_features(new_semmed_X, comb_y,
                                                     args.semmed_keep_percentage,
                                                     features_outfile=args.semmed_features_outfile)
        print(new_semmed_X.shape)
        print("{} + {}".format(tfidf_X.shape, new_semmed_X.shape))
        comb_X = np.hstack((tfidf_X, new_semmed_X))
        cols = list(tfidf_feats) + list(semmed_feats)
        comb_X_df = pd.DataFrame(data=comb_X, columns=cols)
        comb_X, comb_feats = select_features(comb_X_df, comb_y,
                                             args.comb_keep_percentage,
                                             features_outfile=args.comb_features_outfile)
        print(comb_X.shape)
        run_classifier(comb_X, comb_y, args.classifier,
                       args.kernel, args.loss_func,
                       grid_search=args.grid_search)
        if args.save_comb_X is not None:
            header = ['"{}"'.format(f) for f in comb_feats]
            header.append('"T/F"')
            outrows = np.hstack((comb_X, comb_y.values.reshape(-1,1)))
            header = ','.join(header)
            np.savetxt(args.save_comb_X, outrows, delimiter=',',
                       header=header, comments='')

    print("===")
        

if __name__ == '__main__':
    main()
