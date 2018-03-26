#! /usr/bin/env python
"""
Convert the tab-separated SemMedDB CSV file into feature vectors
readable by pandas, numpy, etc.
"""

import argparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif, chi2


def semmed_features(data, y, keep_percentage=100):
    data = data.drop(["T/F", "SENTENCE"], axis=1)
    y = y.ravel()
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(data.to_dict('records'))
    feature_names = np.array(vectorizer.get_feature_names())
    print(X.shape)
    if keep_percentage < 100:
        feature_finder = SelectPercentile(f_classif, percentile=keep_percentage)
        X = feature_finder.fit_transform(X, y)
        print("After feature selection: {}".format(X.shape))
        support = feature_finder.get_support()
        feature_names = feature_names[support]
    feature_names = ['"{}"'.format(f) for f in feature_names]
    return X, feature_names

def tfidf_features(data, y, keep_percentage=100,
                   ngram_range=(1,1), binary=False):
    sentences = data["SENTENCE"].values
    y = y.ravel()
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words="english",
                                 token_pattern=r'(?u)\b[\w-][\w-]+\b', binary=binary)
    X = vectorizer.fit_transform(sentences).toarray()
    feature_names = np.array(vectorizer.get_feature_names())
    print(X.shape)
    if keep_percentage < 100:
        feature_finder = SelectPercentile(f_classif, percentile=keep_percentage)
        X = feature_finder.fit_transform(X, y)
        print("After feature selection: {}".format(X.shape))
        support = feature_finder.get_support()
        feature_names = feature_names[support]
    feature_names = ['"{}"'.format(f) for f in feature_names]
    return X, feature_names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("incsv", type=str, help="SemmedDB CSV to convert.")
    parser.add_argument("outcsv", type=str, help="Where to save the features.")
    parser.add_argument("--feature_type", type=str, choices=["semmed", "tfidf"],
                        help="Type of features to compute.")
    parser.add_argument("--keep_percentage", type=int, default=100,
                        help="Percentage of top tf-idf features to keep. default 100.")
    parser.add_argument("--tfidf_ngram_range", type=int, nargs=2, default=[1,1],
                        help="TfidfVectorizer ngram_range parameter. default: 1 1 .")
    parser.add_argument("--tfidf_binary_counts", action="store_true", default=False,
                        help="Use binary counts when computing tf-idf.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    semmed_data = pd.read_csv(args.incsv, sep='\t')
    labels = semmed_data["T/F"]
    y = LabelBinarizer().fit_transform(labels)

    if args.feature_type == "semmed":
        print("In shape: {}".format(semmed_data.shape))
        X, feature_names = semmed_features(semmed_data, y,
                                           keep_percentage=args.keep_percentage)
        print("Out shape: {}".format(X.shape))
    elif args.feature_type == "tfidf":
        X, feature_names = tfidf_features(semmed_data, y,
                                          keep_percentage=args.keep_percentage,
                                          ngram_range=args.tfidf_ngram_range,
                                          binary=args.tfidf_binary_counts)

    feature_names.append('"T/F"')
    header = ','.join(feature_names)
    outrows = np.hstack((X, y))
    print("With labels: {}".format(outrows.shape))
    np.savetxt(args.outcsv, outrows, delimiter=',', header=header, comments='') 
    print("Saved.")


if __name__ == "__main__":
    main()
