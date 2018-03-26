#! /usr/bin/env python
"""
Given a cross validation data directory, as created by al.py, prints
tables of statistics for each CV fold to ensure proper stratification.
"""

import os
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cvdir", type=str, help="E.g. cv_splits/10/01")
    parser.add_argument("nfolds", type=int, help="Number of cv folds.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    template = "| {0:<4} || {1:<15.0f} | {2:<15} | {3:<15} | {4:<15} |"
    h_template = "| {0:<4} || {1:<15} | {2:<15} | {3:<15} | {4:<15} |"
    header = h_template.format("CV", "+ class (%)", "# INTERACTS", "# STIMULATES", "# INHIBITS")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i in range(args.nfolds):
        i += 1
        xfile = "all_train-x_{0:02d}.csv".format(i)
        xpath = os.path.join(args.cvdir, xfile)
        yfile = "all_train-y_{0:02d}.csv".format(i)
        ypath = os.path.join(args.cvdir, yfile)
        X = pd.read_csv(xpath)
        y = pd.read_csv(ypath)
    
        percent_pos = (y["T/F"].sum() / y.shape[0]) * 100
        num_inter = X[X["PREDICATE=INTERACTS_WITH"]==1].shape[0]
        num_stim = X[X["PREDICATE=STIMULATES"]==1].shape[0]
        num_inhib = X[X["PREDICATE=INHIBITS"]==1].shape[0]

        print(template.format(i, percent_pos, num_inter, num_stim, num_inhib))
    print("-" * len(header))


if __name__ == "__main__":
    main()
