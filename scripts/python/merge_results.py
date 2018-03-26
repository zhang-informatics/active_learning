#! /usr/bin/env python
"""
Averages scores over multiple iterations of al.py.
"""

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('resultsdir', type=str,
                    help="Directory of results iterations. E.g. 'results/random/05/'")
parser.add_argument('outfile', type=str, help="Path to output file.")
parser.add_argument('filename', type=str,
                    help="Results filename. E.g. 'avg.txt'")
parser.add_argument('--ignore_missing', action='store_true', default=False,
                    help="Ignore incomplete iterations.")
args = parser.parse_args()

dirs = os.listdir(args.resultsdir)
avg_scores = []
for directory in dirs:
    path = os.path.join(args.resultsdir, directory, args.filename)
    try:
        lines = open(path, 'r').readlines()
    except FileNotFoundError as err:
        if args.ignore_missing is True:
            continue
        else:
            raise err
    scores = np.array([float(s.strip()) for s in lines])
    avg_scores.append(scores)
avg_scores = np.array(avg_scores)
avg_scores = np.mean(avg_scores, axis=0)

with open(args.outfile, 'w') as outF:
    for score in avg_scores:
        outF.write("{0}\n".format(score))
