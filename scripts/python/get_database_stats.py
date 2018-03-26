#! /usr/bin/env python
"""
Given an annotated SemMedDb CSV file, prints various statistics.
"""

import argparse
import pandas as pd
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="SemMedDB CSV file to process.")
    parser.add_argument("ds_cuis_file", type=str, help="File containing DS CUIs.")
    parser.add_argument("--print_outdata", action="store_true", default="False",
                        help="Print Python dicts representing data in tables.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    data = pd.read_csv(args.infile, sep='\t', compression='infer')
    ds_cuis = [l.strip() for l in open(args.ds_cuis_file).readlines()]
    all_cuis = set(data.SUBJECT_CUI).union(set(data.OBJECT_CUI))
    print("Number of Unique CUIs: {}".format(len(all_cuis)))
    print("Number of DS CUIs: {}".format(len([cui for cui in all_cuis if cui in ds_cuis])))

    # ======= PREDICATE ==========
    pred_data = {}
    for pred in data["PREDICATE"].unique():
        pred_data[pred] = data[data["PREDICATE"]==pred]
    pred_data = dict(sorted(pred_data.items(), key=lambda x: x[1].shape[0], reverse=True))

    h_template = "| {0:<15} || {1:<15} | {2:<15} |"
    header = ["PREDICATE", "# PREDICATIONS", "+ CLASS (%)"]
    header_str = h_template.format(*header)
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    val_template = "| {0:<15} || {1:<15} | {2:<15.0f} |"
    outdata = {}
    for (pred, df) in pred_data.items():
        num_preds = df.shape[0]
        percent_pos = ((df["T/F"]=='y').sum() / num_preds) * 100
        outdata[pred] = {'size': num_preds, 'class_dist': (percent_pos, 1-percent_pos)}
        print(val_template.format(pred, num_preds, percent_pos))
    print("-" * len(header_str))
    if args.print_outdata is True:
        print(outdata)
    print()

    template = "| {0:<15} || {1:<15} | {2:<15} | {3:<8} | {4:<8} |"
    header = ["PREDICATE", "Subj CUI is DS", "Obj CUI is DS", "ANY", "ALL"]
    header_str = template.format(*header)
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    outdata = {}
    for (pred, df) in pred_data.items():
        subj_ds, obj_ds = df[["SUBJECT_CUI", "OBJECT_CUI"]].isin(ds_cuis).sum(axis=0).values
        anys = df[["SUBJECT_CUI", "OBJECT_CUI"]].isin(ds_cuis).any(axis=1).sum()
        alls = df[["SUBJECT_CUI", "OBJECT_CUI"]].isin(ds_cuis).all(axis=1).sum()
        outdata[pred] = {"subj_ds": subj_ds, "obj_ds": obj_ds,
                         "any": anys, "all": alls}
        print(template.format(pred, subj_ds, obj_ds, anys, alls))
    print("-" * len(header_str))
    if args.print_outdata is True:
        print(outdata)
    print()

    # ======= INDICATOR TYPE ==========
    it_data = {}
    for it in data["INDICATOR_TYPE"].unique():
        it_data[it] = data[data["INDICATOR_TYPE"]==it]
    it_data = dict(sorted(it_data.items(), key=lambda x: x[1].shape[0], reverse=True))

    h_template = "| {0:<15} || {1:<15} | {2:<15} |"
    header = ["INDICATOR TYPE", "# PREDICATIONS", "+ CLASS (%)"]
    header_str = h_template.format(*header)
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    val_template = "| {0:<15} || {1:<15} | {2:<15.0f} |"
    outdata = {}
    for (it, df) in it_data.items():
        num_preds = df.shape[0]
        percent_pos = ((df["T/F"]=='y').sum() / num_preds) * 100
        outdata[it] = {'size': num_preds, 'class_dist': (percent_pos, 1-percent_pos)}
        print(val_template.format(it, num_preds, percent_pos))
    print("-" * len(header_str))
    if args.print_outdata is True:
        print(outdata)

    print()

    template = "| {0:<15} || {1:<15} | {2:<15} | {3:<8} | {4:<8} |"
    header = ["INDICATOR TYPE", "Subj CUI is DS", "Obj CUI is DS", "ANY", "ALL"]
    header_str = template.format(*header)
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    outdata = {}
    for (it, df) in it_data.items():
        subj_ds, obj_ds = df[["SUBJECT_CUI", "OBJECT_CUI"]].isin(ds_cuis).sum(axis=0).values
        anys = df[["SUBJECT_CUI", "OBJECT_CUI"]].isin(ds_cuis).any(axis=1).sum()
        alls = df[["SUBJECT_CUI", "OBJECT_CUI"]].isin(ds_cuis).all(axis=1).sum()
        outdata[it] = {"subj_ds": subj_ds, "obj_ds": obj_ds,
                         "any": anys, "all": alls}
        print(template.format(it, subj_ds, obj_ds, anys, alls))
    print("-" * len(header_str))
    if args.print_outdata is True:
        print(outdata)

    print()

    template = "| {0:<15} || {1:<15} | {2:<15} | {3:<15} |"
    header = ["INDICATOR TYPE", "INTERACTS_WITH", "STIMULATES", "INHIBITS"]
    header_str = template.format(*header)
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    for (it, df) in it_data.items():
        counts = defaultdict(int, df["PREDICATE"].value_counts())
        outdata[it] = {"INTERACTS_WITH": counts["INTERACTS_WITH"],
                       "STIMULATES": counts["STIMULATES"],
                       "INHIBITS": counts["INHIBITS"]}
        print(template.format(it, counts["INTERACTS_WITH"],
                                  counts["STIMULATES"],
                                  counts["INHIBITS"]))
    print("-" * len(header_str))
    if args.print_outdata is True:
        print(outdata)

if __name__ == "__main__":
    main()
