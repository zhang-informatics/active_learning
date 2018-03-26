#! /usr/bin/env python
"""
An interactive annotation script for SemMedDB data. Requires the following fields:
    SUBJECT_TEXT
    OBJECT_TEXT
    SUBJECT_SEMTYPE
    OBJECT_SEMTYPE
    PREDICATE
    INDICATOR_TYPE
    TYPE
    SENTENCE

Given these fields, prints the sentence with the subject and object highlighted.
User inputs 'y' (yes), 'n' (no), or 's' (skip) as the annotation labels.
"""


import sys
import os
import re
import time
import curses
import argparse
import pandas as pd


def display_row(row, semmed_ver, stdscr):
    if semmed_ver == "30":
        s_type_key = "SUBJECT_SEMTYPE"
        o_type_key = "OBJECT_SEMTYPE"
        pred_key = "PREDICATE"
    else:
        s_type_key = "s_type"
        o_type_key = "o_type"
        pred_key = "predicate"
    row = row[1]
    sentence = ' '.join(row["SENTENCE"].split())
    try:
        subj = ' '.join(row["SUBJECT_TEXT"].split())
    except AttributeError:
        stdscr.addstr(sentence)
        stdscr.addstr(2, 0, "Couldn't split subject '{}'".format(row["SUBJECT_TEXT"]))
        return
    try:
        obj = ' '.join(row["OBJECT_TEXT"].split())
    except AttributeError:
        stdscr.addstr(sentence)
        stdscr.addstr(2, 0, "Couldn't split object '{}'".format(row["OBJECT_TEXT"]))
        return

    try:
        subj_start_idx = sentence.index(subj)
        subj_end_idx = subj_start_idx + len(subj)
    except ValueError as e:
        stdscr.addstr(sentence)
        stdscr.addstr(2, 0, "Failed to find subject '{}' in the sentence. Object is '{}'. Predicate is '{}'."
                             .format(subj, obj, row[pred_key]))
        return
    try:
        obj_start_idx = sentence.index(obj)
        obj_end_idx = obj_start_idx + len(obj)
    except ValueError as e:
        stdscr.addstr(sentence)
        stdscr.addstr(2, 0, "Failed to find object '{}' in the sentence. Subject is '{}'. Predicate is '{}'."
                             .format(obj, subj, row[pred_key]))
        return
    if subj_start_idx < obj_start_idx:
        first_s, first_e = subj_start_idx, subj_end_idx
        second_s, second_e = obj_start_idx, obj_end_idx
    else:
        first_s, first_e = obj_start_idx, obj_end_idx
        second_s, second_e = subj_start_idx, subj_end_idx

    addstr_args = [[sentence[:first_s]],
                   [sentence[first_s:first_e], curses.A_STANDOUT],  # First word
                   [sentence[first_e:second_s]],
                   [sentence[second_s:second_e], curses.A_STANDOUT], # Second word
                   [sentence[second_e:]]
                  ]
    for args in addstr_args:
        stdscr.addstr(*args)

    y, x = stdscr.getyx()
    template = "{0} ({1}) ** {2} ({3}) ** {4} ({5})"
    stdscr.addstr(y + 2, 0, template.format(subj,
                                            row[s_type_key],
                                            row[pred_key],
                                            row["INDICATOR_TYPE"],
                                            obj,
                                            row[o_type_key]))
    stdscr.addstr(y + 3, 0, "Title/Abstract: {}".format(row["TYPE"]))
    if "Label" in row:
        stdscr.addstr(y + 4, 0, "Label: {}".format(row["Label"]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvfile", type=str, help="CSV file to annotate")
    parser.add_argument("annotation_file", type=str, help="Annotation file")
    parser.add_argument("--sep", type=str, default=",",
                        help="CSV field delimiter (default: ',')")
    parser.add_argument("--semmed_ver", type=str, choices=["30", "26"],
                        default="26", help="SemmedDB version. (default: 26)")
    args = parser.parse_args()
    return args

def main(stdscr, args):
    csv_data = pd.read_csv(args.csvfile, sep=args.sep)

    num_anns = 0
    if os.path.isfile(args.annotation_file):
        num_anns = len(open(args.annotation_file, 'r').readlines())

    stdscr = curses.initscr()
    stdscr.keypad(True)
    curses.echo()
    
    try:
        for (i, row) in enumerate(csv_data.iterrows()):
            if i <= num_anns:
                continue
            stdscr.clear()
            display_row(row, args.semmed_ver, stdscr)
            ann = ''
            while ann not in ['y', 'n', 's']:
                y, x = stdscr.getyx()
                stdscr.addstr(y + 2, 0, "y/n/(s)kip: ")
                stdscr.refresh()
                y, x = stdscr.getyx()
                ann = stdscr.getstr(y, x, 1).decode(encoding="utf-8").lower()
                if ann not in ['y', 'n', 's']:
                    stdscr.addstr(y + 1, 0, "Input must be 'y', 'n', or 's' (skip)")
            with open(args.annotation_file, 'a') as annF:
                annF.write("{}\n".format(ann))
    except KeyboardInterrupt:
        y, x = stdscr.getyx() 
        stdscr.addstr(y + 1, x, "Bye!")
        stdscr.refresh()
        time.sleep(1)
        return


if __name__ == '__main__':
    args = parse_args()
    curses.wrapper(main, args)
