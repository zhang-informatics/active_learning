#! /usr/bin/env python
"""
Reads tracemalloc snapshots and displays top memory usage.
"""

import os
import argparse
import tracemalloc
import linecache


def display_top(snapshot, key_type="lineno", limit=3, unit="kb"):
    if unit == "kb":
        size = 1000
    elif unit == "mb":
        size = 1000000
    elif unit == "gb":
        size = 1000000000
    else:
        raise ValueError("Unit '{}' not supported.")
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top {} lines".format(limit))
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#{0}: {1}:{2}: {3:.1f} {4}".format(index, filename,
                                                 frame.lineno,
                                                 stat.size / size, unit))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    {}'.format(line))

    other = top_stats[limit:]
    if other:
        size = sum([stat.size for stat in other])
        print("{0} other: {1:.1f} {2}".format(len(other), size / size, unit))
    total = sum([stat.size for stat in top_stats])
    print("Total allocated size: {0:.1f} {1}".format(total / size, unit))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_dir", type=str,
                        help="Directory in which snapshots are stored.")
    parser.add_argument("-u", "--unit", type=str, choices=["kb", "mb", "gb"],
                        default="kb", help="Memory size unit to use.")
    parser.add_argument("--cv_only", action="store_true", default=False,
                        help="Only show profiles for CV runs.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    files = os.listdir(args.snapshot_dir)
    if args.cv_only is True:
        files = [f for f in files if "cv" in f]
    paths = [os.path.join(args.snapshot_dir, f) for f in files]

    snapshots = []
    tm_filters = (tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                  tracemalloc.Filter(False, "<unknown>"))

    for fn in paths:
        print(os.path.basename(fn))
        snapshot = tracemalloc.Snapshot.load(fn)
        snapshots.append(snapshot)
        display_top(snapshot, unit=args.unit)
        print("-" * 10)
        print()
        if len(snapshots) > 1:
            stats = snapshot.filter_traces(tm_filters).compare_to(snapshots[-2], 'filename')
            for stat in stats[:5]:
                print("{} new kb, {} total kb, {} new, {} total memory blocks: "
                      .format(stat.size_diff/1000, stat.size/1000, stat.count_diff, stat.count))
                for line in stat.traceback.format():
                    print(line)
        print()


if __name__ == "__main__":
    main()
