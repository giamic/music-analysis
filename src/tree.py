# DISCLAIMER : rewritten from https://github.com/annazhukova/music/blob/master/py/matrix_formatter.py
import os
import sys
from subprocess import call

import numpy as np
import pandas as pd


def _create_dm_file(dm, output_folder):
    matrix_fp = os.path.join(output_folder, 'matrix.phy')
    df = pd.DataFrame(dm)
    with open(matrix_fp, 'w+') as f:
        f.write('{}\n'.format(len(df)))
    df.to_csv(matrix_fp, sep='\t', header=False, mode='a')
    return


def _create_metadata(ids, times, annotations, output_folder):
    names = ["index", "CrossComp-ID", "ClipTime"]
    # np.savetxt(os.path.join(output_folder, 'labels.txt'), ids, fmt='%s')
    df = pd.DataFrame(dict(zip(names, [np.arange(len(ids)), ids, times])))
    res = df.merge(annotations, on="CrossComp-ID").sort_values("index").set_index("index")
    res = res[['Composer', 'CrossComp-ID', 'ClipTime', 'CompBirth', 'CompDeath', 'SongYear']]
    res.to_csv(os.path.join(output_folder, 'metadata.tab'), sep='\t', header=True)
    return


def _reconstruct_tree(DATA_DIR):
    MATRIX = os.path.join(DATA_DIR, 'matrix.phy')
    TREE = os.path.join(DATA_DIR, 'tree.nwk')
    INFO = os.path.join(DATA_DIR, 'fastME.info')

    # Reconstruct a tree from the matrix
    call(["docker", "run", "-v", "{}:/data".format(DATA_DIR), "-t", "evolbioinfo/fastme:v2.1.6.1",
          "-i", "/data/{}".format(os.path.split(MATRIX)[1]), "-o", "/data/{}".format(os.path.split(TREE)[1]),
          "-I", "/data/{}".format(os.path.split(INFO)[1])])
    return


def _visualize_tree(DATA_DIR):
    ANNOTATIONS = os.path.join(DATA_DIR, 'metadata.tab')
    TREE = os.path.join(DATA_DIR, 'tree.nwk')
    HTML = os.path.join(DATA_DIR, 'tree.html')
    MAP = os.path.join(DATA_DIR, 'map.html')

    # Visualise a tree with pastml such as every song has its own colour (names are numerical ids)
    call(["docker", "run", "-v", "{}:/data".format(DATA_DIR), "-t", "evolbioinfo/pastml",
          "--tree", "/data/{}".format(os.path.split(TREE)[1]),
          "--data", "/data/{}".format(os.path.split(ANNOTATIONS)[1]),
          "--html", "/data/{}".format(os.path.split(HTML)[1]),
          "--html_compressed", "/data/{}".format(os.path.split(MAP)[1]), "--columns", "Composer", "-v",
          "--tip_size_threshold", '30', '--model', 'JC'])
    return


def tree_analysis(dm, ids, times, annotations, output_folder):
    ids = list(map(lambda x: 'CrossComp-' + str(x[0]).zfill(4), ids))
    times = list(map(lambda x: x[0], times))
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        pass

    _create_dm_file(dm, output_folder)
    _create_metadata(ids, times, annotations, output_folder)
    _reconstruct_tree(output_folder)
    _visualize_tree(output_folder)
    return


if __name__ == '__main__':
    _reconstruct_tree(sys.argv[1])
    _visualize_tree(sys.argv[1])
