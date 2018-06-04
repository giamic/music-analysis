# DISCLAIMER : Taken from https://github.com/annazhukova/music/blob/master/py/matrix_formatter.py
import os
import sys
from subprocess import call

import numpy as np
import pandas as pd
import tensorflow as tf


def _create_annotations(data_folder, output_folder, ids, times):
    annotations = pd.read_csv(os.path.join(data_folder, 'cross-composer_annotations.csv'))
    names = ["index", "CrossComp-ID", "ClipTime"]
    df = pd.DataFrame(dict(zip(names, [np.arange(len(ids)), ids, times])))
    res = df.merge(annotations, on="CrossComp-ID")
    res = res.sort_values("index")
    res = res[['Composer', 'CrossComp-ID', 'ClipTime', 'CompLifetime', 'SongYear']]
    res.to_csv(os.path.join(output_folder, 'metadata.tab'), sep='\t', header=True)
    return


def _reconstruct_tree(DATA_DIR):
    DM = os.path.join(DATA_DIR, 'dm.txt')
    MATRIX = os.path.join(DATA_DIR, 'matrix.phy')
    TREE = os.path.join(DATA_DIR, 'tree.nwk')

    # Create a matrix with numbers as ids
    df = pd.read_table(DM, header=None, sep=' ')
    with open(MATRIX, 'w+') as f:
        f.write('{}\n'.format(len(df)))
    df.to_csv(MATRIX, sep='\t', header=False, mode='a')

    # Reconstruct a tree from the matrix
    call(["docker", "run", "-v", "{}:/data".format(DATA_DIR), "-t", "evolbioinfo/fastme:v2.1.6.1",
          "-i", "/data/{}".format(os.path.split(MATRIX)[1]), "-o", "/data/{}".format(os.path.split(TREE)[1])])
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


def tree_analysis(dm, ids, times, data_folder, output_folder):
    ids = list(map(lambda b: tf.compat.as_text(b), ids))

    try:
        os.mkdir(output_folder)
    except FileExistsError:
        pass
    np.savetxt(os.path.join(output_folder, 'dm.txt'), dm)
    _create_annotations(data_folder, output_folder, ids, times)
    _reconstruct_tree(output_folder)
    _visualize_tree(output_folder)
    return


if __name__ == '__main__':
    _reconstruct_tree(sys.argv[1])
    _visualize_tree(sys.argv[1])
