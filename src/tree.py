# DISCLAIMER : Taken from https://github.com/annazhukova/music/blob/master/py/matrix_formatter.py

import os
import sys

import pandas as pd
from subprocess import call


def reconstruct_tree(DATA_DIR):
    DM = os.path.join(DATA_DIR, 'dm.txt')
    LABELS = os.path.join(DATA_DIR, 'labels.txt')
    MATRIX = os.path.join(DATA_DIR, 'matrix.phy')
    ANNOTATIONS = os.path.join(DATA_DIR, 'metadata.tab')
    TREE = os.path.join(DATA_DIR, 'tree.nwk')
    HTML = os.path.join(DATA_DIR, 'tree.html')
    MAP = os.path.join(DATA_DIR, 'map.html')

    # Create a matrix with numbers as ids
    df = pd.read_table(DM, header=None, sep=' ')
    with open(MATRIX, 'w+') as f:
        f.write('{}\n'.format(len(df)))
    df.to_csv(MATRIX, sep='\t', header=False, mode='a')

    # Reconstruct a tree from the matrix
    call(["docker", "run", "-v", "{}:/data".format(DATA_DIR), "-t", "evolbioinfo/fastme:v2.1.6.1",
          "-i", "/data/{}".format(os.path.split(MATRIX)[1]), "-o", "/data/{}".format(os.path.split(TREE)[1])])

    # Create an annotation file mapping ids to their labels
    notes = pd.read_csv(ANNOTATIONS, sep='\t')
    df['composer'] = notes["Composer"]
    # df['label'] = pd.read_csv(LABELS, header=None)[0]
    # df['time'] = df['label'].str.replace('^CrossEra\-\d+_t=', '')
    # df['song'] = df['label'].str.replace('^CrossEra\-', '').str.replace('_t=\d+(\.\d+){0,1}', '')
    # df['composer'] = df['label'].map(lambda _: _[:_.find('CrossEra')])
    # df = df[['label', 'song', 'time', 'composer']]
    # df.to_csv(ANNOTATIONS, sep='\t', header=True)

    # Visualise a tree with pastml such as every song has its own colour (names are numerical ids)
    call(["docker", "run", "-v", "{}:/data".format(DATA_DIR), "-t", "evolbioinfo/pastml",
          "--tree", "/data/{}".format(os.path.split(TREE)[1]),
          "--data", "/data/{}".format(os.path.split(ANNOTATIONS)[1]),
          "--html", "/data/{}".format(os.path.split(HTML)[1]),
          "--html_compressed", "/data/{}".format(os.path.split(MAP)[1]), "--columns", "Composer", "-v",
          "--tip_size_threshold", '30', '--model', 'JC'])
    return


if __name__ == '__main__':
    reconstruct_tree(sys.argv[1])
