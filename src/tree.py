# DISCLAIMER : Taken from https://github.com/annazhukova/music/blob/master/py/matrix_formatter.py

import os
import pandas as pd
from subprocess import call


def reconstruct_tree(DATA_DIR):
    DM = os.path.join(DATA_DIR, 'dm.txt')
    LABELS = os.path.join(DATA_DIR, 'labels.txt')
    MATRIX = os.path.join(DATA_DIR, 'matrix.phy')
    ANNOTATIONS = os.path.join(DATA_DIR, 'metadata.tab')
    TREE = os.path.join(DATA_DIR, 'tree.nwk')
    HTML = os.path.join(DATA_DIR, 'tree.html')

    # Create a matrix with numbers as ids
    df = pd.read_table(DM, header=None, sep=' ')
    with open(MATRIX, 'w+') as f:
        f.write('{}\n'.format(len(df)))
    df.to_csv(MATRIX, sep='\t', header=False, mode='a')

    # Reconstruct a tree from the matrix
    call(["docker", "run", "-v", "{}:/data".format(DATA_DIR), "-t", "evolbioinfo/fastme:v2.1.6.1",
          "-i", "/data/{}".format(os.path.split(MATRIX)[1]), "-o", "/data/{}".format(os.path.split(TREE)[1])])

    # Create an annotation file mapping ids to their labels
    df['label'] = pd.read_csv(LABELS, header=None)[0]
    df['time'] = df['label'].str.replace('^CrossEra\-\d+_t=', '')
    df['song'] = df['label'].str.replace('^CrossEra\-', '').str.replace('_t=\d+(\.\d+){0,1}', '')
    df = df[['label', 'song', 'time']]
    df.to_csv(ANNOTATIONS, sep='\t', header=True)

    # Visualise a tree with pastml such as every song has its own colour (names are numerical ids)
    call(["docker", "run", "-v", "{}:/data".format(DATA_DIR), "-t", "evolbioinfo/pastml",
          "--tree", "/data/{}".format(os.path.split(TREE)[1]),
          "--data", "/data/{}".format(os.path.split(ANNOTATIONS)[1]),
          "--html", "/data/{}".format(os.path.split(HTML)[1]), "--columns", "song", "-v"])
    return
