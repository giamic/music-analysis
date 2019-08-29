# DISCLAIMER : rewritten from https://github.com/annazhukova/music/blob/master/py/matrix_formatter.py
import os
from subprocess import call

import numpy as np
import pandas as pd


def _create_dm_file(dm, matrix_fp):
    df = pd.DataFrame(dm)
    with open(matrix_fp, 'w+') as f:
        f.write('{}\n'.format(len(df)))
    df.to_csv(matrix_fp, sep='\t', header=False, mode='a')
    return matrix_fp


def _create_metadata(ids, times, annotations_file, md_file):
    names = ["index", "CrossComp-ID", "ClipTime"]
    # np.savetxt(os.path.join(output_folder, 'labels.txt'), ids, fmt='%s')
    annotations = pd.read_csv(annotations_file, header=0)
    df = pd.DataFrame(dict(zip(names, [np.arange(len(ids)), ids, times])))
    res = df.merge(annotations, on="CrossComp-ID").sort_values("index").set_index("index")
    res = res[['Composer', 'CrossComp-ID', 'ClipTime', 'CompBirth', 'CompDeath', 'SongYear']]
    res.to_csv(md_file, sep='\t', header=True)
    return md_file


def _reconstruct_tree(matrix, tree, info=None):
    work_dir, matrix_filename = os.path.split(matrix)
    tree_filename = 'tree.nwk'
    info_filename = 'fastME.info'

    # Reconstruct a tree from the matrix
    call(["docker", "run", "-v", "{}:/data".format(work_dir), "-t", "evolbioinfo/fastme:v2.1.6.1",
          "-i", "/data/{}".format(matrix_filename), "-o", "/data/{}".format(tree_filename),
          "-I", "/data/{}".format(info_filename)])
    call(['mv', '{}/{}'.format(work_dir, tree_filename), tree])
    if info:
        call(['mv', '{}/{}'.format(work_dir, info_filename), info])
    return tree, info


def _date_tree(tree, annotations_file, dated_tree_file=None):
    dates = '{}.dates.tab'.format(tree)
    df = pd.read_csv(annotations_file, header=0)
    df.index = df['CrossComp-ID'].map(lambda x: x[10:])
    df = df[['SongYear', 'CompBirth', 'CompDeath']]
    df['date'] = df.apply(lambda row: row['SongYear'] if not pd.isnull(row['SongYear']) \
        else 'b({},{})'.format(row['CompBirth'], row['CompDeath']), axis=1)
    with open(dates, 'w+') as f:
        f.write('{}\n'.format(len(df)))
    df['date'].to_csv(dates, header=False, index=True, mode='a', sep=' ')
    call(["lsd", "-i", tree, "-d", dates, "-v", "2", "-c", "-r", "a"])
    if dated_tree_file:
        call(['mv', '{}.result.date.newick'.format(tree), dated_tree_file])
    else:
        dated_tree_file = '{}.result.date.newick'.format(tree)
    return dated_tree_file


def _visualize_tree(tree, metadata, html, map):
    work_dir, tree_filename = os.path.split(tree)
    call(['cp', metadata, '{}/{}.metadata.tab'.format(work_dir, tree_filename)])
    html_filename = 'tree.html'
    map_filename = 'map.html'
    # Visualise a tree with pastml such as every song has its own colour (names are numerical ids)
    call(["docker", "run", "-v", "{}:/data".format(work_dir), "-t", "evolbioinfo/pastml",
          "--tree", "/data/{}".format(tree_filename),
          "--data", "/data/{}".format('{}.metadata.tab'.format(tree_filename)),
          "--html", "/data/{}".format(html_filename),
          "--html_compressed", "/data/{}".format(map_filename), "--columns", "Composer", "-v",
          "--tip_size_threshold", '-1', '--model', 'JC'])
    call(['rm', '{}/{}.metadata.tab'.format(work_dir, tree_filename)])
    call(['mv', '{}/{}'.format(work_dir, html_filename), html])
    call(['mv', '{}/{}'.format(work_dir, map_filename), map])
    return html, map


def tree_analysis(dm, ids, times, annotations_file, output_folder):
    ids = list(map(lambda x: 'CrossComp-' + str(x[0]).zfill(4), ids))
    times = list(map(lambda x: x[0], times))
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        pass

    matrix_file = os.path.join(output_folder, 'matrix.phy')
    _create_dm_file(dm, matrix_file)
    metadata_file = os.path.join(output_folder, 'metadata.tab')
    _create_metadata(ids, times, annotations_file, metadata_file)
    tree_file = os.path.join(output_folder, 'tree.nwk')
    info_file = os.path.join(output_folder, 'fastME.info')
    _reconstruct_tree(matrix_file, tree_file, info_file)
    dated_tree_file = os.path.join(output_folder, 'tree.dated.nwk')
    _date_tree(tree_file, annotations_file, dated_tree_file)
    html = os.path.join(output_folder, 'tree.html')
    html_compressed = os.path.join(output_folder, 'map.html')
    _visualize_tree(dated_tree_file, metadata_file, html, html_compressed)
    return


if __name__ == '__main__':
    data_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data',
                               'dataset_audiolabs_crosscomposer')
    annotations = os.path.join(data_folder, 'cross-composer_annotations.csv')
    output_folder = '/home/gianluca/PycharmProjects/music-analysis/models/match_5cl_pool_sigm_2018-07-03_18-35-52/test/2018-07-03_22-08-51'
    tree_file = os.path.join(output_folder, 'tree.nwk')
    metadata_file = os.path.join(output_folder, 'metadata.tab')
    dated_tree_file = os.path.join(output_folder, 'tree.dated.nwk')
    _date_tree(tree_file, annotations, dated_tree_file)
    html = os.path.join(output_folder, 'tree.html')
    map = os.path.join(output_folder, 'map.html')
    _visualize_tree(dated_tree_file, metadata_file, html, map)
