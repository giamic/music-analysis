# DISCLAIMER : rewritten from https://github.com/annazhukova/music/blob/master/py/matrix_formatter.py
import os
from subprocess import call

import numpy as np
import pandas as pd

from stft.config import COMPOSERS_DATA, MODELS_FOLDER


def _create_dm_file(dm, matrix_fp):
    df = pd.DataFrame(dm)
    with open(matrix_fp, 'w+') as f:
        f.write('{}\n'.format(len(df)))
    df.to_csv(matrix_fp, sep='\t', header=False, mode='a')
    return matrix_fp


def _create_metadata(excerpt_ids, md_file):
    names = ["index", "ExcerptID", "CompID", "SongYear"]
    comp_ids = np.array([i.split('_')[0] for i in excerpt_ids], dtype=np.int64)
    df = pd.DataFrame(dict(zip(names, [np.arange(len(excerpt_ids)), excerpt_ids, comp_ids, [None] * len(excerpt_ids)])))
    comp_dates = pd.read_csv(COMPOSERS_DATA, header=0)
    res = df.merge(comp_dates, on="CompID").sort_values("index").set_index("index")
    res = res[['ExcerptID', 'LastName', 'FirstName', 'BirthYear', 'DeathYear', 'SongYear']]
    # res = df  # TODO: re-implement the CompBirth, CompDeath, SongYear
    res.to_csv(md_file, sep='\t', header=True)
    return md_file


def _reconstruct_tree(matrix, tree, info=None, improvements=False):
    work_dir, matrix_filename = os.path.split(matrix)
    tree_filename = 'tree.nwk'
    info_filename = 'fastME.info'

    # Reconstruct a tree from the matrix
    if improvements:
        call(["docker", "run",
              "-v", "{}:/data".format(work_dir),  # set working directory
              "-t", "evolbioinfo/fastme:v2.1.6.1",  # choose tool to use
              "-i", "/data/{}".format(matrix_filename),  # input
              "-o", "/data/{}".format(tree_filename),  # output
              "-I", "/data/{}".format(info_filename),  # log file
              "-n", "-s"])  # tree topology improvements (NNI and SPR)
    else:
        call(["docker", "run",
              "-v", "{}:/data".format(work_dir),  # set working directory
              "-t", "evolbioinfo/fastme:v2.1.6.1",  # choose tool to use
              "-i", "/data/{}".format(matrix_filename),  # input
              "-o", "/data/{}".format(tree_filename),  # output
              "-I", "/data/{}".format(info_filename)])  # log file
    call(['mv', '{}/{}'.format(work_dir, tree_filename), tree])
    if info:
        call(['mv', '{}/{}'.format(work_dir, info_filename), info])
    return tree, info


def _date_tree(tree, metadata_file, dated_tree_file=None):
    dates = '{}.dates.tab'.format(tree)
    df = pd.read_csv(metadata_file, sep='\t', header=0, index_col=0)
    df = df[['SongYear', 'BirthYear', 'DeathYear']]
    df['date'] = df.apply(lambda row: row['SongYear']
    if not pd.isnull(row['SongYear'])
    else 'b({},{})'.format(row['BirthYear'], row['DeathYear']),
                          axis=1)
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
          "--html_compressed", "/data/{}".format(map_filename), "--columns", "LastName", "-v",
          "--tip_size_threshold", '-1', '--model', 'JC'])
    call(['rm', '{}/{}.metadata.tab'.format(work_dir, tree_filename)])
    call(['mv', '{}/{}'.format(work_dir, html_filename), html])
    call(['mv', '{}/{}'.format(work_dir, map_filename), map])
    return html, map


def tree_analysis(dm, comp_ids, song_ids, output_folder, run_analysis=True):
    excerpt_ids = [str(c[0]) + '_' + str(s[0]) for c, s in zip(comp_ids, song_ids)]
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        pass

    matrix_file = os.path.join(output_folder, 'matrix.phy')
    _create_dm_file(dm, matrix_file)
    metadata_file = os.path.join(output_folder, 'metadata.tab')
    _create_metadata(excerpt_ids, metadata_file)
    if run_analysis:
        tree_file = os.path.join(output_folder, 'tree.nwk')
        info_file = os.path.join(output_folder, 'fastME.info')
        _reconstruct_tree(matrix_file, tree_file, info_file)
        dated_tree_file = os.path.join(output_folder, 'tree.dated.nwk')
        _date_tree(tree_file, metadata_file, dated_tree_file)
        html = os.path.join(output_folder, 'tree.html')
        html_compressed = os.path.join(output_folder, 'map.html')
        _visualize_tree(tree_file, metadata_file, html, html_compressed)
    return


def _create_itol_annotations(metadata_file):
    work_dir, metadata_filename = os.path.split(metadata_file)
    call(["docker", "run", "-v", "{}:/data".format(work_dir), "-t", "evolbioinfo/table2itol",
          "-i", "index",
          "/data/{}".format(metadata_filename),
          "-D", "/data/"])
    return


if __name__ == '__main__':
    output_folder = os.path.join(MODELS_FOLDER, 'extract_3c2_rnn_bn_pool_sigmoid_2018-11-19_17-40-15', 'test',
                                 '2018-11-19_18-43-44')
    matrix_file = os.path.join(output_folder, 'matrix.phy')
    metadata_file = os.path.join(output_folder, 'metadata.tab')
    tree_file = os.path.join(output_folder, 'tree.nwk')
    info_file = os.path.join(output_folder, 'fastME.info')
    dated_tree_file = os.path.join(output_folder, 'tree.dated.nwk')
    html = os.path.join(output_folder, 'tree.html')
    map = os.path.join(output_folder, 'map.html')

    _reconstruct_tree(matrix_file, tree_file, info_file)
    # _date_tree(tree_file, metadata_file, dated_tree_file)
    # _visualize_tree(tree_file, metadata_file, html, map)
    _create_itol_annotations(metadata_file)

# /home/gianluca/PycharmProjects/music-analysis/models/match_5cl_pool_sigm_2018-07-03_18-35-52/test/2018-07-03_22-08-51/tree.dated.nwk.metadata.tab
# /data//home/gianluca/PycharmProjects/music-analysis/models/match_5cl_pool_sigm_2018-07-03_18-35-52/test/2018-07-03_22-08-51/tree.dated.nwk.metadata.tab
