from subprocess import call

import numpy as np
import os

data_folder = '/home/gianluca/PycharmProjects/music-analysis/data/dataset_audiolabs_crosscomposer/chroma_features/by_song'
output_folder = '/home/gianluca/PycharmProjects/music-analysis/data/dataset_audiolabs_crosscomposer/test/chroma_features/by_song'
# fp = sorted([os.path.join(data_folder, p) for p in os.listdir(data_folder)])

prefix = 'chroma-nnls_CrossComp-'
suffix = '.csv'

np.random.seed(18)
for n in range(11):
    low, high = 100 * n + 1, 100 * (n + 1) + 1
    ids = np.random.choice(np.arange(low, high), 10, replace=False)
    ids = [str(i).zfill(4) for i in ids]
    paths = [os.path.join(data_folder, prefix + i + suffix) for i in ids]
    destinations = [os.path.join(output_folder, prefix + i + suffix) for i in ids]
    for p, d in zip(paths, destinations):
        call(["mv", p, d])

# Had to change something because songs 0439, 0566, 0884, 0892 don't have 10 excerpts inside.
# I replaced them with 0442, 0574, 0880, 0889
