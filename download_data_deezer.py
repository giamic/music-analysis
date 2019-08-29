import json
import os
from urllib.request import urlretrieve
import logging

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_files(query, folder, counter, composer):
    response = json.loads(requests.get(query).text)
    next_query = response['next'] if 'next' in response else None
    for album in response['data']:
        # if composer != album['artist']['name']:
        #     continue
        logger.info("Working on album {}, {} previews processed so far".format(album['title'], counter))
        tracklist_query = album['tracklist']
        while tracklist_query is not None:
            tracklist = json.loads(requests.get(tracklist_query).text)
            for t in tracklist['data']:
                local_path = os.path.join(folder, str(counter).zfill(6) + '_' + t['title'].replace('/', '_'))
                urlretrieve(t['preview'], local_path)
                counter += 1
            tracklist_query = tracklist['next'] if 'next' in tracklist else None
    return next_query, counter


n_loops = 1000
limit = 100000
composers = ['Johann Sebastian Bach', 'Ludwig van Beethoven', 'Frederic Chopin', 'Gabriel Faure', 'Robert Schumann',
             'Franz Schubert', 'Wolfgang Amadeus Mozart', 'Franz Liszt', 'Claude Debussy', 'Maurice Ravel',
             'Domenico Scarlatti', 'Dmitri Shostakovich', 'Sergei Prokofiev', 'Sergei Rachmaninov']
# output_folders = [os.path.join('..', 'data', 'deezer_previews', str(i).zfill(3) + '_' + c) for i, c in
#                   enumerate(composers)]
output_folders = [os.path.join(os.path.abspath(os.sep), 'media', 'giamic', 'Local Disk', 'music-analysis', 'data', 'deezer_previews', str(i).zfill(3) + '_' + c) for i, c in
                  enumerate(composers)]
for of in output_folders:
    try:
        os.makedirs(of)
    except FileExistsError:
        # logger.warning("Couldn't create the folders")
        pass

for c, of in zip(composers, output_folders):
    logger.info("Working on {}".format(c))
    logger.info("Output folder is {}".format(of))
    counter, loop = 0, 0
    query = 'https://api.deezer.com/search/album?q=artist:"{}"'.format(c)
    while query is not None and loop < n_loops:
        # query = 'https://api.deezer.com/search/track?q=artist:"{}"&index={}'.format(c, counter)
        logger.info("Asking query {}, so far {} elements processed, loop {}".format(query, counter, loop))
        query, counter = download_files(query, of, counter, c)
        loop += 1
        if query is None:
            logger.warning("The next query is empty!")

# print(response.content)
