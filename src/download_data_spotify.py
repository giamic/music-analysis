import json
import os
from urllib.request import urlretrieve
import logging

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_files(query, folder, counter):
    response = json.loads(requests.get(query).text)
    next_query = response['next'] if 'next' in response else None
    for album in response['items']:
        logger.info("Working on album {}, {} previews processed so far".format(album['title'], counter))
        tracklist_query = 'https://api.spotify.com/v1/albums/{}/tracks'.format(album['id'])
        while tracklist_query is not None:
            tracklist = json.loads(requests.get(tracklist_query).text)
            for t in tracklist['items']:
                local_path = os.path.join(folder, str(counter).zfill(6) + '_' + t['name'].replace('/', '_'))
                urlretrieve(t['preview_url'], local_path)
                counter += 1
            tracklist_query = tracklist['next'] if 'next' in tracklist else None
    return next_query, counter


n_loops = 1000
limit = 100000
c2id = {
    'Johann Sebastian Bach': '5aIqB5nVVvmFsvSdExz408',
    'Ludwig van Beethoven': '2wOqMjp9TyABvtHdOSOTUS',
    'Frederic Chopin': '7y97mc3bZRFXzT2szRM4L4',
    'Gabriel Faure': '2gClsBep1tt1rv1CN210SO',
    'Robert Schumann': '2UqjDAXnDxejEyE0CzfUrZ',
    'Franz Schubert': '2p0UyoPfYfI76PCStuXfOP',
    'Wolfgang Amadeus Mozart': '4NJhFmfw43RLBLjQvxDuRS',
    'Franz Liszt': '1385hLNbrnbCJGokfH2ac2',
    'Claude Debussy': '1Uff91EOsvd99rtAupatMP',
    'Maurice Ravel': '17hR0sYHpx7VYTMRfFUOmY',
    'Domenico Scarlatti': '0mFblCBw0GcoY7zY1P8tzE',
    'Dmitri Shostakovich': '6s1pCNXcbdtQJlsnM1hRIA',
    'Sergei Prokofiev': '4kHtgiRnpmFIV5Tm4BIs8l',
    'Sergei Rachmaninov': '0Kekt6CKSo0m5mivKcoH51'
}

composers = sorted(c2id.keys())
output_folders = [
    os.path.join(os.path.abspath(os.sep), 'media', 'giamic', 'Local Disk', 'music-analysis', 'data', 'spotify_previews',
                 str(i).zfill(3) + '_' + c) for i, c in enumerate(composers)]
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

    query = 'https://api.spotify.com/v1/artists/{}/albums'.format(c2id[c])
    while query is not None and loop < n_loops:
        # query = 'https://api.deezer.com/search/track?q=artist:"{}"&index={}'.format(c, counter)
        logger.info("Asking query {}, so far {} elements processed, loop {}".format(query, counter, loop))
        query, counter = download_files(query, of, counter)
        loop += 1
        if query is None:
            logger.warning("The next query is empty!")

# print(response.content)
