import csv
import json
import logging
import os
from urllib.request import urlretrieve

from oauthlib.oauth2 import BackendApplicationClient
from requests.auth import HTTPBasicAuth
from requests_oauthlib import OAuth2Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CREDENTIALS_FILE = os.path.join('..', 'data', 'spotify_credentials.csv')
with open(CREDENTIALS_FILE) as f:
    csvfile = csv.reader(f)
    _, client_id = next(csvfile)
    _, client_secret = next(csvfile)

auth = HTTPBasicAuth(client_id, client_secret)
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)
token = oauth.fetch_token(token_url='https://accounts.spotify.com/api/token', auth=auth)


def create_filename(track_name, counter):
    track_name = track_name. \
        replace('/', ''). \
        replace(';', ''). \
        replace(':', ''). \
        replace('*', ''). \
        replace('=', ''). \
        replace('(', ''). \
        replace(')', ''). \
        replace('[', ''). \
        replace(']', ''). \
        replace('?', ''). \
        replace('!', ''). \
        replace(' ', '_'). \
        replace("'", '-'). \
        replace("`", '-'). \
        replace('"', '-'). \
        replace('&', 'and')

    name = str(counter).zfill(6) + '_' + track_name + '.mp3'
    return name


def download_files(query, folder, counter):
    response = json.loads(oauth.get(query).text)
    next_query = response['next'] if 'next' in response else None
    for album in response['items']:
        logger.info("Working on album {}, {} previews processed so far".format(album['name'], counter))
        tracklist_query = 'https://api.spotify.com/v1/albums/{}/tracks'.format(album['id'])
        while tracklist_query is not None:
            tracklist = json.loads(oauth.get(tracklist_query).text)
            for t in tracklist['items']:
                if t['preview_url'] is not None:
                    file_name = create_filename(t['name'], counter)
                    local_path = os.path.join(folder, file_name)
                    urlretrieve(t['preview_url'], local_path)
                    counter += 1
            tracklist_query = tracklist['next'] if 'next' in tracklist else None
    return next_query, counter


n_loops = 1000
n_previews = 3000
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

composers = ['Claude Debussy']
# composers = sorted(c2id.keys())
output_folders = [
    os.path.join(os.path.abspath(os.sep), 'media', 'giamic', 'Local Disk', 'music_analysis', 'data', 'spotify_previews',
                 'recordings', str(i).zfill(3) + '_' + c) for i, c in enumerate(composers)]
for of in output_folders:
    try:
        os.makedirs(of)
    except FileExistsError:
        pass

for c, of in zip(composers, output_folders):
    logger.info("Working on {}".format(c))
    logger.info("Output folder is {}".format(of))
    counter = 0

    query = 'https://api.spotify.com/v1/artists/{}/albums'.format(c2id[c])
    while query is not None and counter < n_previews:
        logger.info("Asking query {}, so far {} elements processed".format(query, counter))
        query, counter = download_files(query, of, counter)
        if query is None:
            logger.warning("The next query is empty!")

# print(response.content)
