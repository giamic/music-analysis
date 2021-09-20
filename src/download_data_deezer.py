import json
import logging
import os
from datetime import datetime
from urllib.request import urlretrieve

import requests

from src import AUDIO_FOLDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_files(query, folder, counter):
    response = json.loads(requests.get(query).text)
    next_query = response["next"] if "next" in response else None
    for album in response["data"]:
        # FIXME: Some tracks are from the wrong composer. We need better filters
        if "various" in album["artist"]["name"].lower():
            continue
        logger.info(f"Downloading {album['title']}, {counter} previews obtained so far")
        tracklist_query = album["tracklist"]
        while tracklist_query is not None and counter < info["tracks_per_composer"]:
            tracklist = json.loads(requests.get(tracklist_query).text)
            for t in tracklist["data"]:
                filename = str(counter).zfill(6) + "_" + t["title"].replace("/", "_")
                local_path = os.path.join(folder, filename)
                # FIXME: We might download the same track twice, coming from two
                #  different albums. Check if it is possible to avoid it
                urlretrieve(t["preview"], local_path)
                counter += 1
            tracklist_query = tracklist["next"] if "next" in tracklist else None
    return next_query, counter


ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_folder = os.path.join(AUDIO_FOLDER, f"deezer_previews_{ts}")
os.makedirs(base_folder)
info = {
    "tracks_per_composer": 1_000,
    "composers": [
        "Johann Sebastian Bach",
        "Ludwig van Beethoven",
        "Frederic Chopin",
        "Gabriel Faure",
        "Robert Schumann",
        "Franz Schubert",
        "Wolfgang Amadeus Mozart",
        "Franz Liszt",
        "Claude Debussy",
        "Maurice Ravel",
        "Domenico Scarlatti",
        "Dmitri Shostakovich",
        "Sergei Prokofiev",
        "Sergei Rachmaninov",
    ],
}
with open(os.path.join(base_folder, "config.info"), "w") as f:
    json.dump(info, f)

for i, c in enumerate(info["composers"]):
    of = os.path.join(base_folder, str(i).zfill(3) + "_" + c)
    os.makedirs(of)
    logger.info(f"Working on {c}")
    logger.info(f"Output folder is {of}")
    counter, loop = 0, 0
    query = f'https://api.deezer.com/search/album?q=artist:"{c}"'
    while query is not None and counter < info["tracks_per_composer"]:
        logger.info(f"{query}, {counter} previews downloaded, loop {loop+1}")
        # TODO: Understand better how these query works.
        #  Apparently, they really do get different files when I run the loop again
        #  but I don't really get why
        query, counter = download_files(query, of, counter)
        loop += 1
        if query is None:
            logger.warning("The next query is empty!")
