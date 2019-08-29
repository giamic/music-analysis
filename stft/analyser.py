import os
import random

import librosa
import numpy as np
from matplotlib import cm
from matplotlib.image import imsave
from scipy.signal import spectrogram, stft

from config_general import SPOTIFY_FOLDER
from stft.config import SR, FRAME_SIZE, N_FFT, FREQUENCY_CAP

np.random.seed(18)


def analyse_single_file(audio_file, mode='stft_abs'):
    samples, sample_rate = librosa.load(audio_file, sr=SR)

    if len(samples.shape) == 2:
        samples = np.average(samples, axis=1)  # magically transform the file to mono
    # frequency resolution = SR * window_main_lobe_size / FRAME_SIZE = 44100 Hz * 4 / 2001 = 88 Hz
    # time resolution on single frame = FRAME_SIZE / SR = 2001 / 44100 Hz = 0.045 s = 45 ms
    # time distance between frames = HOP_SIZE / SR = FRAME_SIZE / (2*SR) = 22.5 ms
    if mode == 'stft_abs':
        f, t, v = np.abs(stft(samples, sample_rate, window='hamming', nperseg=FRAME_SIZE, nfft=N_FFT))
    elif mode == 'stft_log':
        f, t, v = np.abs(stft(samples, sample_rate, window='hamming', nperseg=FRAME_SIZE, nfft=N_FFT))
        v = np.log10(v + 1e-3)
    elif mode == 'spectrogram':
        f, t, v = spectrogram(samples, sample_rate, window='hamming', nperseg=FRAME_SIZE, nfft=N_FFT)
    else:
        raise ValueError("Specify either 'stft_abs', 'stft_log', or 'spectrogram' as mode.")
    return f, t, v


def cap_frequency(f, t, v):
    f_end = np.searchsorted(f, FREQUENCY_CAP)
    return f[:f_end], t, v[:f_end]


def _check_files_already_analysed(images_folder):
    files = os.listdir(images_folder)
    if len(files) == 0:
        return -1, -1
    last_file = sorted(files)[-1]
    last_file = last_file.split("_")
    artist, track = last_file[1], last_file[2]
    return artist, track


def analyse_data(recordings_folder, images_folder):
    """
    :param recordings_folder:
    :param images_folder:
    :return:
    """
    artists = sorted(os.listdir(recordings_folder))
    last_artist, last_track = _check_files_already_analysed(images_folder)
    for artist in artists:
        tracks = sorted(os.listdir(os.path.join(recordings_folder, artist)))
        print(f"Working on {artist}, total of {len(tracks)} tracks")
        artist_id = artist.split("_")[0]
        if artist_id < last_artist:
            print(f"The data for {artist} has already been analysed. Skipping")
            continue
        if artist_id == last_artist:
            tracks = tracks[int(last_track):]
            print(f"Some of the data for {artist} has already been analysed. Skipping first {last_track} tracks.")
        for track in tracks:
            track_name = ''.join(track.split(".")[:-1])
            track_id = track.split("_")[0]
            if random.randint(0, 99) == 0:
                print("track number {}".format(track_id))
            audio_file = os.path.join(recordings_folder, artist, track)
            f, t, v = analyse_single_file(audio_file)
            f, t, v = cap_frequency(f, t, v)
            # The maximum length of the file name path accepted by Ubuntu 19.04 is 255
            output_file = os.path.join(images_folder, 'CRS_{}_{}'.format(artist_id, track_name))[:251] + ".png"
            imsave(output_file, v, cmap=cm.get_cmap("Greys_r"), origin='lower')
    return


if __name__ == '__main__':
    images_folder = os.path.join(SPOTIFY_FOLDER, 'images')
    os.makedirs(images_folder, exist_ok=True)
    analyse_data(os.path.join(SPOTIFY_FOLDER, 'recordings'), images_folder)
