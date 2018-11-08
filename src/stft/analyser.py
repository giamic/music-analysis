import os
import random

import librosa
import numpy as np
from matplotlib import cm
from matplotlib.image import imsave
from scipy.signal import spectrogram, stft

from stft.config import SR, FRAME_SIZE, N_FFT, FREQUENCY_CAP, EXTERNAL_DATA_FOLDER

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


def analyse_data(recordings_folder, images_folder):
    """
    :param recordings_folder:
    :param images_folder:
    :return:
    """
    artists = sorted(os.listdir(recordings_folder))
    for artist in artists:
        tracks = sorted(os.listdir(os.path.join(recordings_folder, artist)))
        print("Working on {}, total of {} tracks".format(artist, len(tracks)))
        for track in tracks:
            artist_id = artist.split("_")[0]
            track_name = ''.join(track.split(".")[:-1])
            track_id = track.split("_")[0]
            if random.randint(0, 99) == 0:
                print("track number {}".format(track_id))
            audio_file = os.path.join(recordings_folder, artist, track)
            f, t, v = analyse_single_file(audio_file)
            f, t, v = cap_frequency(f, t, v)
            output_file = os.path.join(images_folder, 'CRS_{}_{}.png'.format(artist_id, track_name))
            imsave(output_file, v, cmap=cm.get_cmap("Greys_r"), origin='lower')
    return


if __name__ == '__main__':
    analyse_data(os.path.join(EXTERNAL_DATA_FOLDER, 'recordings'), os.path.join(EXTERNAL_DATA_FOLDER, 'images'))
