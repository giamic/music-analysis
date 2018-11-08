import os
import random

import librosa
import numpy as np
from matplotlib import cm
from matplotlib.image import imsave
from scipy.signal import spectrogram, stft

from stft.config import SR, FRAME_SIZE, N_FFT, CLIP_TIME, FREQUENCY_CAP

np.random.seed(18)


def analyse_single_file(audio_file, mode='stft_abs', nperseg=FRAME_SIZE, nfft=N_FFT):
    samples, sample_rate = librosa.load(audio_file, sr=SR)

    # loader = essentia.standard.MonoLoader(audio_file)
    # audio = loader()
    if len(samples.shape) == 2:
        samples = np.average(samples, axis=1)  # magically transform the file to mono
    # frequency resolution = fs * window_main_lobe_size / nperseg = 44100 Hz * 4 / 2001 = 88 Hz
    # time resolution = nperseg / fs = 2001 / 44100 Hz = 0.045 s = 45 ms
    if mode == 'stft_abs':
        f, t, v = np.abs(stft(samples, sample_rate, window='hamming', nperseg=nperseg, nfft=nfft))
    elif mode == 'stft_log':
        f, t, v = np.abs(stft(samples, sample_rate, window='hamming', nperseg=nperseg, nfft=nfft))
        v = np.log10(v + 1e-3)
    elif mode == 'spectrogram':
        f, t, v = spectrogram(samples, sample_rate, window='hamming', nperseg=nperseg, nfft=nfft)
    else:
        raise ValueError("Specify either 'stft_abs', 'stft_log', or 'spectrogram' as mode.")
    return f, t, v


def save_clips(f, t, v):
    clip_size = int(CLIP_TIME / (t[1] - t[0]))
    n_clips = int(t[-1] / CLIP_TIME)
    for _ in range(n_clips):
        clip_start = np.random.randint(0, len(t) - clip_size)
        clip_end = clip_start + clip_size
        output_file = os.path.join(images_folder, 'CRS_{}_{}_{}_{}.png'.format(i, j, k, clip_start))
        s = v[:, clip_start:clip_end]
        imsave(output_file, s, cmap=cm.get_cmap("Greys_r"), origin='lower')
    return


def cap_frequency(f, t, v):
    f_end = np.searchsorted(f, FREQUENCY_CAP)
    return f[:f_end], t, v[:f_end]


def analyse_data(data_folder, images_folder, mode='previews'):
    """
    Call the mode "previews" if working with Spotify or Deezer previews.
    Call the mode "tracks" if working with tracks ripped from a cd
    :param data_folder:
    :param images_folder:
    :param mode:
    :return:
    """
    artists = sorted(os.listdir(data_folder))
    if mode == 'test':
        audio_file = 'audio.flac'
        track, t, v = analyse_single_file(audio_file)
        track, t, v = cap_frequency(track, t, v)
        clip_size = int(CLIP_TIME / (t[1] - t[0]))
        n_clips = int(t[-1] / CLIP_TIME)
        for k in range(n_clips):
            clip_start = np.random.randint(0, len(t) - clip_size)
            clip_end = clip_start + clip_size
            output_file = os.path.join('test4_{}.png'.format(clip_start))
            s = v[:, clip_start:clip_end]
            imsave(output_file, s, cmap=cm.get_cmap("Greys_r"), origin='lower')
    elif mode == 'tracks':
        for i, artist in enumerate(artists):
            print("Working on {}".format(artist))
            recordings = sorted(os.listdir(os.path.join(data_folder, artist)))
            for j, r in enumerate(recordings):
                tracks = sorted(os.listdir(os.path.join(data_folder, artist, r)))
                tracks = [t for t in tracks if t[-4:] == 'flac']
                for k, track in enumerate(tracks):
                    audio_file = os.path.join(data_folder, artist, r, track)
                    f, t, v = analyse_single_file(audio_file)
                    f, t, v = cap_frequency(f, t, v)
                    clip_size = int(CLIP_TIME / (t[1] - t[0]))
                    n_clips = int(t[-1] / CLIP_TIME)
                    for _ in range(n_clips):
                        clip_start = np.random.randint(0, len(t) - clip_size)
                        clip_end = clip_start + clip_size
                        output_file = os.path.join(images_folder, 'CRS_{}_{}_{}_{}.png'.format(i, j, k, clip_start))
                        s = v[:, clip_start:clip_end]
                        imsave(output_file, s, cmap=cm.get_cmap("Greys_r"), origin='lower')
    elif mode == 'previews':
        for artist in artists:
            tracks = sorted(os.listdir(os.path.join(data_folder, artist)))
            print("Working on {}, total of {} tracks".format(artist, len(tracks)))
            for track in tracks:
                artist_id = artist.split("_")[0]
                track_name = ''.join(track.split(".")[:-1])
                track_id = track.split("_")[0]
                if random.randint(0, 99) == 0:
                    print("track number {}".format(track_id))
                audio_file = os.path.join(data_folder, artist, track)
                f, t, v = analyse_single_file(audio_file)
                f, t, v = cap_frequency(f, t, v)
                output_file = os.path.join(images_folder, 'CRS_{}_{}.png'.format(artist_id, track_name))
                imsave(output_file, v, cmap=cm.get_cmap("Greys_r"), origin='lower')
    else:
        raise ValueError("please specify either 'test', 'tracks', or 'previews' as mode")
    return


if __name__ == '__main__':
    # data_folder = os.path.join('..', '..', 'data', 'recordings')
    # images_folder = os.path.join('..', '..', 'data', 'images')
    data_folder = os.path.join(os.path.abspath(os.sep), 'media', 'giamic', 'Local Disk', 'music_analysis', 'data',
                               'spotify_previews', 'recordings')
    images_folder = os.path.join(data_folder, '..', 'images')
    mode = 'previews'
    analyse_data(data_folder, images_folder, mode=mode)
