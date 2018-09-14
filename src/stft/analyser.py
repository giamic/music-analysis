import os

import numpy as np
import soundfile as sf
from matplotlib import cm
from matplotlib.image import imsave
from scipy.signal import spectrogram, stft

data_folder = os.path.join('..', '..', 'data', 'recordings')
images_folder = os.path.join('..', '..', 'data', 'images')
artists = sorted(os.listdir(data_folder))
# recordings = [sorted(os.listdir(af)) for af in [os.path.join(data_folder, a) for a in artists]]
clip_time = 20
frequency_cap = 5000
np.random.seed(18)

test = False

if test:
    audio_file = 'audio.flac'
    samples, sample_rate = sf.read(audio_file)
    samples = np.average(samples, axis=1)  # magically transform the file to mono
    # frequencies, times, values = spectrogram(samples, sample_rate, window='blackman', nperseg=1000, nfft=2048)
    f1, t1, v1 = np.abs(stft(samples, sample_rate, window='hamming', nperseg=2001, nfft=2048))
    f2, t2, v2 = np.abs(stft(samples, sample_rate, window='hamming', nperseg=2001, nfft=2048))
    v2 = v2 ** 2
    f3, t3, v3 = spectrogram(samples, sample_rate, window='hamming', nperseg=2001, nfft=2048)
    f4, t4, v4 = np.abs(stft(samples, sample_rate, window='hamming', nperseg=2001, nfft=2048))
    v4 = np.log10(v4 + 1e-3)
    f_end = np.searchsorted(f1, frequency_cap)
    clip_size = int(clip_time / (t1[1] - t1[0]))
    n_clips = int(t1[-1] / clip_time)
    for k in range(n_clips):
        clip_start = np.random.randint(0, len(t1) - clip_size)
        clip_end = clip_start + clip_size
        output_file = os.path.join('test4_{}.png'.format(clip_start))
        s = v4[:f_end, clip_start:clip_end]
        imsave(output_file, s, cmap=cm.get_cmap("Greys_r"), origin='lower')
else:
    for i, a in enumerate(artists):
        print("Working on {}".format(a))
        recordings = sorted(os.listdir(os.path.join(data_folder, a)))
        for j, r in enumerate(recordings):
            audio_files = sorted(os.listdir(os.path.join(data_folder, a, r)))
            audio_files = [a for a in audio_files if a[-4:] == 'flac']
            for k, f in enumerate(audio_files):
                audio_file = os.path.join(data_folder, a, r, f)
                samples, sample_rate = sf.read(audio_file)
                samples = np.average(samples, axis=1)  # magically transform the file to mono
                f, t, v = np.abs(stft(samples, sample_rate, window='hamming', nperseg=2001, nfft=2048))
                # frequency resolution = fs * window_main_lobe_size / nperseg = 44100 Hz * 4 / 2001 = 88 Hz
                # time resolution = nperseg / fs = 2001 / 44100 Hz = 0.045 s = 45 ms
                f_end = np.searchsorted(f, frequency_cap)
                clip_size = int(clip_time / (t[1] - t[0]))
                n_clips = int(t[-1] / clip_time)
                for _ in range(n_clips):
                    clip_start = np.random.randint(0, len(t) - clip_size)
                    clip_end = clip_start + clip_size
                    output_file = os.path.join(images_folder, 'CRS_{}_{}_{}_{}.png'.format(i, j, k, clip_start))
                    s = v[:f_end, clip_start:clip_end]
                    imsave(output_file, s, cmap=cm.get_cmap("Greys_r"), origin='lower')
