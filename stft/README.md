This package takes a series of audio previews, extract its STFT, and analyzes it with a machine learning model to
predict the composer of the song.

1. Run analyser.py to obtain the STFT (three available modes: absolute value, logarithm, and spectrogram, i.e., absolute value squared)
2. Run data_serialization.py to transform the data into tfrecords, faster to use.
