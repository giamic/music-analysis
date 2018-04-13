"""
IDEA FOR THE ALGORITHM:
  - put labels on all the chroma features depending on the ID of the song
  - read the chroma features, all numerical columns
  - create a convolution neural network
  - apply triplet loss (thanks to the amazing guy that posted it on github)
  - train
"""

import tensorflow as tf
import numpy as np

from data_loading import train_input_fn, test_input_fn
from model import cnn_model_fn

music_analysis = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir="../models/model1/"
    )

music_analysis.train(input_fn=train_input_fn, steps=2_000)

music_analysis.evaluate(input_fn=test_input_fn)


