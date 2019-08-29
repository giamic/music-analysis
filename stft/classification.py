"""
IDEA FOR THE ALGORITHM:
  - CNN on the magnitude of the STFT
  - apply triplet loss (thanks to the amazing guy that posted it on github)
  - train
"""

import logging
import os
from datetime import datetime

from tensorflow.python.keras.backend import flatten

from models import classify_3c2_rnn_bn_pool_sigmoid
from stft.config import TRAIN_PATH, VALIDATION_PATH, MODELS_FOLDER, PARAMS
from stft.data_loading import create_tfrecords_dataset
from tensorflow.keras import callbacks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

x_train = create_tfrecords_dataset(TRAIN_PATH, PARAMS['bs'], PARAMS['sb'])
x_valid = create_tfrecords_dataset(VALIDATION_PATH, PARAMS['bs'])
# for i, j in x_train:
#     print(flatten(i), j)

model = classify_3c2_rnn_bn_pool_sigmoid()
model_folder = os.path.join(MODELS_FOLDER, model.name + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

os.makedirs(model_folder, exist_ok=True)
with open(os.path.join(model_folder, 'params.txt'), 'w') as file:
    for (k, v) in PARAMS.items():
        file.write("{}: {}\n".format(k, v))

logits = model.layers[-2].output
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy')
callbacks = [
    callbacks.TensorBoard(log_dir=os.path.join(model_folder, 'logs')),
    callbacks.EarlyStopping(patience=3)
]
model.fit(x_train, steps_per_epoch=PARAMS['steps_train'],
          validation_data=x_valid, validation_steps=PARAMS['steps_validation'],
          callbacks=callbacks)
