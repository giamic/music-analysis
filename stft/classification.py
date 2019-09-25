"""
IDEA FOR THE ALGORITHM:
  - CNN on the magnitude of the STFT
  - apply triplet loss (thanks to the amazing guy that posted it on github)
  - train
"""

import logging
import os
import time
from datetime import datetime

from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping

from models import classify_3dn2_gru_bn, TimeOut
from stft.config import TRAIN_PATH, VALIDATION_PATH, MODELS_FOLDER, PARAMS
from stft.data_loading import create_tfrecords_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

train_data = create_tfrecords_dataset(TRAIN_PATH, PARAMS['bs'], PARAMS['sb'])
valid_data = create_tfrecords_dataset(VALIDATION_PATH, PARAMS['bs'])

model = classify_3dn2_gru_bn(2, 4)
model_folder = os.path.join(MODELS_FOLDER, model.name + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

os.makedirs(model_folder, exist_ok=True)
with open(os.path.join(model_folder, 'params.txt'), 'w') as file:
    for (k, v) in PARAMS.items():
        file.write("{}: {}\n".format(k, v))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy')
callbacks = [
    TensorBoard(log_dir=os.path.join(model_folder, 'logs')),
    EarlyStopping(patience=3),
    TimeOut(t0=time.time(), timeout=58),
]
model.fit(train_data, steps_per_epoch=PARAMS['steps_train'],
          validation_data=valid_data, validation_steps=PARAMS['steps_validation'],
          callbacks=callbacks)
