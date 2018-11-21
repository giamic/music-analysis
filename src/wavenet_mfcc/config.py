import math
import os

# PATH
DATA_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data')
COMPOSERS_DATA = os.path.join(DATA_FOLDER, 'composers.csv')
# MODELS_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'models')
MAX_FILENAME_LENGTH = 255

STORAGE_FOLDER = os.path.join(os.path.abspath(os.sep), 'media', 'giamic', 'Local Disk', 'Pycharm Projects', 'music_analysis')
EXTERNAL_DATA_FOLDER = os.path.join(STORAGE_FOLDER, 'data', 'spotify_previews')
TRAIN_PATH = os.path.join(EXTERNAL_DATA_FOLDER, 'mfcc_train.tfrecords')
VALIDATION_PATH = os.path.join(EXTERNAL_DATA_FOLDER, 'mfcc_validation.tfrecords')
MODELS_FOLDER = os.path.join(STORAGE_FOLDER, 'models')

# AUDIO PREPROCESSING
SR = 44100  # the sampling rate

# CLASSIFICATION PARAMETERS
PARAMS = {
    'n_train_examples': len(os.listdir(os.path.join(EXTERNAL_DATA_FOLDER, 'images', 'train'))),
    'n_validation_examples': len(os.listdir(os.path.join(EXTERNAL_DATA_FOLDER, 'images', 'validation'))),
    'bs_test': 16,  # batch_size
    'bs_train': 16,
    'sb_test': None,  # shuffle_buffer
    'sb_train': 500,
    'x.shape': [-1, 2048, 20],
    'lr': 1e-3,  # learning rate
    'n_embeddings': 512,
    'n_composers': 13,  # number of composers in the classification task
    'epochs': 200,
    'log_step': 10,
    'profile_step': -1,
}

# PARAMS['sb_test'] = PARAMS['n_validation_examples']  # shuffle_buffer, shuffle completely
# PARAMS['sb_train'] = PARAMS['n_train_examples']  # shuffle_buffer, shuffle completely
PARAMS['steps_validation'] = int(PARAMS['n_validation_examples'] / PARAMS['bs_test'])
PARAMS['steps_train'] = int(PARAMS['n_train_examples'] / PARAMS['bs_train'])
PARAMS['steps'] = (PARAMS['steps_train'] + 1) * PARAMS['epochs']
PARAMS['test_step'] = PARAMS['steps_train'] + 1
PARAMS['triplet_loss_margin'] = math.sqrt(PARAMS['n_embeddings']) / 2
