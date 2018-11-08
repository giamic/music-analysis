import os

# PATH
DATA_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data')
EXTERNAL_DATA_FOLDER = os.path.join(os.path.abspath(os.sep), 'media', 'giamic', 'Local Disk', 'Pycharm Projects',
                                    'music_analysis', 'data', 'spotify_previews')
TRAIN_PATH = os.path.join(DATA_FOLDER, 'train.tfrecords')
VALIDATION_PATH = os.path.join(DATA_FOLDER, 'validation.tfrecords')
MODELS_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'models')

# AUDIO PREPROCESSING
SR = 44100  # the sampling rate
N_FFT = 2048  # the size of the FFT
FRAME_SIZE = 2001  # how many samples per frame of the STFT analysis
FREQUENCY_CAP = 5000  # in Hz, throw away the part of the spectrogram above to save memory and improve speed

# CLASSIFICATION PARAMETERS
PARAMS = {
    'n_train_examples': len(os.listdir(os.path.join(EXTERNAL_DATA_FOLDER, 'images', 'train'))),
    'n_validation_examples': len(os.listdir(os.path.join(EXTERNAL_DATA_FOLDER, 'images', 'validation'))),
    'bs_test': 16,  # batch_size
    'bs_train': 16,
    'sb_test': None,  # shuffle_buffer
    'sb_train': 1000,
    'x.shape': [-1, 233, 1323, 1],
    'loss_margin': 10,
    'lr': 0.001,  # learning rate
    'f1': 8,  # number of filters in the 1st layer
    'f2': 8,
    'f3': 16,
    'f4': 16,
    'k1_f': 8,  # kernel size of filters in the 1st layer (length of the filter vector)
    'k1_t': 16,  # kernel size of filters in the 1st layer (length of the filter vector)
    'k2_f': 8,  # kernel size of filters in the 1st layer (length of the filter vector)
    'k2_t': 16,  # kernel size of filters in the 1st layer (length of the filter vector)
    'k3_f': 8,
    'k3_t': 16,
    'k4_f': 8,
    'k4_t': 16,
    'n_embeddings': 24,  # number of elements in the final embeddings vector
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
