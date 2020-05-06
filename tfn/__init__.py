from pkg_resources import resource_filename

TRAIN_FILE = resource_filename(__name__, 'data/train.csv')
RESULTS_FILE = resource_filename(__name__, 'data/results.csv')
GLOVE_FILE = resource_filename(__name__, 'misc/glove.twitter.27B.%sd.txt')
GLOVE_WV_FILE = resource_filename(__name__, 'misc/glove_%s.wv')
AUG_PATH = resource_filename(__name__, 'data/augmentation')