from pkg_resources import resource_filename

TRAIN_FILE = resource_filename(__name__, 'data/train.csv')
GLOVE_25_FILE = resource_filename(__name__, 'misc/glove.twitter.27B.25d.txt')
GLOVE_25_WV_FILE = resource_filename(__name__, 'misc/glove_25.wv')