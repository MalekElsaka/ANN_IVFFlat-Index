import numpy as np

NUMBER_OF_CLUSTERS = "7000"
SIZE_OF_DATABASE = "20M"
INDEX_DIRECTORY = 'index_'+SIZE_OF_DATABASE+'_'+NUMBER_OF_CLUSTERS

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70
QUERY_SEED_NUMBER = 44