
from typing import Dict, List, Annotated
from IVFFlat import IVFFlatIndex
from define import *
import os


class VecDB:
    def __init__(self, database_file_path , index_file_path, new_db = True, db_size = None) -> None:

        self.db_path = database_file_path
        self.index_path = index_file_path
        self.index = None

        if new_db:

            if db_size is None:
                raise ValueError("You need to provide the size of the database")

            if os.path.exists(self.db_path):
                os.remove(self.db_path)

            self.generate_database(db_size)

        else:
            self._build_index()



    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()


    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()


    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)


    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index()


    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"


    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)


    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        if self.index is None:
            raise ValueError("Index has not been built. Please build the index first.")

        return self.index.IVFFlat_retrieve(query, top_k)



    def _build_index(self):

        nProbe = 4
        numberOfClusters = int(NUMBER_OF_CLUSTERS)

        if os.path.exists(INDEX_DIRECTORY):
            self.index = IVFFlatIndex(nProbe=nProbe,numberOfClusters=numberOfClusters)
            return

        vectors = self.get_all_rows()

        print(f"Building index with {numberOfClusters} clusters and {nProbe} probes")

        self.index = IVFFlatIndex(nProbe=nProbe,numberOfClusters=numberOfClusters)

        self.index.IVFFlat_build_index(vectors,numberOfClusters=numberOfClusters)

