
from typing import Dict, List, Annotated
from IVFFlat import IVFFlatIndex
from define import *
from dataclasses import dataclass
from memory_profiler import memory_usage
import gc
from time import *
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






@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

def run_queries(db, queries, top_k, actual_ids, num_runs):
    """
    Run queries on the database and record results for each query.

    Parameters:
    - db: Database instance to run queries on.
    - queries: List of query vectors.
    - top_k: Number of top results to retrieve.
    - actual_ids: List of actual results to evaluate accuracy.
    - num_runs: Number of query executions to perform for testing.

    Returns:
    - List of Result
    """
    global results
    results = []
    for i in range(num_runs):
        tic = time.time()
        db_ids = db.retrieve(queries[i], top_k)
        toc = time.time()
        run_time = toc - tic
        results.append(Result(run_time, top_k, db_ids, actual_ids[i]))
    return results

def memory_usage_run_queries(args):
    """
    Run queries and measure memory usage during the execution.

    Parameters:
    - args: Arguments to be passed to the run_queries function.

    Returns:
    - results: The results of the run_queries.
    - memory_diff: The difference in memory usage before and after running the queries.
    """
    global results
    mem_before = max(memory_usage())
    mem = memory_usage(proc=(run_queries, args, {}), interval = 1e-3)
    return results, max(mem) - mem_before

def evaluate_result(results: List[Result]):
    """
    Evaluate the results based on accuracy and runtime.
    Scores are negative. So getting 0 is the best score.

    Parameters:
    - results: A list of Result objects

    Returns:
    - avg_score: The average score across all queries.
    - avg_runtime: The average runtime for all queries.
    """
    scores = []
    run_time = []
    for res in results:
        run_time.append(res.run_time)
        # case for retireving number not equal to top_k, socre will be the lowest
        if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
            scores.append( -1 * len(res.actual_ids) * res.top_k)
            continue
        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 3:
                    score -= ind
            except:
                score -= len(res.actual_ids)
        scores.append(score)

    return sum(scores) / len(scores), sum(run_time) / len(run_time)

def get_actual_ids_first_k(actual_sorted_ids, k):
    """
    Retrieve the IDs from the sorted list of actual IDs.
    actual IDs has the top_k for the 20 M database but for other databases we have to remove the numbers higher than the max size of the DB.

    Parameters:
    - actual_sorted_ids: A list of lists containing the sorted actual IDs for each query.
    - k: The DB size.

    Returns:
    - List of lists containing the actual IDs for each query for this DB.
    """
    return [[id for id in actual_sorted_ids_one_q if id < k] for actual_sorted_ids_one_q in actual_sorted_ids]

def _write_vectors_to_file(vectors: np.ndarray, db_path) -> None:
    mmap_vectors = np.memmap(db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
    mmap_vectors[:] = vectors[:]
    mmap_vectors.flush()

def generate_database(size: int) -> None:
    rng = np.random.default_rng(DB_SEED_NUMBER)
    vectors = rng.random((size, DIMENSION), dtype=np.float32)
    return vectors

vectors = generate_database(20*10**6)

db_filename_size_20M = 'saved_db_20M.dat'
if not os.path.exists(db_filename_size_20M): _write_vectors_to_file(vectors, db_filename_size_20M)
db_filename_size_15M = 'saved_db_15M.dat'
if not os.path.exists(db_filename_size_15M): _write_vectors_to_file(vectors[:15*10**6], db_filename_size_15M)
db_filename_size_10M = 'saved_db_10M.dat'
if not os.path.exists(db_filename_size_10M): _write_vectors_to_file(vectors[:10*10**6], db_filename_size_10M)
db_filename_size_1M = 'saved_db_1M.dat'
if not os.path.exists(db_filename_size_1M): _write_vectors_to_file(vectors[:10**6], db_filename_size_1M)

needed_top_k = 10000
rng = np.random.default_rng(QUERY_SEED_NUMBER)
query1 = rng.random((1, 70), dtype=np.float32)
query2 = rng.random((1, 70), dtype=np.float32)
query3 = rng.random((1, 70), dtype=np.float32)
query_dummy = rng.random((1, 70), dtype=np.float32)

actual_sorted_ids_20m_q1 = np.argsort(vectors.dot(query1.T).T / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query1)), axis= 1).squeeze().tolist()[::-1][:needed_top_k]
gc.collect()
actual_sorted_ids_20m_q2 = np.argsort(vectors.dot(query2.T).T / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query2)), axis= 1).squeeze().tolist()[::-1][:needed_top_k]
gc.collect()
actual_sorted_ids_20m_q3 = np.argsort(vectors.dot(query3.T).T / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query3)), axis= 1).squeeze().tolist()[::-1][:needed_top_k]
gc.collect()

queries = [query1, query2, query3]
actual_sorted_ids_20m = [actual_sorted_ids_20m_q1, actual_sorted_ids_20m_q2, actual_sorted_ids_20m_q3]

del vectors
gc.collect()

results = []
to_print_arr = []

db = VecDB(database_file_path = 'saved_db_20M.dat', index_file_path = INDEX_DIRECTORY, new_db = False)
actual_ids = get_actual_ids_first_k(actual_sorted_ids_20m, 20*10**6)

res = run_queries(db, query_dummy, 5, actual_ids, 1)

res, mem = memory_usage_run_queries((db, queries, 5, actual_ids, 3))

eval = evaluate_result(res)

to_print = f"{'20M'}\tscore\t{eval[0]}\ttime\t{eval[1]:.2f}\tRAM\t{mem:.2f} MB"
print(to_print)
to_print_arr.append(to_print)


del db
del actual_ids
del res
del mem
del eval
gc.collect()