from sklearn.cluster import KMeans
from define import *
import pickle

class IVFFlatIndex:
    def __init__(self, nProbe, numberOfClusters):

        self.dimension = DIMENSION
        self.numberOfClusters = numberOfClusters
        self.nProbe = nProbe
        self.loaded_clusters = set()
        self.centroids = None
        self.invertedList = {}


    def IVFFlat_build_index(self, vectors, numberOfClusters):

        kmeans = KMeans(n_clusters=numberOfClusters, random_state=DB_SEED_NUMBER, n_init=10, max_iter=300)
        assignments = kmeans.fit_predict(vectors)


        self.centroids = kmeans.cluster_centers_
        self.invertedList = {i: [] for i in range(numberOfClusters)}

        for idx, clusterId in enumerate(assignments):
            self.invertedList[clusterId].append(idx)


        self.IVFFlat_save_index()

        del kmeans
        del assignments

        self.centroids = None
        self.invertedList = {}



    def IVFFlat_find_nearest_centroids(self, query, n_probe):

        distances = np.linalg.norm(self.centroids - query, axis=1)
        return np.argsort(distances)[:n_probe]


    def IVFFlat_retrieve(self, query, top_k):


        if self.centroids is None:
            self.IVFFlat_load_centroids()

        nearestCentroids = self.IVFFlat_find_nearest_centroids(query, self.nProbe)

        candidates = []
        for centroid_idx in nearestCentroids:
            self.IVFFlat_load_cluster(centroid_idx)
            candidates.extend(self.invertedList[centroid_idx])

        scores = []
        for idx in candidates:
            vector = db.get_one_row(idx)
            score = self.IVFFlat_cal_score(query, vector)
            scores.append((score, idx))

        del candidates

        scores = sorted(scores, reverse=True)[:top_k]

        return [s[1] for s in scores]


    def IVFFlat_cal_score(self, vec1, vec2):

        dot_product = np.dot(vec1, vec2)
        normVec1 = np.linalg.norm(vec1)
        normVec2 = np.linalg.norm(vec2)
        return dot_product / (normVec1 * normVec2)


    def IVFFlat_save_index(self):

        os.makedirs(INDEX_DIRECTORY, exist_ok=True)

        # Save centroids file
        centroids_path = os.path.join(INDEX_DIRECTORY, 'centroids.pkl')
        with open(centroids_path, 'wb') as f:
            pickle.dump(self.centroids, f)

        # Save each cluster's inverted list to a separate file
        for cluster_id, vector_indices in self.invertedList.items():
            cluster_file_path = os.path.join(INDEX_DIRECTORY, f'cluster_{cluster_id}.pkl')
            with open(cluster_file_path, 'wb') as f:
                pickle.dump(vector_indices, f)


        print(f"Index saved to directory: {INDEX_DIRECTORY}")



    def IVFFlat_load_centroids(self):

        centroids_path = os.path.join(INDEX_DIRECTORY, 'centroids.pkl')
        with open(centroids_path, 'rb') as f:
            self.centroids = pickle.load(f)


        print(f"Index loaded from directory: {INDEX_DIRECTORY}")


    def IVFFlat_load_cluster(self, cluster_id):

        if cluster_id in self.loaded_clusters:
            return

        cluster_file_path = os.path.join(INDEX_DIRECTORY, f'cluster_{cluster_id}.pkl')

        try:
            with open(cluster_file_path, 'rb') as f:
                cluster_indices = pickle.load(f)

            self.invertedList[cluster_id] = cluster_indices
            self.loaded_clusters.add(cluster_id)

        except FileNotFoundError:
            print(f"Cluster {cluster_id} file not found.")

        return
