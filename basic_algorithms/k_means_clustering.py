import numpy as np
import logging

logging.getLogger().setLevel(logging.INFO)

class KMeans:
    def __init__(self, initial_centroids, max_iterations):
        self.initial_centroids = initial_centroids
        self.k = len(self.initial_centroids)
        self.max_iterations = max_iterations
        logging.info(f"Initialized K-means with centroids: {self.initial_centroids}, num_clusters: {self.k}, and "
                     f"max_iterations: {self.max_iterations}")

    def fit(self, points):
        n_samples = len(points)
        new_centroids = centroids = self.initial_centroids

        for _ in range(self.max_iterations):
            # check the distance between centroids and the points
            clusters = {i: [] for i in range(self.k)}
            for i in range(n_samples):
                cl = np.argmin(np.array([self._euclidean_distance(points[i], centroids[j]) for j in range(self.k)]))
                # assign the point to the cluster
                clusters[int(cl)].append(np.array(points[i]))
            # compute the mean of the clusters, and make them new centroids
            new_centroids = self._calculate_centroids(clusters)

            # check if new_centroids and old centroids are equal
            if all(np.array_equal(c, nc) for c, nc in zip(centroids, new_centroids)):
                break

            centroids = new_centroids

        return [tuple((float(j) for j in i)) for i in new_centroids]

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(np.square(np.array(x1) - np.array(x2))))/len(x1)

    def _calculate_centroids(self, clusters):

        new_centroids = [np.mean(np.array(clusters[i]), axis=0) for i in clusters.keys()]
        return new_centroids

if __name__ == "__main__":

    points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)]
    k = 2
    initial_centroids = [(1, 1), (10, 1)]
    max_iterations = 10

    k_means = KMeans(initial_centroids, max_iterations)
    new_centroids = k_means.fit(points)

    print(new_centroids)