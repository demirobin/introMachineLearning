import numpy as np

"""
Code by demirobin
"""
# class for a cluster
class Cluster:
    def __init__(self, pos, index):
        self.centroid = np.array(pos)
        self.points = []
        self.previous = []  # store previous point set of this Cluster
        self.index = index  # index of this Cluster used to label points

    def update_centroid(self):
        sum_x = np.array([0, 0, 0, 0])
        for x in self.points:
            sum_x = sum_x + np.array(x)
        count = len(self.points)
        self.centroid = sum_x / count

    def reset(self):  # reset the point set for next round
        self.previous = self.points
        self.points = []


class KMeans:
    def __init__(self, data):
        self.dataset = data
        self.label = []  # list of the cluster labels of the points in dataset
        self.K = 3  # 3 centroids given in instruction
        self.clusters = []
        # initialize centroids
        self.centroids = [[1.03800476, 0.09821729, 1.0469454, 1.58046376],
                          [0.18982966, -1.97355361, 0.70592084, 0.3957741],
                          [1.2803405, 0.09821729, 0.76275827, 1.44883158]]
        # initialize cluster sets
        for z in range(0, self.K):
            self.clusters.append(Cluster(self.centroids[z], z))
        self.n = 0  # iterations count

    # For each point x in the dataset, assign that point to the cluster with the closest centroid.
    # That is, form clusters sets Cz = {x: ||x-Uz|| < ||x-Uj||, any j in range(0,clusters), j!=z}
    # Loop until no points are re-assigned
    def fit(self):
        self.n = 0
        fit = False
        while not fit:
            self.label = []
            for point in self.dataset:
                closest = self.assign_cluster(point)
                self.label.append(closest.index)
                closest.points.append(point)
            for C in self.clusters:
                C.update_centroid()
            stable = 0  # denote the number of clusters that stabilize in this round
            for C in self.clusters:
                a = np.array(C.points)
                b = np.array(C.previous)
                a.sort()
                b.sort()
                if np.array_equal(a, b):
                    stable = stable + 1
            if stable == self.K:
                fit = True
            else:
                for C in self.clusters:
                    C.reset()
            self.n += 1

    # return the cluster that has the closest centroid to the point
    def assign_cluster(self, point):
        distances = {}
        x = np.array(point)
        for C in self.clusters:
            distances[C] = np.linalg.norm(C.centroid - x)
        minimum = list(distances.values())[0]
        closest = self.clusters[0]
        for d in distances.keys():
            if distances.get(d) <= minimum:
                minimum = distances.get(d)
                closest = d
        return closest

    def save_parameters(self):
        output = np.array(self.label)
        np.savetxt("kmeans_output.tsv", output, fmt='%5.0f', delimiter="\t")


if __name__ == '__main__':
    X = np.genfromtxt("Data.tsv", delimiter="\t")
    model = KMeans(X)
    model.fit()
    model.save_parameters()
