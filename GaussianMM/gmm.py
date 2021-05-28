import numpy as np
import math


def matrix_multiply(xi, mk):  # used to compute covariance
    comps = []
    vector = np.array(xi) - np.array(mk)
    for i in vector:
        component = vector * i
        comps.append(component)
    matrix = np.matrix(comps)
    return matrix


# class for a cluster
class Cluster:
    def __init__(self, cov, pi, points, index):
        self.covariance = cov
        self.pi = pi
        self.index = index
        self.points = points
        self.centroid = np.array([0, 0, 0, 0])
        self.mean = self.centroid  # here mean vector is initialized the same way as centroid

    def init_centroid(self):
        sum_x = np.array([0, 0, 0, 0])
        for x in self.points:
            sum_x = sum_x + np.array(x)
        count = len(self.points)
        self.centroid = sum_x / count

    def update_centroid(self, data, scores):
        numerator = 0
        denominator = 0
        i = 0
        for xi in data:
            x = np.array(xi)
            numerator += (scores[(i, self.index)] * x)
            denominator += scores[(i, self.index)]
            i += 1
        point = np.array(numerator) / denominator
        self.centroid = point
        return point

    def set_cov(self, cov):
        self.covariance = cov

    def set_pi(self, coeff):
        self.pi = coeff


def normal(x, cluster):
    determinant = np.linalg.det(cluster.covariance)
    N = (2 * math.pi) ** (-1 / 2)
    N = N * (determinant ** (-1 / 2))
    inverse = np.linalg.inv(cluster.covariance)
    vector = np.array(x) - np.array(cluster.centroid)
    exponent = np.matmul(vector, inverse)
    exponent = np.dot(exponent, vector)
    N = N * math.exp(-1/2*exponent)
    return N


class GMM:
    def __init__(self, data):
        self.X = data
        self.parts = np.array_split(self.X, 3)
        self.clusters = []
        self.labels = []
        self.scores = {}  # r(x,k) = P(z=k|x) score of each point x
        initlabels = np.genfromtxt("kmeans_output.tsv", delimiter="\t")
        part0=[]
        part1=[]
        part2=[]
        i = 0
        for x in self.X:
            if initlabels[i]==0:
                part0.append(x)
            else:
                if initlabels[i]==1:
                    part1.append(x)
                else:
                    part2.append(x)
            i+=1
        cov = np.cov(np.transpose(part0))
        cluster = Cluster(cov, 0.3333, part0, 0)
        cluster.init_centroid()
        self.clusters.append(cluster)
        cov = np.cov(np.transpose(part1))
        cluster = Cluster(cov, 0.3333, part1, 1)
        cluster.init_centroid()
        self.clusters.append(cluster)
        cov = np.cov(np.transpose(part2))
        cluster = Cluster(cov, 0.3333, part2, 2)
        cluster.init_centroid()
        self.clusters.append(cluster)
        # for c in range(0, 3):  # initialize clusters
        #     cov = np.cov(np.transpose(self.parts[c]))
        #     cluster = Cluster(cov, 0.3333, self.parts[c], c)
        #     cluster.init_centroid()
        #     self.clusters.append(cluster)
        self.n = 0  # iterations count
        self.previous = 0  # previous log likelihood
        self.loglike = 0

    def log(self):
        result = 0
        for x in self.X:
            inside = 0
            for k in self.clusters:
                inside += (k.pi * normal(x, k))
            result += math.log(inside)
        self.loglike = result
        return result

    def assign_clusters(self):
        i = 0
        for x in self.X:
            prob = self.scores.get((i, self.clusters[0].index))
            assign = self.clusters[0].index
            for k in self.clusters:
                if self.scores[(i, k.index)] > prob:
                    prob = self.scores[(i, k.index)]
                    assign = k.index
            i += 1
            self.labels.append(assign)

    # First, we perform a (soft) assignment of each datapoint x in D to a cluster Zi = k.
    # That is, we estimate the latent variable for each point,
    # based on our current best guesses for the model components.
    def fit(self):
        self.n = 0
        threshold = 10 ** (-5)
        fit = False
        while not fit:
            for z in self.clusters:  # expectation step, compute r(x,k)
                i = 0
                for x in self.X:
                    N = normal(x, z)
                    numer = z.pi * N
                    denom = 0
                    for j in self.clusters:
                        denom += j.pi * normal(x, j)
                    self.scores[(i, z.index)] = numer / denom
                    i += 1
                    # distance = np.linalg.norm(x - z.centroid)
                    # denominator = 0
                    # for j in self.clusters:
                    #     dd = np.linalg.norm(x - j.centroid)
                    #     denominator += math.exp(-dd * dd)
                    # self.scores[(x, z)] = math.exp(-distance * distance) / denominator
            for k in self.clusters:  # maximization step
                k.update_centroid(self.X, self.scores)
                matrix_sum = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
                score_sum = 0
                i = 0
                for x in self.X:
                    matrix = np.array(matrix_multiply(x, k.centroid))
                    matrix = matrix * self.scores[(i, k.index)]
                    matrix_sum = np.add(matrix_sum, matrix)
                    score_sum += self.scores[(i, k.index)]
                    i += 1
                covariance = np.array(matrix_sum) / score_sum
                k.set_cov(covariance)
                coefficient = score_sum / len(self.X)
                k.set_pi(coefficient)
            self.log()
            epsilon = self.loglike - self.previous
            if epsilon < threshold:
                fit = True
            else:
                self.previous = self.loglike
            self.n += 1
        self.assign_clusters()

    def save_parameters(self):
        output = np.array(self.labels)
        np.savetxt("gmm_output.tsv", output, fmt='%5.0f', delimiter="\t")


# xchen 260856206
# data initialized using kmeans
if __name__ == '__main__':
    X = np.genfromtxt("Data.tsv", delimiter="\t")
    model = GMM(X)
    model.fit()
    model.save_parameters()
