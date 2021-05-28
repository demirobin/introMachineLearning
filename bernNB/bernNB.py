import numpy as np
import math

"""
Code by demirobin
"""
class BernoulliNaiveBayes:
    def __init__(self, training):
        self.dataset = training
        self.X = training[:, :-1]
        self.y = training[:, -1]
        self.feature = {}  # theta_j_k= P(x[j] = 1|y = k)
        # self.prior = 0

    def prior(self):
        total = len(self.y)
        num = 0
        for yi in self.y:
            if yi == 1:
                num += 1
        theta_1 = num / total
        # self.prior = theta_1
        return theta_1

    def fit(self):
        for j in range(0, len(self.X[0])):
            for k in range(0, 2):
                self.feature[(j, k)] = 0
        total = len(self.y)
        k1 = 0  # the total number of (X,y) with y ==1
        for yi in self.y:
            if yi == 1:
                k1 += 1
        k0 = total - k1  # the total number of (X,y) with y ==0
        for j in range(0, len(self.X[0])):  # j features
            hit = 0  # feature occurs when y==1
            miss = 0  # feature occurs when y==0
            for i in range(0, len(self.X)):
                if self.X[i][j] == 1 and self.y[i] == 1:
                    hit += 1
                if self.X[i][j] == 1 and self.y[i] == 0:
                    miss += 1
            self.feature[(j, 1)] = hit / k1
            self.feature[(j, 0)] = miss / k0
        print("fit done")

    def predict(self, matrix):
        theta_1 = self.prior()
        sum_j1 = 0
        sum_j0 = 0
        for j in range(0, len(self.X[0])):
            sum_j1 += math.log(1 - self.feature[(j, 1)])
            sum_j0 += math.log(1 - self.feature[(j, 0)])
        bias = math.log(theta_1) - math.log(1 - theta_1) + sum_j1 - sum_j0
        w = np.zeros(len(self.X[0]))
        for j in range(0, len(self.X[0])):
            w[j] = math.log(self.feature[(j, 1)]) - math.log(1 - self.feature[(j, 1)]) - math.log(self.feature[(j, 0)])
            w[j] += math.log(1 - self.feature[(j, 0)])
        f_theta = np.zeros(len(matrix))
        for i in range(0, len(matrix)):
            sum_wx = 0
            for j in range(0, len(self.X[0])):
                sum_wx += w[j] * matrix[i][j]
            if bias + sum_wx > 0:
                f_theta[i] = 1
            else:
                f_theta[i] = 0
        return f_theta

    def positive(self):
        pos_like = np.array(self.feature[(0, 1)])
        for j in range(1, len(self.X[0])):
            pos_like = np.append(pos_like, self.feature[(j, 1)])
        return pos_like

    def negative(self):
        neg_like = np.array(self.feature[(0, 0)])
        for j in range(1, len(self.X[0])):
            neg_like = np.append(neg_like, self.feature[(j, 0)])
        return neg_like

    def save_parameters(self):
        priors = np.append(self.prior(), 1 - self.prior())
        pos = self.positive()
        neg = self.negative()
        np.savetxt("class_priors.tsv", priors, delimiter="\\t")
        print("priors saved")
        np.savetxt("positive_feature_likelihoods.tsv", pos, delimiter="\\t")
        print("pos saved")
        np.savetxt("negative_feature_likelihoods.tsv", neg, delimiter="\\t")
        print("neg saved")


if __name__ == '__main__':
    data = np.loadtxt(".//train_dataset.tsv", delimiter="\t")
    model = BernoulliNaiveBayes(data)
    model.fit()
    model.save_parameters()
