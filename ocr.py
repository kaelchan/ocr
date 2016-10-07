#encoding=utf-8

import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm # colormap
import numpy as np
from numpy import matrix
from math import pow
from collections import namedtuple
import math
import random
import os
import json

SIZE_OF_DATA   = 400 + 1 # 20*20的图片 + 偏置
SIZE_OF_OUTPUT = 10 # 10个数字

def show(x, name=""):
    print("\n" + name)
    print(type(x))
    print(x.shape)
    print(x)
    print()

def ass(x, tup):
    assert x.shape == tup

class OCRNeuralNetwork:
    LEARNING_RATE   = 0.1
    WIDTH_IN_PIXELS = 20
    NN_FILE_PATH    = 'nn.json'

    def __init__(self, num_hidden_nodes, data_matrix, data_labels, training_indices, use_file = True, save = True):
        self.sigmoid = np.vectorize(self._sigmoid_scalar)
        self.sigmoid_diff = np.vectorize(self._sigmoid_scalar_diff)
        self._use_file = use_file
        # self.data_matrix = data_matrix
        # self.data_labels = data_labels
        if use_file:
            try:
                self._load()
                return
            except:
                print("** Error while loading json file of the neural network! **")
        # if (not os.path.isfile(OCRNeuralNetwork.NN_FILE_PATH) or not use_file):
        # Initialize with random small weights
        self.theta1 = self._rand_initialize_weights(SIZE_OF_DATA, num_hidden_nodes)
        self.theta1 = np.mat(self.theta1)
        self.theta2 = self._rand_initialize_weights(num_hidden_nodes, SIZE_OF_OUTPUT)
        self.theta2 = np.mat(self.theta2)
        # self.bias1 = self._rand_initialize_weights(1, num_hidden_nodes)
        # self.bias2 = self._rand_initialize_weights(1, SIZE_OF_OUTPUT)
        # Train using sample data
        # TrainData = namedtuple('TrainData', ['y0', 'label'])
        self.train([{'y0': data_matrix[i], 'label': data_labels[i]} for i in training_indices])
        if save:
            self.save()

    def _rand_initialize_weights(self, size_in, size_out):
        return [((x * 0.12) - 0.06) for x in np.random.rand(size_out, size_in)]
        # numpy.random.rand(d1, d2, ..)  生成一个(d1, d2, ..)多维数组，元素为0~1的小数

    def _sigmoid_scalar(self, z): # 应该有自带的？
        return 1 / (1 + math.exp(-z))

    def _sigmoid_scalar_diff(self, z): # sigmoid 的导数为 phi(1-phi)
        return np.multiply(z, 1 - z)

    def _calc(self, feature):
        y1 = np.dot(self.theta1, feature)
        y1 = self.sigmoid(y1)
        y2 = np.dot(self.theta2, y1)
        y2 = self.sigmoid(y2)
        return y1, y2

    def train(self, training_data_array):
        count = 0
        for data in training_data_array:
            count += 1
            # print(count, end=",")
            # Bias
            feature = np.mat(data['y0'] + [1]).T
            label   = int(data['label'])

            # Forward Propagation
            y1, y2 = self._calc(feature)
            # show(y1, "y1")
            # show(y2, "y2")

            # Back Propagation
            actual_vals = np.zeros(10)
            actual_vals[label] = 1
            actual_vals = np.mat(actual_vals).T # 转换为列向量
            #   # 累积的delta等于下一层的 (delta 点乘 导数) 再通过权重累积
            delta2 = (actual_vals - y2)
            # delta2 = np.multiply( (actual_vals - y2) , self.sigmoid_diff(y2) ) # this one is much worser, don't know why
            delta1 = np.multiply( self.theta2.T*delta2, self.sigmoid_diff(y1) )

            # Update
            self.theta2 += self.LEARNING_RATE * delta2 * y1.T
            self.theta1 += self.LEARNING_RATE * delta1 * feature.T

    def predict(self, test):
        # bias
        feature = np.mat(test + [1]).T

        # Forward Propagation
        y1, y2 = self._calc(feature)
        result = y2.T.tolist()[0]
        return result.index(max(result))

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "theta1": [array.tolist()[0] for array in self.theta1],
            "theta2": [array.tolist()[0] for array in self.theta2]
        }
        with open(OCRNeuralNetwork.NN_FILE_PATH, "w") as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return

        with open(OCRNeuralNetwork.NN_FILE_PATH, "w") as nnFile:
            nn = json.load(nnFile)
        self.theta1 = [np.array(lis) for lis in nn["theta1"]]
        self.theta2 = [np.array(lis) for lis in nn["theta2"]]