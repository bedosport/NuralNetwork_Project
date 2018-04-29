import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tkinter import *
import segment as seg
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from sklearn.metrics import confusion_matrix
import math
import copy
from sklearn.cluster import KMeans

# cat = 1
# laptop = 2
# apple = 3
# car = 4
# helicopter = 5

# best  k = 22
# best epochs = 200


class RBF:
    def __init__(self, k, epochs):
        self.epochs = epochs
        self.k = k
        # self.load_data()
        self.TrainData = np.genfromtxt("TrainData.txt", delimiter=",")
        self.Train_labels = np.genfromtxt("Train_labels.txt", delimiter=",")

        self.TestData = np.genfromtxt("TestData.txt", delimiter=',')
        self.Test_labels = np.genfromtxt("Test_labels.txt", delimiter=',')

        self.Weight_out = np.zeros([5, k])  # 2nd level of weights //

        self.hidden_neurons = self.kmean(k, self.TrainData)  # initiate hidden neurons k* num of samples
        kmeans = KMeans(n_clusters=22, random_state=0).fit(self.TrainData)
        # self.hidden_neurons=kmeans.cluster_centers_
        # print(kmeans.cluster_centers_)

        # = k*25

        self.hidden_Gaussian = np.zeros(k)
        self.init_sigma()
        self.init_weights()
        self.OutError1 = np.zeros((25, 1))
        self.OutError2 = np.zeros((25, 1))
        self.OutError3 = np.zeros((25, 1))
        self.OutError4 = np.zeros((25, 1))
        self.OutError5 = np.zeros((25, 1))

        # [0, 0: self.NumberOfNeurons[Level - 1]]
    def ReadTrainingData(self):
        self.TrainData = np.loadtxt('TrainingDataBeforePca.txt')

        self.TrainData = self.TrainData.reshape((25, 2500))
    def confusion(self, pred, real):
        con = confusion_matrix(real, pred)
        print(con)
        acc = 0
        for i in range(5):
            acc += con[i, i]
        return (acc / len(real)) * 100

    def calc_Gaussian(self, sample):  # guess WRONG ??????????????????
        tmp_hidden_Gaussian = np.zeros([self.k])
        for i in range(self.k):  # k
            tmp_hidden_Gaussian[i] = math.exp(
                -(self.Euclidean_dis(sample, self.hidden_neurons[i]) ** 2) / (2 * self.sigma ** 2))

        return tmp_hidden_Gaussian

    def init_weights(self):
        # out level
        np.random.seed(0)
        for i in range(5):
            for j in range(self.k):
                self.Weight_out[i][j] = np.random.uniform(-1, 1)

    def Euclidean_dis(self, first, second):
        dis = 0
        for i in range(len(first)):  # number of features
            dis += (first[i] - second[i]) * (first[i] - second[i])
        dis = math.sqrt(dis)
        return dis

    def init_sigma(self):  # ?
        Max_dis = 1
        # Max_dis => max distance between any 2 centroids
        max = -100000000000
        for i in range(len(self.hidden_neurons)):
            for j in range(i + 1, len(self.hidden_neurons)):
                cur_dist = self.Euclidean_dis(self.hidden_neurons[i], self.hidden_neurons[j])
                if cur_dist > max:
                    max = cur_dist
        Max_dis = max
        self.sigma = Max_dis / math.sqrt(2 * self.k)

    def kmean(self, k, data_set):
        # comment this and add h ,w dimensions

        w = len(data_set[0])  # number of features
        h = len(data_set)  # number of data
        # print(len(data_set))

        self.Matrix = data_set
        # print("===============")
        centroids = []
        # intalize empty adjancy lists
        for i in range(k):
            tmp_list = []
            centroids.append(tmp_list)

        for i in range(k):
            for j in range(w):
                centroids[i].append(self.Matrix[i][j])

        # print(centroids)

        New_centroids = copy.deepcopy(centroids)
        cc = 0
        while (1):

            classes = []
            classes.clear()
            # intalize empty adjancy lists
            for i in range(k):
                tmp_list = []
                classes.append(tmp_list)

            for i in range(h):  # iterate over the data
                min = 1000000
                assigned_cluster = 0  # assume it's always belong to thr 1st cluster
                for j in range(k):  # iterate over k classes
                    Euclidean = 0
                    for l in range(w):
                        Euclidean += (self.Matrix[i][l] - centroids[j][l]) * (
                                self.Matrix[i][l] - centroids[j][l])

                    Euclidean = math.sqrt(Euclidean);
                    if (Euclidean < min):
                        min = Euclidean
                        assigned_cluster = j
                # print(assigned_cluster)
                classes[assigned_cluster].append(i)

            # print(classes,end='')
            # calculating  new centroid of each class
            # print(centroids)

            for i in range(k):
                for o in range(w):
                    avg = 0
                    for j in range(len(classes[i])):
                        avg += self.Matrix[classes[i][j]][o]
                    New_centroids[i][o] = avg / len(classes[i])

            # print(centroids)
            # print(New_centroids)
            # check for the stopping condition
            cnt = 0
            cnt2 = 0
            for i in range(k):
                cnt = 0
                for j in range(w):
                    if (New_centroids[i][j] - centroids[i][j] <= 0.0):
                        cnt += 1;
                if (cnt == w):
                    cnt2 += 1;
            if cnt2 == k:
                # print(New_centroids)
                return New_centroids
                break
            cc += 1
            # print(cc)
            centroids = copy.deepcopy(New_centroids)

    def train(self, learn_rate=0.09, mse_thresh=0.01):

        epoches = self.epochs
        epoch_list = np.zeros([epoches, 1])

        for e in range(epoches):
            # print(self.Weight_out)
            for i in range(len(self.TrainData)):  # iterate over samples
                # net = w * φ T
                X = self.TrainData[i]
                cur_hidden_Gaussian = self.calc_Gaussian(X)
                # print(cur_hidden_Gaussian.shape)
                # print(self.Weight_out[0, 0:self.k].shape)

                vnet1 = np.sum(self.Weight_out[0, 0:self.k] * cur_hidden_Gaussian)
                vnet2 = np.sum(self.Weight_out[1, 0:self.k] * cur_hidden_Gaussian)
                vnet3 = np.sum(self.Weight_out[2, 0:self.k] * cur_hidden_Gaussian)
                vnet4 = np.sum(self.Weight_out[3, 0:self.k] * cur_hidden_Gaussian)
                vnet5 = np.sum(self.Weight_out[4, 0:self.k] * cur_hidden_Gaussian)

                D = self.Train_labels[i]

                if D == 1:
                    D1 = 1
                    D2 = 0
                    D3 = 0
                    D4 = 0
                    D5 = 0
                elif D == 2:
                    D1 = 0
                    D2 = 1
                    D3 = 0
                    D4 = 0
                    D5 = 0
                elif D == 3:
                    D1 = 0
                    D2 = 0
                    D3 = 1
                    D4 = 0
                    D5 = 0
                elif D == 4:
                    D1 = 0
                    D2 = 0
                    D3 = 0
                    D4 = 1
                    D5 = 0
                elif D == 5:
                    D1 = 0
                    D2 = 0
                    D3 = 0
                    D4 = 0
                    D5 = 1
                # calc errors
                self.OutError1[i] = (D1 - vnet1)
                self.OutError2[i] = (D2 - vnet2)
                self.OutError3[i] = (D3 - vnet3)
                self.OutError4[i] = (D4 - vnet4)
                self.OutError5[i] = (D5 - vnet5)
                # update weights
                self.Weight_out[0, 0:self.k] = self.Weight_out[0, 0:self.k] + self.OutError1[
                    i] * learn_rate * cur_hidden_Gaussian
                self.Weight_out[1, 0:self.k] = self.Weight_out[1, 0:self.k] + self.OutError2[
                    i] * learn_rate * cur_hidden_Gaussian
                self.Weight_out[2, 0:self.k] = self.Weight_out[2, 0:self.k] + self.OutError3[
                    i] * learn_rate * cur_hidden_Gaussian
                self.Weight_out[3, 0:self.k] = self.Weight_out[3, 0:self.k] + self.OutError4[
                    i] * learn_rate * cur_hidden_Gaussian
                self.Weight_out[4, 0:self.k] = self.Weight_out[4, 0:self.k] + self.OutError5[
                    i] * learn_rate * cur_hidden_Gaussian
                # print (self.Weight_out)
            MSE1 = 0.5 * np.mean((self.OutError1 ** 2))
            MSE2 = 0.5 * np.mean((self.OutError2 ** 2))
            MSE3 = 0.5 * np.mean((self.OutError3 ** 2))
            MSE4 = 0.5 * np.mean((self.OutError4 ** 2))
            MSE5 = 0.5 * np.mean((self.OutError5 ** 2))
            TotalMSE = (MSE1 + MSE2 + MSE3 + MSE4 + MSE5) / 5
            print("epoch", str(e) + "---", TotalMSE)
            epoch_list[e] = e
            if TotalMSE <= mse_thresh:
                break
        self.save_Weight()
        self.save_Cenroids()
        return epoch_list

    def test(self):

        self.k = 22
        self.ReadWeight()

        self.ReadCenroids()
        self.init_sigma()

        y = np.zeros([len(self.TestData), 1])
        for i in range(0, len(self.TestData)):
            X = self.TestData[i]
            cur_hidden_Gaussian = self.calc_Gaussian(X)
            vnet1 = np.sum(self.Weight_out[0, 0:self.k] * cur_hidden_Gaussian)
            vnet2 = np.sum(self.Weight_out[1, 0:self.k] * cur_hidden_Gaussian)
            vnet3 = np.sum(self.Weight_out[2, 0:self.k] * cur_hidden_Gaussian)
            vnet4 = np.sum(self.Weight_out[3, 0:self.k] * cur_hidden_Gaussian)
            vnet5 = np.sum(self.Weight_out[4, 0:self.k] * cur_hidden_Gaussian)
            if vnet1 > vnet2 and vnet1 > vnet3 and vnet1 > vnet4 and vnet1 > vnet5:
                y[i] = 1
            elif vnet2 > vnet1 and vnet2 > vnet3 and vnet2 > vnet4 and vnet2 > vnet5:
                y[i] = 2
            elif vnet3 > vnet1 and vnet3 > vnet2 and vnet3 > vnet4 and vnet3 > vnet5:
                y[i] = 3
            elif vnet4 > vnet1 and vnet4 > vnet3 and vnet4 > vnet2 and vnet4 > vnet5:
                y[i] = 4
            else:
                y[i] = 5
        d = self.Test_labels
        # print(y)
        print("test accuarcy is   ", self.confusion(y, d))

        return

    def test_sample(self, X):

        self.k = 22
        self.ReadWeight()

        self.ReadCenroids()
        self.init_sigma()
        X=X.T
        print (X.shape)

        cur_hidden_Gaussian = self.calc_Gaussian(X)
        vnet1 = np.sum(self.Weight_out[0, 0:self.k] * cur_hidden_Gaussian)
        vnet2 = np.sum(self.Weight_out[1, 0:self.k] * cur_hidden_Gaussian)
        vnet3 = np.sum(self.Weight_out[2, 0:self.k] * cur_hidden_Gaussian)
        vnet4 = np.sum(self.Weight_out[3, 0:self.k] * cur_hidden_Gaussian)
        vnet5 = np.sum(self.Weight_out[4, 0:self.k] * cur_hidden_Gaussian)
        if vnet1 > vnet2 and vnet1 > vnet3 and vnet1 > vnet4 and vnet1 > vnet5:
            label = 1
        elif vnet2 > vnet1 and vnet2 > vnet3 and vnet2 > vnet4 and vnet2 > vnet5:
            label = 2
        elif vnet3 > vnet1 and vnet3 > vnet2 and vnet3 > vnet4 and vnet3 > vnet5:
            label = 3
        elif vnet4 > vnet1 and vnet4 > vnet3 and vnet4 > vnet2 and vnet4 > vnet5:
            label = 4
        else:
            label = 5
        return label

    def load_data(self):
        mydata = np.genfromtxt("IrisDataset.txt", delimiter=",")
        # print(mydata)
        self.my_data = mydata

    def save_Weight(self):
        with open('RBF_Weights.txt', 'w') as outfile:
            for data_slice in self.Weight_out:
                np.savetxt(outfile, data_slice, fmt='%-7.10f')

    def save_Cenroids(self):
        with open('Centroids.txt', 'w') as outfile:
            for data_slice in self.hidden_neurons:
                np.savetxt(outfile, data_slice, fmt='%-7.5f')

    def ReadWeight(self):
        self.Weight_out = np.loadtxt('RBF_Weights.txt')
        self.Weight_out = self.Weight_out.reshape(5, self.k)

    def ReadCenroids(self):
        self.hidden_neurons = np.loadtxt('Centroids.txt')
        ## k=22
        self.hidden_neurons = self.hidden_neurons.reshape(22, 25)


class MultiLayerNN:
    def __init__(self, no_layers=2, no_neu="15", b=0, lr=.3, no_ep=200, af="Sigmoid", sc="MSE", mse=.01, bool=0):
        # System Variables
        self.TestingData = []
        self.CrossData = []
        self.TrainingData = []
        self.TrainingLabels = []
        self.TestingLabels = []
        self.CrossLabels = []
        self.CrossBool = bool
        self.bias = int(b)
        self.NumberOfLayers = int(no_layers)
        self.no_epochs = 0
        x = no_neu.split(',')
        z = []
        # fill the neurons number at each layer
        for i in x:
            z.append(int(i))
        arr = np.array(z)
        if arr.shape[0] == 1:
            self.NumberOfNeurons = np.full(self.NumberOfLayers, arr[0])
            self.MaxNeuron = int(arr[0])
        elif arr.shape[0] == self.NumberOfLayers:
            self.NumberOfNeurons = np.zeros(self.NumberOfLayers, int)
            self.MaxNeuron = 0
            for i in range(0, self.NumberOfLayers):
                self.NumberOfNeurons[i] = arr[i]
                self.MaxNeuron = max(self.MaxNeuron, arr[i])
        else:
            print("Please enter correct number of neurons per layer");
            return
        self.learning_rate = float(lr)
        self.ActivationFunction = af
        self.StoppingCondition = sc
        if sc == "Fix The Number Of Epochs":
            self.no_epochs = int(no_ep)
        elif sc == "MSE":
            self.MSEThreshold = float(mse)
        if self.CrossBool == 1:
            self.NumberOfFeatures = 20
        else:
            self.NumberOfFeatures = 25
        self.NumberOfClasses = 5
        self.Weights = np.zeros((self.NumberOfLayers + 1, max(self.MaxNeuron, self.NumberOfFeatures), max(self.MaxNeuron, self.NumberOfFeatures)))
        self.biasList = np.zeros((self.NumberOfLayers + 1, 1))
        self.Out = np.zeros((self.NumberOfLayers + 1, max(self.MaxNeuron, self.NumberOfFeatures)))
        self.Error = np.zeros((self.NumberOfLayers + 1, max(self.MaxNeuron, self.NumberOfFeatures)))
        self.OutError1 = np.zeros((25, 1))
        self.OutError2 = np.zeros((25, 1))
        self.OutError3 = np.zeros((25, 1))
        self.OutError4 = np.zeros((25, 1))
        self.OutError5 = np.zeros((25, 1))
        self.Epochs = []

    def Start(self):
        # Start Read the images
        self.Read("/Users/mac/PycharmProjects/NNProject/Training/", "/Users/mac/PycharmProjects/NNProject/TestFiles/")
        # Normalize the Image data set
        self.WriteTrainingData()
        self.Normalize()
        with open('TestingAfterNorm.txt', 'w') as outfile:
            np.savetxt(outfile, self.TestingData, fmt='%-7.2f')
        # save the training data for found PCA
        # Calculate PCA Number
        self.calculate_pca()
        # initialize the weight matrix and bias list vector
        self.initialize()
        # Train The Network
        self.train()
        # Save the weights into file
        self.WriteWeightAndBias()
        # Testing phase
        self.TestData()
        cv.waitKey(0)

    def initialize(self):
        np.random.seed(0)
        for i in range(0, self.Weights.shape[0]):
            for j in range(0, self.Weights.shape[1]):
                self.Weights[i, j] = np.random.uniform(-1, 1)
        if self.bias == 1:
            for i in range(0, self.biasList.shape[0]):
                self.biasList[i] = np.random.uniform(-1, 1)

    def ActFunction(self, vnet):
        if self.ActivationFunction == "Sigmoid":
            return 1 / (1 + math.exp(vnet * -1))
        else:
            return (1 - math.exp(vnet * -1)) / (1 + math.exp(vnet * -1))

    def WriteTrainingData(self):
        with open('TrainingDataBeforePca.txt', 'w') as outfile:
            np.savetxt(outfile, self.TrainingData, fmt='%-7.2f')

    def ReadTrainingData(self):
        self.TrainingData = np.loadtxt('TrainingDataBeforePca.txt')
        if self.CrossBool == 1:
            self.TrainingData = self.TrainingData.reshape((20, 2500))
        else:
            self.TrainingData = self.TrainingData.reshape((25, 2500))

    def train(self):
        preAcc = -1
        Epoch_Number = 0
        OK = True
        while OK:
            for index in range(0, self.TrainingData.shape[0]):
                X = self.TrainingData[index]
                D = self.TrainingLabels[index]
                # Forward Step....
                for Level in range(0, self.NumberOfLayers + 1):
                    if Level == 0:
                        Weight = self.Weights[Level]
                        for NeuronIndx in range(0, self.NumberOfNeurons[Level]):
                            Vnet = np.sum(Weight[NeuronIndx, 0:self.NumberOfFeatures] * X) + self.biasList[Level] * self.bias
                            self.Out[Level, NeuronIndx] = self.ActFunction(Vnet)

                    elif Level == self.NumberOfLayers:
                        Weight = self.Weights[Level]
                        X = self.Out[Level - 1, 0:self.NumberOfNeurons[Level - 1]]

                        Vnet = np.sum(Weight[0, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[
                            Level] * self.bias
                        Out1 = self.ActFunction(Vnet)

                        Vnet = np.sum(Weight[1, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[
                            Level] * self.bias
                        Out2 = self.ActFunction(Vnet)

                        Vnet = np.sum(Weight[2, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[
                            Level] * self.bias
                        Out3 = self.ActFunction(Vnet)

                        Vnet = np.sum(Weight[3, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[
                            Level] * self.bias
                        Out4 = self.ActFunction(Vnet)

                        Vnet = np.sum(Weight[4, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[
                            Level] * self.bias
                        Out5 = self.ActFunction(Vnet)

                        if D == 1:
                            D1 = 1
                            D2 = 0
                            D3 = 0
                            D4 = 0
                            D5 = 0
                        elif D == 2:
                            D1 = 0
                            D2 = 1
                            D3 = 0
                            D4 = 0
                            D5 = 0
                        elif D == 3:
                            D1 = 0
                            D2 = 0
                            D3 = 1
                            D4 = 0
                            D5 = 0
                        elif D == 4:
                            D1 = 0
                            D2 = 0
                            D3 = 0
                            D4 = 1
                            D5 = 0
                        elif D == 5:
                            D1 = 0
                            D2 = 0
                            D3 = 0
                            D4 = 0
                            D5 = 1

                        self.OutError1[index] = (D1 - Out1)
                        self.OutError2[index] = (D2 - Out2)
                        self.OutError3[index] = (D3 - Out3)
                        self.OutError4[index] = (D4 - Out4)
                        self.OutError5[index] = (D5 - Out5)
                        E1 = (D1 - Out1) * Out1 * (1 - Out1)
                        E2 = (D2 - Out2) * Out2 * (1 - Out2)
                        E3 = (D3 - Out3) * Out3 * (1 - Out3)
                        E4 = (D4 - Out4) * Out4 * (1 - Out4)
                        E5 = (D5 - Out5) * Out5 * (1 - Out5)
                        self.Error[Level, 0] = E1
                        self.Error[Level, 1] = E2
                        self.Error[Level, 2] = E3
                        self.Error[Level, 3] = E4
                        self.Error[Level, 4] = E5
                    else:
                        Weight = self.Weights[Level]
                        X = self.Out[Level - 1, 0:self.NumberOfNeurons[Level - 1]]
                        for NeuronIndx in range(0, self.NumberOfNeurons[Level]):
                            Vnet = np.sum(Weight[NeuronIndx, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[
                                Level] * self.bias
                            self.Out[Level, NeuronIndx] = self.ActFunction(Vnet)
                # Backward Step .......
                for Level in range(self.NumberOfLayers, -1, -1):
                    if Level == self.NumberOfLayers:
                        weight = self.Weights[Level]
                        for NeuronIndx in range(0, self.NumberOfNeurons[Level - 1]):
                            f = self.Out[Level - 1, NeuronIndx]
                            self.Error[Level - 1, NeuronIndx] = f * (1 - f) * np.sum(
                                self.Error[Level, 0:5] * weight[0:5, NeuronIndx])
                            Temp = weight[0:5, NeuronIndx] + self.learning_rate * self.Error[Level, 0:5] * f
                            self.Weights[Level, 0:5, NeuronIndx] = Temp[0:5]
                        self.biasList[Level] = self.biasList[Level] + np.sum(
                            self.learning_rate * self.Error[Level, 0:5] * self.bias)
                    elif Level == 0:
                        weight = self.Weights[Level]
                        for NeuronIndx in range(0, self.NumberOfFeatures):
                            X = self.TrainingData[index]
                            NTemp = weight[0:self.NumberOfNeurons[Level], NeuronIndx] + self.learning_rate * self.Error[
                                                                                                             Level, 0:
                                                                                                                    self.NumberOfNeurons[
                                                                                                                        Level]] * \
                                    X[NeuronIndx]
                            self.Weights[Level, 0:self.NumberOfNeurons[Level], NeuronIndx] = NTemp[
                                                                                             0:self.NumberOfNeurons[
                                                                                                 Level]]
                        self.biasList[Level] = self.biasList[Level] + np.sum(
                            self.learning_rate * self.Error[Level, 0:self.NumberOfNeurons[Level]] * self.bias)

                    else:
                        weight = self.Weights[Level]
                        for NeuronIndx in range(0, self.NumberOfNeurons[Level - 1]):
                            f = self.Out[Level - 1, NeuronIndx]
                            self.Error[Level - 1, NeuronIndx] = f * (1 - f) * np.sum(
                                self.Error[Level, 0:self.NumberOfNeurons[Level]] * weight[0:self.NumberOfNeurons[Level],
                                                                                   NeuronIndx])
                            Temp = weight[0:self.NumberOfNeurons[Level], NeuronIndx] + self.learning_rate * self.Error[
                                                                                                            Level, 0:
                                                                                                                   self.NumberOfNeurons[
                                                                                                                       Level]] * f
                            self.Weights[Level, 0:self.NumberOfNeurons[Level], NeuronIndx] = Temp[
                                                                                             0:self.NumberOfNeurons[
                                                                                                 Level]]
                        self.biasList[Level] = self.biasList[Level] + np.sum(
                            self.learning_rate * self.Error[Level, 0:self.NumberOfNeurons[Level]] * self.bias)

            MSE1 = 0.5 * np.mean((self.OutError1 ** 2))
            MSE2 = 0.5 * np.mean((self.OutError2 ** 2))
            MSE3 = 0.5 * np.mean((self.OutError3 ** 2))
            MSE4 = 0.5 * np.mean((self.OutError4 ** 2))
            MSE5 = 0.5 * np.mean((self.OutError5 ** 2))
            TotalMSE = (MSE1 + MSE2 + MSE3 + MSE4 + MSE5)/5
            self.Epochs.append(TotalMSE)
            if self.StoppingCondition == "Cross Validation":
                if (Epoch_Number+1) % 50 == 0:
                    print(preAcc)
                    if preAcc == -1:
                        preAcc = self.CrossTest()
                    else:
                        Acc = self.CrossTest()
                        if preAcc > Acc:
                            print("Early stopping !!")
                            break
                        preAcc = Acc
            elif self.StoppingCondition == "MSE":
                print(TotalMSE)
                if TotalMSE <= self.MSEThreshold:
                    break
            else:
                if Epoch_Number == self.no_epochs - 1:
                    break
            Epoch_Number += 1

    def WriteWeightAndBias(self):
        with open('Weights.txt', 'w') as outfile:
            for data_slice in self.Weights:
                np.savetxt(outfile, data_slice, fmt='%-7.2f')
        with open('Bias.txt', 'w') as outfile:
            for data_slice in self.biasList:
                np.savetxt(outfile, data_slice, fmt='%-7.2f')

    def ReadWeightAndBias(self):
        self.Weights = np.loadtxt('Weights.txt')
        self.Weights = self.Weights.reshape((self.NumberOfLayers + 1, max(self.MaxNeuron, self.NumberOfFeatures), max(self.MaxNeuron, self.NumberOfFeatures)))
        self.biasList = np.loadtxt('Bias.txt')
        self.biasList = self.biasList.reshape((self.NumberOfLayers + 1, 1))

    @staticmethod
    def confusion(predicted, real):
        con = confusion_matrix(real, predicted)
        acc = 0
        for i in range(5):
            acc += con[i, i]
        print(con)
        return (acc / len(real)) * 100

    def test(self, X):
        # Forward Step....
        for Level in range(0, self.NumberOfLayers + 1):
            if Level == 0:
                Weight = self.Weights[Level]
                for NeuronIndx in range(0, self.NumberOfNeurons[Level]):
                    Vnet = np.sum(Weight[NeuronIndx, 0:self.NumberOfFeatures] * X) + self.biasList[Level] * self.bias
                    self.Out[Level, NeuronIndx] = self.ActFunction(Vnet)

            elif Level == self.NumberOfLayers:
                Weight = self.Weights[Level]
                X = self.Out[Level - 1, 0:self.NumberOfNeurons[Level - 1]]

                Vnet = np.sum(Weight[0, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[Level] * self.bias
                Out1 = self.ActFunction(Vnet)

                Vnet = np.sum(Weight[1, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[Level] * self.bias
                Out2 = self.ActFunction(Vnet)

                Vnet = np.sum(Weight[2, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[Level] * self.bias
                Out3 = self.ActFunction(Vnet)

                Vnet = np.sum(Weight[3, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[Level] * self.bias
                Out4 = self.ActFunction(Vnet)

                Vnet = np.sum(Weight[4, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[Level] * self.bias
                Out5 = self.ActFunction(Vnet)

                if Out1 > Out2 and Out1 > Out3 and Out1 > Out4 and Out1 > Out5:
                    return 1
                elif Out2 > Out1 and Out2 > Out3 and Out2 > Out4 and Out2 > Out5:
                    return 2
                elif Out3 > Out1 and Out3 > Out2 and Out3 > Out4 and Out3 > Out5:
                    return 3
                elif Out4 > Out1 and Out4 > Out3 and Out4 > Out2 and Out4 > Out5:
                    return 4
                else:
                    return 5
            else:
                Weight = self.Weights[Level]
                X = self.Out[Level - 1, 0:self.NumberOfNeurons[Level - 1]]
                for NeuronIndx in range(0, self.NumberOfNeurons[Level]):
                    Vnet = np.sum(Weight[NeuronIndx, 0:self.NumberOfNeurons[Level - 1]] * X) + self.biasList[
                        Level] * self.bias
                    self.Out[Level, NeuronIndx] = self.ActFunction(Vnet)

    def TestData(self):
        print("Testing is Starting.....")
        predicted = []
        for i in range(0, self.TestingData.shape[0]):
            predicted.append(self.test(self.TestingData[i]))
        pre = np.array(predicted)
        print("Accuracy : ", self.confusion(pre, self.TestingLabels))
        print("# Hidden Layers : ", self.NumberOfLayers)
        print("# Neurons : ", self.NumberOfNeurons[:])
        ep = np.array(self.Epochs)
        epoch = np.zeros((ep.shape[0], 1))
        for i in range(0, ep.shape[0]):
            epoch[i] = i + 1
        plt.plot(epoch, ep)
        plt.xlabel("epoch number ")
        plt.ylabel("MSE")
        plt.title("Learning Curve")
        plt.show()

    def CrossTest(self):
        predicted = []
        for i in range(0, self.CrossData.shape[0]):
            predicted.append(self.test(self.CrossData[i]))
        pre = np.array(predicted)
        return self.confusion(pre, self.CrossLabels)

    def Normalize(self):
        # Standardizing the features
        scaler = StandardScaler().fit(self.TrainingData)
        self.TrainingData = scaler.transform(self.TrainingData)
        self.TestingData = scaler.transform(self.TestingData)
        if self.CrossBool == 1:
            self.CrossData = scaler.transform(self.CrossData)
        # with open('TestingDataBeforePCA.txt', 'w') as outfile:
        # np.savetxt(outfile, self.TestingData, fmt='%-7.2f')
        #for i in range(0, 2500):
        #    mean = np.mean(self.TrainingData[:, i])
        #    max = np.max(self.TrainingData[:, i])
           # self.TrainingData[:, i] -= mean
            #self.TrainingData[:, i] /= 255
            #self.TestingData[:, i] /= 255

    def Read(self, train_path, test_path):
        # Reading Training Dat
        count = 0
        for each in glob(train_path + "*"):
            take = False
            st = ""
            for ch in each:
                if ch == ' ' and st != "":
                    break
                if ch == '.':
                    break
                if take and ch != ' ':
                    st += ch
                if ch == '-':
                    take = True
            im = cv.imread(each, 0)
            im = cv.resize(im, (50, 50))
            final_data = np.reshape(im, 2500)
            if self.CrossBool == 1 and (count == 0 or count == 1 or count == 6 or count == 8 or count == 17):
                self.CrossData.append(final_data)
                if st == "Cat":
                    self.CrossLabels.append(1)
                elif st == "Laptop":
                    self.CrossLabels.append(2)
                elif st == "Apple":
                    self.CrossLabels.append(3)
                elif st == "Car":
                    self.CrossLabels.append(4)
                elif st == "Helicopter":
                    self.CrossLabels.append(5)
            else:
                self.TrainingData.append(final_data)
                if st == "Cat":
                    self.TrainingLabels.append(1)
                elif st == "Laptop":
                    self.TrainingLabels.append(2)
                elif st == "Apple":
                    self.TrainingLabels.append(3)
                elif st == "Car":
                    self.TrainingLabels.append(4)
                elif st == "Helicopter":
                    self.TrainingLabels.append(5)
            count += 1
        self.TrainingData = np.array(self.TrainingData, dtype='float64')
        if self.CrossBool == 1:
            self.CrossData = np.array(self.CrossData, dtype='float64')
        # Reading Testing Data
        for each in glob(test_path + "*"):
            word = each.split("/")[-1]
            for imagefile in glob(test_path + word + "/*"):
                im = cv.imread(imagefile, 0)
                im = cv.resize(im, (50, 50))
                im = np.reshape(im, 2500)
                self.TestingData.append(im)
                self.TestingLabels.append(int(word))
        self.TestingData = np.array(self.TestingData, dtype='float64')

    def calculate_pca(self):
        if self.CrossBool == 1:
            pca = PCA(20)
        else:
            pca = PCA(25)
        pca.fit(self.TrainingData)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()
        self.TrainingData = pca.transform(self.TrainingData)
        self.TestingData = pca.transform(self.TestingData)
        if self.CrossBool == 1:
            self.CrossData = pca.transform(self.CrossData)

    @staticmethod
    def split_and_save(test_path):
        count = 0
        count2 = 0
        for each in glob(test_path + "*"):
            if count % 2 == 0:
                original_path = each
            else:
                all, pos = seg.segment(original_path, each)
                for element in all:
                    cv.imwrite("/home/abdlrhman/Desktop/NeuralNetworksProject-SemiFinalDelivery/TestFiles/" + str(count2) + each.split("/")[-1],
                               element)
                    count2 += 1
            count += 1


class TrainForm(QWidget):
    def __init__(self):
        super(TrainForm, self).__init__()
        self.title = 'Training'
        self.left = 10
        self.top = 10
        self.width = 500
        self.height = 500
        self.label1 = QLabel("Enter Number Of Hidden Layers :", self)
        self.label2 = QLabel("Enter Number Of Neurons in Each Hidden Layer :", self)
        self.label3 = QLabel("Enter learning Rate :", self)
        self.label4 = QLabel("Enter Number of Epochs :", self)
        self.label5 = QLabel("Choose The Activation Function Type :", self)
        self.label6 = QLabel("Choose The Stopping Criteria :", self)
        self.label7 = QLabel("Enter MSE Threshold :", self)
        self.label8 = QLabel("Choose a Classifier", self)
        self.CheckBox = QCheckBox("Bias", self)
        self.textboxHiddenLayers = QLineEdit(self)
        self.textboxNeuronsPerLayer = QLineEdit(self)
        self.textboxLr = QLineEdit(self)
        self.textboxEp = QLineEdit(self)
        self.textboxMSE = QLineEdit(self)
        self.button = QPushButton('Run', self)
        self.button.setToolTip('Run The Program')
        self.ActivationFunctionType = QComboBox(self)
        self.StoppingCriteria = QComboBox(self)
        self.Classifier = QComboBox(self)
        # input variables
        self.bias = 0
        self.b = 0
        self.NumberOfLayers = 1
        self.NumberOfNeurons = 1
        self.learning_rate = 0
        self.no_epochs = 1
        self.ActivationFunction = "Sigmoid"
        self.StoppingCondition = "Fix The Number Of Epochs"
        self.MSEThreshold = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.label8.setAlignment(Qt.AlignLeft)
        self.label8.move(10, 0)

        self.label1.setAlignment(Qt.AlignLeft)
        self.label1.move(10, 40)

        self.label2.setAlignment(Qt.AlignLeft)
        self.label2.move(10, 80)

        self.label3.setAlignment(Qt.AlignLeft)
        self.label3.move(10, 120)

        self.label4.setAlignment(Qt.AlignLeft)
        self.label4.move(10, 240)

        self.label5.setAlignment(Qt.AlignLeft)
        self.label5.move(10, 200)

        self.label6.setAlignment(Qt.AlignLeft)
        self.label6.move(10, 280)

        self.label7.setAlignment(Qt.AlignLeft)
        self.label7.move(30, 320)

        self.CheckBox.move(10, 160)

        self.textboxHiddenLayers.move(220, 40)
        self.textboxHiddenLayers.resize(40, 20)

        self.textboxNeuronsPerLayer.move(310, 80)
        self.textboxNeuronsPerLayer.resize(40, 20)

        self.textboxLr.move(180, 120)
        self.textboxLr.resize(40, 20)

        self.textboxEp.move(180, 240)
        self.textboxEp.resize(40, 20)

        self.button.move(200, 420)
        self.button.clicked.connect(self.on_click)

        self.ActivationFunctionType.move(250, 200)
        self.ActivationFunctionType.addItem("Sigmoid")
        self.ActivationFunctionType.addItem("Hyperbolic")

        self.StoppingCriteria.move(250, 280)
        self.StoppingCriteria.addItem("Fix The Number Of Epochs")
        self.StoppingCriteria.addItem("MSE")
        self.StoppingCriteria.addItem("Cross Validation")

        self.Classifier.move(200, 5)
        self.Classifier.addItem("MLP")
        self.Classifier.addItem("RBF")
        self.Classifier.currentTextChanged.connect(self.on_combobox_changed)
        self.textboxMSE.resize(40, 20)
        self.textboxMSE.move(350, 350)
        self.show()

    @pyqtSlot()
    def on_click(self):
        if self.Classifier.currentText() == "MLP":
            self.NumberOfLayers = self.textboxHiddenLayers.text()
            self.NumberOfNeurons = self.textboxNeuronsPerLayer.text()
            self.learning_rate = self.textboxLr.text()
            self.no_epochs = self.textboxEp.text()
            if self.CheckBox.isChecked():
                self.bias = 1
            else:
                self.bias = 0
            self.ActivationFunction = self.ActivationFunctionType.currentText()
            if self.StoppingCriteria.currentText() == "MSE":
                self.StoppingCondition = "MSE"
                self.MSEThreshold = self.textboxMSE.text()
            elif self.StoppingCriteria.currentText() == "Cross Validation":
                self.StoppingCondition = "Cross Validation"
                self.b = 1
            else:
                self.StoppingCondition = "Fix The Number Of Epochs"

            MyClass = MultiLayerNN(self.NumberOfLayers, self.NumberOfNeurons, self.bias, self.learning_rate,
                               self.no_epochs, self.ActivationFunction, self.StoppingCondition, self.MSEThreshold,self.b)
            MyClass.Start()
        else:
            self.NumberOfNeurons = self.textboxNeuronsPerLayer.text()
            self.learning_rate = self.textboxLr.text()
            self.no_epochs = self.textboxEp.text()
            self.MSEThreshold = self.textboxMSE.text()
            MyClass = RBF(22, 200)
            MyClass.test()
        self.hide()

    def on_combobox_changed(self):
        if self.Classifier.currentText() == "MLP":
            self.label2.setText("Enter Number Of Neurons in Each Hidden Layer :")
            self.textboxHiddenLayers.show()
            self.label1.show()
            self.CheckBox.show()
            self.label5.show()
            self.label6.show()
            self.label7.show()
            self.ActivationFunctionType.show()
            self.StoppingCriteria.show()
            self.textboxMSE.show()
        else:
            self.label2.setText("Enter Number Of Hidden Neurons :")
            self.textboxHiddenLayers.hide()
            self.label1.hide()
            self.CheckBox.hide()
            self.label5.hide()
            self.label6.hide()
            self.label7.hide()
            self.ActivationFunctionType.hide()
            self.StoppingCriteria.hide()
            self.textboxMSE.hide()


class TestForm(QWidget):
    def __init__(self):
        super(TestForm, self).__init__()
        self.title = 'Testing'
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 300
        self.test_path = "/home/abdlrhman/Desktop/NeuralNetworksProject-SemiFinalDelivery/Testing/"
        self.Classifier = QComboBox(self)
        self.label1 = QLabel("Choose a Classifier", self)
        self.label2 = QLabel("Choose The Original Image", self)
        self.label3 = QLabel("Choose The Colored Image", self)
        self.OriginalImagesCB = QComboBox(self)
        self.ColoredImagesCB = QComboBox(self)
        self.DisplayImages()
        self.CrossBool = 0
        self.Button1 = QPushButton('Test Model', self)
        self.Button1.setToolTip('Test The given Image')
        self.initUI()

    def initUI(self):
        self.Classifier.move(200, 5)
        self.Classifier.addItem("MLP")
        self.Classifier.addItem("RBF")

        self.label1.setAlignment(Qt.AlignLeft)
        self.label1.move(10, 5)

        self.label2.setAlignment(Qt.AlignLeft)
        self.label2.move(10, 50)

        self.label3.setAlignment(Qt.AlignLeft)
        self.label3.move(10, 100)

        self.OriginalImagesCB.move(200, 50)
        self.ColoredImagesCB.move(200, 100)

        self.Button1.move(200, 150)
        self.Button1.clicked.connect(self.on_click)

    def DisplayImages(self):
        count = 0
        for each in glob(self.test_path + "*"):
            if count % 2 == 0:
                original_path = each
            else:
                self.ColoredImagesCB.addItem(each.split("/")[-1])
                self.OriginalImagesCB.addItem(original_path.split("/")[-1])
            count += 1

    def on_click(self):
        labels = []
        if self.Classifier.currentText() == "MLP":
            self.MyClass = MultiLayerNN()
            self.MyClass.ReadTrainingData()
            print(self.MyClass.TrainingData.shape)
            scaler = StandardScaler().fit(self.MyClass.TrainingData)
            self.MyClass.TrainingData = scaler.transform(self.MyClass.TrainingData)
            pca = PCA(25)
            pca.fit(self.MyClass.TrainingData)
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
            plt.show()
            im = cv.imread(self.test_path+self.OriginalImagesCB.currentText())
            all, pos = seg.segment(self.test_path+self.OriginalImagesCB.currentText(), self.test_path+self.ColoredImagesCB.currentText())
            self.MyClass.ReadWeightAndBias()
            for image in all:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                image = cv.resize(image, (50, 50))
                image = np.reshape(image, 2500)
                image = scaler.transform([image])
                image = pca.transform(image)
                ret = self.MyClass.test(image)
                if ret == 1:
                    labels.append("Cat")
                elif ret == 2:
                    labels.append("Laptop")
                elif ret == 3:
                    labels.append("Apple")
                elif ret == 4:
                    labels.append("Car")
                else:
                    labels.append("Helicopter")
            for j in range(len(pos)):  # iterate over objects
                # pass each of all list to the classifier
                cv.rectangle(im, (pos[j][0], pos[j][2]), (pos[j][1], pos[j][3]), (255, 0, 0), 3)
                font = cv.FONT_HERSHEY_SIMPLEX
                label = str(labels[j])
                cv.putText(im, label, (pos[j][0], pos[j][2]), font, 0.7, (0, 255, 0), 2, cv.LINE_AA)
            cv.imshow("test", im)
        else:
            self.MyClass = RBF(22, 200)
            self.MyClass.ReadTrainingData()
            scaler = StandardScaler().fit(self.MyClass.TrainData)
            self.MyClass.TrainData = scaler.transform(self.MyClass.TrainData)
            pca = PCA(25)
            pca.fit(self.MyClass.TrainData)
            plt.plot(np.cumsum(pca.explained_variance_ratio_))

            im = cv.imread(self.test_path + self.OriginalImagesCB.currentText())
            all, pos = seg.segment(self.test_path + self.OriginalImagesCB.currentText(),
                                   self.test_path + self.ColoredImagesCB.currentText())
            for image in all:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                image = cv.resize(image, (50, 50))
                image = np.reshape(image, 2500)
                image = scaler.transform([image])
                image = pca.transform(image)
                # image now contain the 25 features
                ret =self.MyClass.test_sample(image)# ismail function :D
                if ret == 1:
                    labels.append("Cat")
                elif ret == 2:
                    labels.append("Laptop")
                elif ret == 3:
                    labels.append("Apple")
                elif ret == 4:
                    labels.append("Car")
                else:
                    labels.append("Helicopter")
            for j in range(len(pos)):  # iterate over objects
                # pass each of all list to the classifier
                cv.rectangle(im, (pos[j][0], pos[j][2]), (pos[j][1], pos[j][3]), (255, 0, 0), 3)
                font = cv.FONT_HERSHEY_SIMPLEX
                label = str(labels[j])
                cv.putText(im, label, (pos[j][0], pos[j][2]), font, 0.7, (0, 255, 0), 2, cv.LINE_AA)
            cv.imshow("test", im)


class MainForm(QWidget):
    def __init__(self):
        super(MainForm, self).__init__()
        self.title = 'Neural Network Project'
        self.left = 10
        self.top = 10
        self.width = 400
        self.height = 200
        self.TrainButton = QPushButton('Train', self)
        self.TrainButton.setToolTip('Train The Model')
        self.TestButton = QPushButton('Test', self)
        self.TestButton.setToolTip('Test The Model')
        self.initUI()

    def initUI(self):
        self.TrainButton.move(200, 100)
        self.TrainButton.clicked.connect(self.Train_on_click)
        self.TestButton.move(100, 100)
        self.TestButton.clicked.connect(self.Test_on_click)
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    @pyqtSlot()
    def Train_on_click(self):
        self.hide()
        self.FT = TrainForm()
        self.FT.show()

    def Test_on_click(self):
        self.hide()
        self.FT = TestForm()
        self.FT.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainForm()
    sys.exit(app.exec_())
