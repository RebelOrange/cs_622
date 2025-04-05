import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os

from sympy.stats import moment

from Model import Model
from DataManager import DataFrameImage, DataManager
from ModelManager import ModelManager

class DavidNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden1 = nn.Linear(225, 8)
        self.hidden2 = nn.Linear(8, 4)
        self.output = nn.Linear(4, 1)
        self.Relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

################################## Network Funcitons ######################################
    ############# Build Network ######################
    def BuildNN(self, inputSize=1, labels=range(10)):
        self.Network = [nn.Conv2d(inputSize, 64, 3), nn.ReLU(),
                   nn.Conv2d(64, 64, 3), nn.ReLU(),
                   nn.Conv2d(64, 64, 3), nn.ReLU(),
                   nn.Conv2d(64, len(labels), 3), nn.ReLU()]

        pass

    def BuildModule1(self, inputSize=1, outputSize=5):
        conv2d = nn.Conv2d(in_channels=inputSize, out_channels=outputSize, kernel_size=3, padding=1)
        batchNorm = nn.BatchNorm2d(num_features=outputSize)
        activation = nn.ReLU()

        self.module_1 = [conv2d, batchNorm, activation]

    def BuildModule2(self, inputSize=5, outputSize=5):
        conv2d_1 = nn.Conv2d(in_channels=inputSize, out_channels=outputSize, kernel_size=3, padding=1)
        batchNorm = nn.BatchNorm2d(num_features=outputSize)
        activation = nn.ReLU()
        zeroPad = nn.ZeroPad2d(padding=1)
        conv2d_2 = nn.Conv2d(in_channels=outputSize, out_channels=outputSize, kernel_size=3, padding=1)

        self.module_2 = [conv2d_1, batchNorm, activation, zeroPad, conv2d_2, batchNorm, activation]


    def BuildModule3(self, inputSize=5, outputSize=3):
        avgPool = nn.AvgPool2d(kernel_size=2, stride=2)
        # add rescale?
        conv2d_1 = nn.Conv2d(in_channels=inputSize, out_channels=outputSize, kernel_size=3, padding=1)
        conv2d_2 = nn.Conv2d(in_channels=outputSize, out_channels=outputSize, kernel_size=3, padding=1)

        self.module_3 = [avgPool, conv2d_1, conv2d_2]

    def ExecuteModules(self, modules, x):
        for mod in modules:
            x = mod(x)

        return x

    def forward(self, x):
        out = self.ExecuteModules(modules=self.Network, x=x)


################################## Base Class Override ###################################
    ############ Train Network ######################
    def Train(self, df):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)


        pass

    def TrainEpoch(self, batchDF):
        loss = 0
        last_loss = 0

        for data in BatchData:
            inputs, labels =


    ############ Predict #############################
    def Predict(self, x):
        x = self.hidden1(x)
        x = self.Relu(x)
        x = self.hidden2(x)
        x = self.Relu(x)
        x = self.output(x)
        x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    model = DavidNet()
    model.BuildNN()
    print(model.parameters())


    print("Test 1: Loading training data...")
    dm = DataManager()
    current_folder = os.getcwd()
    dm.LoadTrainingData(folderName=current_folder + "/../data/", csvFileName="Training_set.csv", numFiles=100)

    print("Test 2: Preprocessing data...")
    dm.RemoveMissingData()
    # maybe using tranformations from pytorch could be better?? if so maybe we can do that in preprocess()
    dm.ResizeImages(TargetSize=(224, 224))
    dm.NormalizeImages()
    dm.ConvertToGrayScale()

    num_classes = len(dm.TrainingData["label"].unique())
    # dm.PrintStats()

    # create test folder to save best model to for save/load model test
    model_dir = os.path.join(current_folder, "../models")
    os.makedirs(model_dir, exist_ok=True)

    print("Test 3: Initializing EfficientNet model...")
    model = DavidNet()
    model.BuildNN(inputSize=dm.ImageSize, labels=dm.TrainingData["label"].unique())
    model.Preprocess(dm.TrainingData)  # preprocess the data again

    # might need to do split data or cross validation to get better results?

    print("Test 4: Training model...")
    model.Train(dm.TrainingData, epochs=10, batch_size=8, save_interval=2, load_model=False)

    print("Test 5: Testing batch prediction...")
    test_batch = dm.TrainingData.sample(10)
    test_images = test_batch["image"]
    test_labels = test_batch["label"]
    predicted_labels = model.Predict(test_images)

    print("\nBatch Prediction Results:")
    print("-------------------------")
    for i, (actual, predicted) in enumerate(zip(test_labels, predicted_labels)):
        status = "v" if actual == predicted else "x"
        print(f"Sample {i + 1}: Actual: {actual}, Predicted: {predicted} {status}")

    # Calculate accuracy
    correct = sum(1 for a, p in zip(test_labels, predicted_labels) if a == p)
    accuracy = 100 * correct / len(test_labels)
    print(f"\nBatch accuracy: {accuracy:.2f}%")