import os

from NNModel import NNModel
from DataManager import *
from PlotUltilities import *

if __name__ == "__main__":

    #### load data ####
    classFilt = ["sitting", "running", "drinking", "eating"]
    currDir = os.getcwd()
    modelDir = os.path.join(currDir, "../models")
    dm = DataManager()
    dm.LoadTrainAndTestData(
        folderName="../data/",
        csvFileName="Training_set.csv",
        numFiles=50,
        classFilter=classFilt,
        split=0.8)
    dm.RemoveMissingData()
    dm.ResizeImages(TargetSize=(128, 128))
    dm.NormalizeImages()
    dm.SetClassNames()

    #### create model  ####
    numOfCLasses = len(dm.TrainingData["label"].unique())
    model = NNModel(
        num_classes=numOfCLasses,
        variant='b0',
        model_dir=modelDir,
        modelType="EfficientNet",
        optimizer="Adam",
        learningRate=0.001)
    model.Preprocess(dm.TrainingData)

    epochStats = model.Train(dm.TrainingData, epochs=5, batch_size=8, save_interval=2, load_model=True)

    model.PlotEpochStats(epochStats)

    #### test model ####
    correctPredictions = 0
    totalPredictions = len(dm.TestData)

    images = dm.GetTestImages()
    labels = dm.GetTestLabels()

    prediction = None
    for image in images:
        for label in labels:
            prediction = model.Predict(image)
            if prediction == label:
                correctPredictions += 1

    acc = (correctPredictions / totalPredictions) * 100
    print(f"Accuracy: {acc:}%")