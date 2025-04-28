import os
import torch
from NNModel import NNModel
from DataManager import *
from PlotUltilities import *
from sklearn.model_selection import KFold

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
    dm.ResizeImages()
    dm.NormalizeImages()
    dm.SetClassNames()

    #### create model  ####
    numOfCLasses = len(dm.TrainingData["label"].unique())
    model = NNModel(
        num_classes=numOfCLasses,
        variant='b0',
        model_dir=modelDir,
        modelType="EfficientNet",
        optimizer="SGD",
        learningRate=0.01)
    model.Preprocess(dm.TrainingData)

    epochStats = model.Train(dm.TrainingData, epochs=5, batch_size=16, save_interval=2, load_model=True)

    model.PlotEpochStats(epochStats)

    data = dm.TrainingData.reset_index(drop=True)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []

    for fold, (trainingIndex, validationIndex) in enumerate(kf.split(data)):
        print(f"\n***** Fold {fold + 1} of 10 *****")

        trainingData = data.iloc[trainingIndex]
        validationData = data.iloc[validationIndex]

        # Create and train model
        model = NNModel(
            num_classes=len(dm.ClassNames),
            variant='b0',
            model_dir=modelDir,
            modelType="EfficientNet",
            optimizer="SGD",
            learningRate=0.01
        )
        model.Preprocess(trainingData)
        model.Train(trainingData, epochs=5, batch_size=8, save_interval=2, load_model=False)

        ##### Evaluate #####
        kfoldImages = validationData["image"].tolist()
        kfoldLabels = validationData["label"].tolist()
        correct = 0

        preds = []
        for img in kfoldImages:
            preds.append(model.Predict(img))

        # calculate accuracy
        correct = 0
        for pred, kfoldLabels in zip(preds, kfoldLabels):
            if pred == kfoldLabels:
                correct += 1
        acc = 100 * correct / len(kfoldLabels)
        accuracies.append(acc)
        print(f"***** Fold {fold + 1} Accuracy: {acc:.2f}% *****")

    avg_acc = np.mean(accuracies)
    print(f"\n***** Average 10-Fold Accuracy: {avg_acc:.2f}% *****")

    ##### Test Model #####

    # get labels for testing data
    imgs = dm.GetTestImages()
    labels = dm.GetTestLabels()

    # make predictions on testing data
    preds = []
    for img in imgs:
        preds.append(model.Predict(img))

    # calculate accuracy
    correct = 0
    for pred, label in zip(preds, labels):
        if pred == label:
            correct += 1
    acc = 100 * correct / len(labels)
    print(f"\nAccuracy of Testing Set: {acc:.2f}%")



