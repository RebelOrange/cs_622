import sys

import matplotlib.pyplot as plt

from ModelEvaluator import ModelEvaluator
from NNModel import NNModel
from DataManager import *
# Import matplotlib for plotting
from PlotUltilities import PlotPiePlot, PlotConfusionMatrix, PlotKFoldConfusionMatrices

if __name__ == "__main__":
    TRAIN_MODEL = True
    LOAD_BEST_MODEL = False
    USE_KFOLD = True

    ######################################### Load Data #########################################################
    #### Load full set, plot and save, then load actual set ####
    dm = DataManager()
    numFilesList = [120]
    classes = ["sitting", "running", "drinking","eating"]#, "listening_to_music"]
    dataSplit = 0.8

    for numFiles in numFilesList:
        dm.ResetData()
        classFilter = None
        if numFiles is not None:
            classFilter = classes
        dm.LoadTrainAndTestData(
            folderName="../data/",
            csvFileName="Training_set.csv",
            numFiles=numFiles,
            classFilter=classFilter,
            split=dataSplit)
        dm.RemoveMissingData()
        dm.ResizeImages(TargetSize=(224, 224))
        dm.NormalizeImages()
        dm.SetClassNames()

    ##### Data Plots ######
    # Pie plot of data distribution, subplot of distribution of training and testing with split and number of files as
    # the title
        dm.PlotDataDistrobution(dm.TrainingData, dm.TestData)
        pass

    # plot example images with labels in a 2x3 layout
        dm.ShowRandomImages(numImages=6, showGrayscale=False, showSegmented=False)

    ######################################### init Single Model ################################################
    modelType = "ResNet"
    num_classes = len(dm.TrainingData["label"].unique())
    model = NNModel(num_classes=num_classes, variant='b0', model_dir="../models/Training", modelType=modelType, optimizer="Adam",
                    learningRate=0.0001)
    model.Preprocess(dm.TrainingData)
    model.SetOptimizer(
        optimizer="Adam",
        learningRate=0.0001)


    ######################################### Train Model #######################################################
    k = 5
    best_accuracy = 0
    best_prediction = 0
    k_labels = []
    k_predictions = []
    if TRAIN_MODEL:
        if USE_KFOLD:
            for i in range(k):
                print(f"Training fold {i+1} of {k}...")
                model.ResetModel()
                model.SetOptimizer(optimizer="Adam", learningRate=0.0001)
                model.SetupClassMapping(dm.TrainingData)
                dm.SplitKFold(k=k, foldIndex=i)
                if i == 0:
                    print("Plotting distribution of training and testing data...")
                    dm.PlotDataDistrobution(dm.KTrainingData, dm.KTestData)
                kFoldStats = model.Train(
                    dm.KTrainingData,
                    epochs=20,
                    batch_size=32,
                    save_interval=1,
                    load_model=LOAD_BEST_MODEL,
                    save_model=False,
                    target_accuracy=100,
                    show_epoch_stats=False)

                labels = dm.GetKTestLabels()
                images = dm.GetKTestImages()

                k_labels.append(labels)

                # predict images
                predictions = []
                for image in images:
                    predictions.append(model.Predict(image))
                k_predictions.append(predictions)

                correct = sum(1 for a, p in zip(labels, predictions) if a == p)
                accuracy = 100 * correct / len(labels)
                print(f"\n{i}-fold accuracy: {accuracy:.2f}%")


                actual_counts = pd.Series(labels).value_counts(normalize=True) * 100
                predicted_counts = pd.Series(predictions).value_counts(normalize=True) * 100

                fig, axes = PlotPiePlot(actual_counts, predicted_counts)
                axes[0].set_title(f"Fold Actual Labels")
                axes[1].set_title(f"Predicted Labels")
                plt.show()

                fig, axes = PlotConfusionMatrix(labels, predictions, dm.ClassNames)
                axes.set_title(f"{i}-fold Confusion Matrix")
                plt.show()


                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_prediction = predictions
                    model.SaveModel(epoch=kFoldStats[-1][0], loss=kFoldStats[-1][2], model_name="ResNet")
                    print("Saved best model")

            fig, axes = PlotKFoldConfusionMatrices(k_labels, k_predictions, dm.ClassNames)
            print(f"Best accuracy: {best_accuracy:.2f}%")
            plt.show()
            #print(f"Best predictions: {best_prediction}")
        else:
            print("No K-Fold")
            model.Train(
                dm.TrainingData,
                epochs=20,
                batch_size=32,
                save_interval=1,
                load_model=LOAD_BEST_MODEL,
                save_model=True,
                target_accuracy=100,
                show_epoch_stats=True)
        pass


    if LOAD_BEST_MODEL:
        model.LoadModel("ResNet")
        model.SetupClassMapping(dm.TrainingData)
        pass
    ######################################## Evaluation #########################################################
    dm.SetClassNames()

    # setup labels
    labels = dm.GetTestLabels()
    images = dm.GetTestImages()

    # predict images
    predictions = []
    for image in images:
        predictions.append(model.Predict(image))
    #predictions = model.PredictBatch(images)

    # Create pie plots for predicted and actual labels
    actual_counts = pd.Series(labels).value_counts(normalize=True) * 100
    predicted_counts = pd.Series(predictions).value_counts(normalize=True) * 100

    PlotPiePlot(actual_counts, predicted_counts, title="Actual vs Predicted Labels Distribution")

    print("\nBatch Prediction Results:")
    print("-------------------------")
    #for i, (actual, predicted) in enumerate(zip(labels, predictions)):
    #    status = "v" if actual == predicted else "x"
    #    print(f"Sample {i+1}: Actual: {actual}, Predicted: {predicted} {status}")

    # Accuracy against test set
    correct = sum(1 for a, p in zip(labels, predictions) if a == p)
    accuracy = 100 * correct / len(labels)
    print(f"\nBatch accuracy: {accuracy:.2f}%")

    # Confusion Matrix
    PlotConfusionMatrix(labels, predictions, dm.ClassNames)









