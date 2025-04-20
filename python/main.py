from NNModel import NNModel
from DataManager import *


if __name__ == "__main__":
    TRAIN_MODEL = True
    LOAD_BEST_MODEL = False
    USE_KFOLD = True

    ######################################### Load Data #########################################################
    #### Load full set, plot and save, then load actual set ####
    dm = DataManager()
    numFilesList = [150]
    classes = ["sitting", "running", "drinking","eating", "listening_to_music"]
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

                # predict images
                predictions = []
                for image in images:
                    predictions.append(model.Predict(image))

                correct = sum(1 for a, p in zip(labels, predictions) if a == p)
                accuracy = 100 * correct / len(labels)
                print(f"\n{i}-fold accuracy: {accuracy:.2f}%")


                actual_counts = pd.Series(labels).value_counts(normalize=True) * 100
                predicted_counts = pd.Series(predictions).value_counts(normalize=True) * 100

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].pie(actual_counts, labels=actual_counts.index, autopct='%1.1f%%')
                axes[0].set_title(f"{i} Fold Actual Labels Distribution \n Accuracy = {kFoldStats[-1][3]}")
                axes[1].pie(predicted_counts, labels=predicted_counts.index, autopct='%1.1f%%')
                axes[1].set_title(f"{i} Fold Predicted Labels Distribution \n Accuracy = {accuracy}")
                plt.tight_layout()
                plt.show()


                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_prediction = predictions
                    model.SaveModel(epoch=kFoldStats[-1][0], loss=kFoldStats[-1][2], model_name="ResNet")
                    print("Saved best model")


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
    # Import matplotlib for plotting
    import matplotlib.pyplot as plt
    
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].pie(actual_counts, labels=actual_counts.index, autopct='%1.1f%%')
    axes[0].set_title("Actual Labels Distribution")
    axes[1].pie(predicted_counts, labels=predicted_counts.index, autopct='%1.1f%%')
    axes[1].set_title("Predicted Labels Distribution")
    plt.tight_layout()
    plt.show()

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








