import pandas as pd
from DataManager import *
import matplotlib.pyplot as plt
from NNModel import NNModel
import csv


def WriteCsv(data, csv_file_name):
    """Save training statistics to a CSV file."""
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Epoch", "Time", "Loss", "Accuracy"])
        # Write each row of data
        for row in data:
            writer.writerow(row)

if __name__ == "__main__":

    print("########################### Loading training data... #################################")
    dm = DataManager()
    current_folder = os.getcwd()
    classes = ["sitting", "running", "drinking","eating"]
    dm.LoadTrainingData(folderName=current_folder + "/../data/", csvFileName="Training_set.csv", numFiles=None, classFilter=classes)

    print("####################### Preprocessing data... #######################################")
    dm.RemoveMissingData()
    # maybe using tranformations from pytorch could be better?? if so maybe we can do that in preprocess()
    dm.ResizeImages(TargetSize=(224, 224))
    dm.NormalizeImages()

    print("####################### Configure Models #######################################")
    num_classes = len(dm.TrainingData["label"].unique())
    model_dir = os.path.join(current_folder, "../models/Training")
    os.makedirs(model_dir, exist_ok=True)

    modelType = "ResNet"
    print(f"Test 3: Initializing {modelType} model...")
    ResNet = NNModel(num_classes=num_classes, variant='b0', model_dir=model_dir, modelType=modelType, optimizer="Adam",
                    learningRate=0.001)
    ResNet.Preprocess(dm.TrainingData)

    modelType = "EfficientNet"
    print(f"Test 3: Initializing {modelType} model...")
    EffNet = NNModel(num_classes=num_classes, variant='b0', model_dir=model_dir, modelType=modelType, optimizer="Adam",
                    learningRate=0.001)
    EffNet.Preprocess(dm.TrainingData)

    print("########################### Initial Training ###################################")
    print("Training to 90% accuracy with Adam 0.1 learning rate...")
    ResNetStats = ResNet.Train(dm.TrainingData, epochs=30, batch_size=32, save_interval=1, load_model=False, save_model=True, target_accuracy=90)

    csv_file_name = f"ResNet_Training_90_stats.csv"
    WriteCsv(ResNetStats, csv_file_name)
    print("########################### Final Training #####################################")
    ResNetStatsFinal = {}
    learning_rates = [0.0001,0.001, 0.01, 0.1]
    n_epochs = 25

    
    for learning_rate in learning_rates:
        print(f"Training {n_epochs} epochs or 100% accuracy with Adam {learning_rate} learning rate...")
        ResNet.SetOptimizer(optimizer="Adam", learningRate=learning_rate)
        stats = ResNet.Train(dm.TrainingData, epochs=n_epochs, batch_size=32, save_interval=1, load_model=True, save_model=False, target_accuracy=100)
        ResNetStatsFinal[learning_rate] = stats
    
        # Write stats to CSV
        csv_file_name = f"ResNet_Training_{learning_rate}_stats.csv"
        WriteCsv(stats, csv_file_name)

    



