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

def PlotTrainingProgress(folder, additional_csvs, model,optimizer,save_plot=False):
    """
    Plot the training progress for the ResNet model by combining data across multiple training phases.

    Parameters:
    1. main_csv (str): The path to the first CSV file (e.g., ResNet_Training_90_stats.csv).
    2. additional_csvs (list of str): Paths to other CSV files generated during additional training phases.
    3. save_plot (bool): If True, saves the plot as a PNG file, otherwise displays it.
    """
    # Read the main CSV file

    # Initialize plot
    plt.figure(figsize=(10, 6))
    #plt.plot(main_epochs, main_data["Accuracy"], label="Main Run Accuracy", color="blue")
    #plt.plot(main_epochs, main_data["Loss"], label="Main Run Loss", color="red", linestyle="--")

    # Process additional CSV files
    for csv_file in additional_csvs:
        print("loading CSV: {}".format(folder+csv_file))
        try:
            additional_data = pd.read_csv(folder+csv_file)
            additional_epochs = [epoch for epoch in additional_data["Epoch"]]
            #last_epoch = additional_epochs[-1]  # Update last_epoch after processing each file
            plt.plot(additional_epochs, additional_data["Accuracy"], label=f"{csv_file} Accuracy", linestyle="--")
            #plt.plot(additional_epochs, additional_data["Loss"], label=f"{csv_file} Loss", linestyle=":")
        except FileNotFoundError:
            print("CSV file not found: {}".format(folder+csv_file))

    # Customize and display/save the plot
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model} Training Progress")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.ylim(0, 100)
    if save_plot:
        plt.savefig(folder + f"Training_{model}_{optimizer}.png")
    else:
        plt.show()


def PlotAllBestTrainingProgress(folder, best_csvs, save_plot=False):
    """
    Plot all the best CSV data into a single figure with proper labeling.

    Parameters:
    1. folder (str): Folder containing the CSV files.
    2. best_csvs (dict): Dictionary with keys as (model, optimizer) and values as the best CSV files.
    3. save_plot (bool): If True, saves the combined plot as a PNG file.
    """
    plt.figure(figsize=(12, 8))
    print(f"Plotting all best CSV data from {folder}")

    for (model, optimizer, lr), csv in best_csvs.items():
        if model == "ResNet":
            lineStyle = "--"
        else:
            lineStyle = "-"
        try:
            data = pd.read_csv(folder + csv)
            epochs = data["Epoch"]
            accuracy = data["Accuracy"]
            plt.plot(epochs, accuracy, label=f"{model}-{optimizer} (LR={lr})", linestyle=lineStyle)
        except FileNotFoundError:
            print(f"Could not load CSV: {folder + csv}")

    # Customize and save/display the plot
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Best Training Trajectory Across Models and Optimizers (500 Training Files)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.ylim(0, 100)
    if save_plot:
        plt.savefig(folder + "All_Models_Optimizers_Training.png")
    else:
        plt.show()


def FindBestLearningRate(folder, models, optimizers, learning_rates):
    """
    Find the best learning rate for each model and optimizer based on time to converge near 100% accuracy.

    Parameters:
    1. folder (str): Path to the folder containing CSV files.
    2. models (list of str): List of model names.
    3. optimizers (list of str): List of optimizer names.
    4. learning_rates (list of float): List of learning rates used.

    Returns:
    dict: Dictionary with the best CSV file for each model-optimizer combination.
    """
    best_csv_files = {}
    for model in models:
        for optimizer in optimizers:
            best_lr = None
            min_time_to_converge = float("inf")
            best_csv = None

            for lr in learning_rates:
                csv_file = f"{model}_Training_{optimizer}_{lr}_stats.csv"
                try:
                    data = pd.read_csv(folder + csv_file)
                    if "Accuracy" in data.columns and "Time" in data.columns:
                        time_to_converge = data.loc[data["Accuracy"] >= 99.9, "Time"].min()  # Assuming 99.9% as near 100%

                        if pd.notna(time_to_converge) and time_to_converge < min_time_to_converge:
                            min_time_to_converge = time_to_converge
                            best_lr = lr
                            best_csv = csv_file
                except FileNotFoundError:
                    print(f"CSV not found: {folder + csv_file}")

            if best_csv is not None:
                best_csv_files[(model, optimizer, best_lr)] = best_csv
                print(f"Best CSV for {model} with {optimizer} is {best_csv} (LR={best_lr})")

    return best_csv_files


if __name__ == "__main__":
    ...
    TRAIN_MODEL = False
    print("########################### Loading training data... #################################")
    dm = DataManager()
    current_folder = os.getcwd()
    classes = ["sitting", "running", "drinking","eating", "listening_to_music"]
    dm.LoadTrainAndTestData(folderName=current_folder + "/../data/", csvFileName="Training_set.csv", numFiles=100, classFilter=classes)

    print("####################### Preprocessing data... #######################################")
    dm.RemoveMissingData()
    # maybe using tranformations from pytorch could
    # be better?? if so maybe we can do that in preprocess()
    dm.ResizeImages(TargetSize=(224, 224))
    dm.NormalizeImages()

    print("####################### Configure Models #######################################")
    num_classes = len(dm.TrainingData["label"].unique())
    model_dir = os.path.join(current_folder, "../models/Training")
    os.makedirs(model_dir, exist_ok=True)

    modelType = "ResNet"
    print(f"Test 3: Initializing {modelType} model...")
    ResNet = NNModel(num_classes=num_classes, variant='b0', model_dir=model_dir, modelType=modelType, optimizer="Adam",
                    learningRate=0.0001)
    ResNet.Preprocess(dm.TrainingData)

    modelType = "EfficientNet"
    print(f"Test 3: Initializing {modelType} model...")
    EffNet = NNModel(num_classes=num_classes, variant='b0', model_dir=model_dir, modelType=modelType, optimizer="Adam",
                    learningRate=0.001)
    EffNet.Preprocess(dm.TrainingData)

    print("########################### Initial Training ###################################")
    print("Training to 90% accuracy with Adam 0.1 learning rate...")
    #ResNetStats = ResNet.Train(dm.TrainingData, epochs=30, batch_size=32, save_interval=1, load_model=False, save_model=True, target_accuracy=90, show_epoch_stats=True)

    csv_file_name = f"ResNet_Training_90_stats.csv"
    #WriteCsv(ResNetStats, csv_file_name)
    print("########################### Loop Training #####################################")
    ResNetStatsFinal = {}
    EffNetStatsFinal = {}
    TRAIN_MODEL = False

    csvFolder = "..//models//Training//csv_data//learning_rate_traj//"
    models = ["ResNet", "EfficientNet"]
    optimizers = ["Adam", "SGD", "Adadelta"]

    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    n_epochs = 40
    for model in models:
        for optimizer in optimizers:


            for learning_rate in learning_rates:
                if TRAIN_MODEL:
                    print(f"Training {n_epochs} epochs or 100% accuracy with {optimizer} {learning_rate} learning rate...")
                    if model == "ResNet":
                        ResNet.SetOptimizer(optimizer=optimizer, learningRate=learning_rate)
                        stats = ResNet.Train(dm.TrainingData, epochs=n_epochs, batch_size=32, save_interval=1, load_model=False, save_model=False, target_accuracy=100, show_epoch_stats=False)
                        ResNetStatsFinal[learning_rate] = stats
                        ResNet.ResetModel()
                    if model == "EfficientNet":
                        EffNet.SetOptimizer(optimizer=optimizer, learningRate=learning_rate)
                        stats = EffNet.Train(dm.TrainingData, epochs=n_epochs, batch_size=32, save_interval=1, load_model=False, save_model=False, target_accuracy=100, show_epoch_stats=False)
                        EffNetStatsFinal[learning_rate] = stats
                        EffNet.ResetModel()

                    # Write stats to CSV
                    csv_file_name = f"{model}_Training_{optimizer}_{learning_rate}_stats.csv"
                    WriteCsv(stats, csvFolder+csv_file_name)



        #try:
        # Plot training stats from CSV files
            main_training_csv = "ResNet_Training_90_stats.csv"

            additional_training_csvs = [f"{model}_Training_{optimizer}_{lr}_stats.csv" for lr in [0.1, 0.01, 0.001, 0.0001]]
            PlotTrainingProgress(csvFolder, additional_training_csvs,model, optimizer,save_plot=True)
    #except FileNotFoundError:
     #   print("Training Stats CSV does not exist. Skipping plotting.")
    
    
    print(f"########################### Plot Training Data ##########################################")

    # Find best learning rates for plotting
    csvFolder = "../models/training/csv_data/learning_rate_traj/"
    models = ["ResNet", "EfficientNet"]
    optimizers = ["Adam", "SGD", "Adadelta"]
    learning_rates = [0.1, 0.01, 0.001, 0.0001]

    best_csvs = FindBestLearningRate(csvFolder, models, optimizers, learning_rates)
    print(f"Best CSVs: {best_csvs}")


    # Plot all best training progress in a single figure
    PlotAllBestTrainingProgress(csvFolder, best_csvs, save_plot=False)
        
    
    



