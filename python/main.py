import os
from DataManager import DataManager
from NN_Model import NNModel
from ModelEvaluator import ModelEvaluator, plotAllVariantsTrainingCurves

# ===================== GLOBAL CONFIGURATION =====================
CURRENT_DIR = os.getcwd()
DATA_DIR = os.path.join(CURRENT_DIR, "../data/")
MODEL_DIR = os.path.join(CURRENT_DIR, "../models")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../output")

CLASSES = ["sitting", "running", "drinking", "eating"]
NUM_FILES = 100
NUM_EPOCHS = 3
BATCH_SIZE = 8
MAX_CONFIGS = 9
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TARGET_ACCURACY = 100.0

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


if __name__ == "__main__":
    # ===================== DATA LOADING =====================
    print("\nLoading training data...")
    dm = DataManager()
    dm.LoadTrainingData(folderName=DATA_DIR, csvFileName="Training_set.csv", numFiles=NUM_FILES, classFilter=CLASSES)
    dm.RemoveMissingData()
    num_classes = len(dm.TrainingData["label"].unique())
    print(f"Loaded {len(dm.TrainingData)} samples with {num_classes} classes")

    # ===================== MODEL EVALUATION =====================
    evaluator = ModelEvaluator()

    architectures = [
        {'class': NNModel, 'model_type': 'EfficientNet', 'variant': 'b0', 'prefix': 'NNEfficientNet'},
        {'class': NNModel, 'model_type': 'ResNet', 'variant': '18', 'prefix': 'NNResNet'},
    ]

    results = evaluator.evaluateArchitectures(
        trainingData=dm.TrainingData,
        architectures=architectures,
        modelDir=MODEL_DIR,
        numEpochs=NUM_EPOCHS,
        batchSize=BATCH_SIZE,
        maxConfigs=MAX_CONFIGS,
        outputDir=os.path.join(OUTPUT_DIR, "architecture_comparison"),
        loadBest=True,
        saveBest=True,
        targetAccuracy=TARGET_ACCURACY
    )

    # ===================== COMPARISON & EXPORT =====================
    print("\nComparing best models across architectures...")
    evaluator.compareModels()
    evaluator.compareBestAcrossModelTypes(
        path=os.path.join(OUTPUT_DIR, "best_architecture_comparison"),
        saveCsv=True
    )

    print("\nExporting confusion matrices and test results...")
    evaluator.exportAllConfusionMatrices(OUTPUT_DIR, saveCsv=True)
    evaluator.exportTestResults(os.path.join(OUTPUT_DIR, "test_results.csv"), saveCsv=True)

    print("\nPlotting ROC curves for all models...")
    for modelName in evaluator.results:
        evaluator.plotRocCurve(
            modelName,
            path=os.path.join(OUTPUT_DIR, f"{modelName}_roc"),
            saveCsv=True
        )

    print("\nPlotting confusion matrices for all models...")
    for modelName in evaluator.confusionMatrices:
        evaluator.plotConfusionMatrix(
            modelName,
            path=os.path.join(OUTPUT_DIR, f"{modelName}_cm"),
            saveCsv=True
        )

    print("\nSaving and plotting epoch stats for all models/configs...")
    from Trainer import WriteCsv
    import csv
    for modelName, model in evaluator.models.items():
        epochStats = None
        if modelName in evaluator.results and 'epochStats' in evaluator.results[modelName]:
            epochStats = evaluator.results[modelName]['epochStats']
        elif hasattr(model, 'epochStats'):
            epochStats = getattr(model, 'epochStats')
        elif hasattr(model, 'trainingHistory'):
            epochStats = getattr(model, 'trainingHistory')
        csvFile = os.path.join(OUTPUT_DIR, f"{modelName}_epoch_stats.csv")
        csvExists = os.path.isfile(csvFile)
        if epochStats and len(epochStats) > 0 and not csvExists:
            WriteCsv(epochStats, csvFile)
        elif epochStats and len(epochStats) > 0 and csvExists:
            existingStats = []
            with open(csvFile, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    existingStats.append([int(row[0]), float(row[1]), float(row[2]), float(row[3])])
            lastEpoch = existingStats[-1][0] if existingStats else 0
            newStats = [row for row in epochStats if row[0] > lastEpoch]
            allStats = existingStats + newStats
            WriteCsv(allStats, csvFile)

    plotAllVariantsTrainingCurves(evaluator, OUTPUT_DIR)

    print("\nAll analysis and plots completed.")