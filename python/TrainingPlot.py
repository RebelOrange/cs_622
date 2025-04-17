import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

def plotAllVariantsTrainingCurves(evaluator, outputDir):
    modelTypeToVariants = defaultdict(list)
    
    for modelName in evaluator.models:
        modelType = modelName.split('_')[0]
        modelTypeToVariants[modelType].append(modelName)
    
    for modelType, variantNames in modelTypeToVariants.items():
        plt.figure(figsize=(14, 6))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
    
        for variantName in variantNames:
            epochStats = None
    
            if variantName in evaluator.results and 'epochStats' in evaluator.results[variantName]:
                epochStats = evaluator.results[variantName]['epochStats']
    
            if not epochStats or len(epochStats) == 0:
                csvFile = os.path.join(outputDir, f"{variantName}_epoch_stats.csv")
    
                if os.path.isfile(csvFile):
    
                    with open(csvFile, 'r') as f:
                        reader = csv.reader(f)
                        rows = list(reader)
    
                    if rows and rows[0][0].lower() == "epoch":
                        rows = rows[1:]
    
                    epochStats = [[int(row[0]), float(row[1]), float(row[2]), float(row[3])] for row in rows]
    
            if not epochStats or len(epochStats) == 0:
                continue
    
            epochs = [row[0] for row in epochStats]
            losses = [row[2] for row in epochStats]
            accs = [row[3] for row in epochStats]
            label = variantName.replace(modelType + "_", "")
    
            ax1.plot(epochs, accs, marker='o', label=label)
            ax2.plot(epochs, losses, marker='o', label=label)
    
        ax1.set_title(f"{modelType}: Accuracy per Variant")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
    
        ax2.set_title(f"{modelType}: Loss per Variant")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
    
        plt.suptitle(f"{modelType} Training Curves (All Variants)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        savePath = os.path.join(outputDir, f"{modelType}_all_variants_training_curves.png")
        plt.savefig(savePath)
        plt.show()
        print(f"Saved training curves for {modelType} to {savePath}")
