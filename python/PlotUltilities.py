import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import rcParams


# change pyplot styling
"""
style.use('classic')
# Customize to resemble MATLAB's figure defaults
rcParams['lines.linewidth'] = 1.5  # Default line width in MATLAB
rcParams['lines.markersize'] = 6  # Marker size
rcParams['axes.grid'] = True  # Grid enabled by default
rcParams['grid.alpha'] = 0.5  # Grid transparency
rcParams['font.size'] = 12  # Font size
rcParams['axes.titlesize'] = 14  # Axes title size
rcParams['axes.labelsize'] = 12  # Axes label size
rcParams['xtick.labelsize'] = 10  # X-axis tick size
rcParams['ytick.labelsize'] = 10  # Y-axis tick size
"""

def PlotPiePlot(actual_counts, predicted_counts, title = ""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].pie(actual_counts, labels=actual_counts.index, autopct='%1.1f%%')
    axes[0].set_title("Actual Labels Distribution")
    axes[1].pie(predicted_counts, labels=predicted_counts.index, autopct='%1.1f%%')
    axes[1].set_title("Predicted Labels Distribution")
    plt.tight_layout()

    return fig, axes

def PlotConfusionMatrix(labels, predictions, class_names, normalize: bool = True):
    cm = confusion_matrix(labels, predictions)
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        cm = np.nan_to_num(cm_norm)  # Handle division by zero if any class is missing data

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=15, colorbar=False, values_format=".1f")
    plt.title("Confusion Matrix (Normalized to Percentages)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.grid(False)
    
    plt.tight_layout()

    return plt.gcf(), plt.gca()

def PlotKFoldConfusionMatrices(k_labels, k_predictions, class_names, normalize: bool = True, figsize=(10, 10)):
    num_folds = len(k_labels)
    cols = min(3, num_folds)
    rows = (num_folds + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, (labels, predictions) in enumerate(zip(k_labels, k_predictions)):
        cm = confusion_matrix(labels, predictions)
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
            cm = np.nan_to_num(cm_norm)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=axes[i], cmap='Blues', xticks_rotation='vertical', colorbar=False, values_format=".1f")
        axes[i].set_title(f'Fold {i + 1}')
        axes[i].grid(False)
    
    # Remove any extra subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    return fig, axes



def savePlot(fig, path, suffix=""):
    if not path: return
    imgPath = path

    if suffix:
        base, ext = os.path.splitext(path)
        imgPath = f"{base}{suffix}{ext}" if ext else f"{base}{suffix}.png"
    elif not path.lower().endswith(('.png', '.jpg', '.jpeg')):
        imgPath = f"{path}.png"

    fig.savefig(imgPath, bbox_inches='tight')
    return imgPath

def plotConfusionMatrix(evaluator, name, path=None, saveCsv=False):
    if name not in evaluator.confusionMatrices or name not in evaluator.classNames: return

    cm = evaluator.confusionMatrices[name]
    classNames = list(evaluator.classNames[name].values())
    rowSums = cm.sum(axis=1)
    safeRowSums = np.where(rowSums > 0, rowSums, 1)
    cmNorm = cm.astype('float') / safeRowSums[:, np.newaxis]
    cmNorm = np.nan_to_num(cmNorm)

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(cmNorm, annot=True, fmt='.2%', cmap='Blues', xticklabels=classNames, yticklabels=classNames, ax=ax)
    ax.set_title(f'Normalized Confusion Matrix for {name}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()

    if path:
        evaluator.savePlot(fig, path)
        if saveCsv:
            normDf = pd.DataFrame(cmNorm, index=classNames, columns=classNames)
            evaluator.saveDfToCsv(normDf, path, "_normalized", index=True)
    plt.show()
    return cm, cmNorm

def plotConfigMatrix(evaluator, matrix, modelType, path=None, saveCsv=False):
    if matrix.empty: return

    xCol = next((col for col in ['optimizerName', 'optimizer'] if col in matrix.columns), None)
    yCol = next((col for col in ['learningRate', 'lr'] if col in matrix.columns), None)

    if xCol and yCol:
        try:
            pivot = pd.pivot_table(matrix, values='Accuracy', index=yCol, columns=xCol, aggfunc='mean')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='crest', cbar_kws={'label': 'Accuracy (%)'}, ax=ax)
            ax.set_title(f'{modelType} Performance by Optimizer and Learning Rate')
            plt.tight_layout()

            if path:
                evaluator.savePlot(fig, path)
                if saveCsv:
                    evaluator.saveDfToCsv(matrix, path, "_full_results")
            plt.show()
            return
        except:
            pass
    evaluator.basePlot(matrix, modelType, path, saveCsv)

def basePlot(evaluator, matrix, modelType="Model", path=None, saveCsv=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(matrix['Model'], matrix['Accuracy'], color='skyblue')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.2f}%', ha='center', va='bottom')

    ax.set_title(f'{modelType} Model Results')
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if path:
        evaluator.savePlot(fig, path)
        if saveCsv:
            evaluator.saveDfToCsv(matrix, path, "_results")
    plt.show()

def plotRocCurve(evaluator, name, path=None, saveCsv=False):
    if name not in evaluator.results or evaluator.results[name].get('probabilities') is None: return

    probs = evaluator.results[name]['probabilities']
    yTrue = evaluator.results[name]['trueLabels']
    classes = list(evaluator.classNames[name].values())
    classToIdx = {cls: i for i, cls in enumerate(classes)}
    yTrueIdx = np.array([classToIdx[label] for label in yTrue])
    yTrueBin = label_binarize(yTrueIdx, classes=range(len(classes)))
    nClasses = min(len(classes), probs.shape[1])

    fig, ax = plt.subplots(figsize=(10, 8))
    rocData = {'class': [], 'fpr': [], 'tpr': [], 'auc': []}

    for i in range(nClasses):
        try:
            if np.sum(yTrueBin[:, i]) > 0:
                fpr, tpr, _ = roc_curve(yTrueBin[:, i], probs[:, i])
                rocAuc = auc(fpr, tpr)

                ax.plot(fpr, tpr, lw=1.5, alpha=0.8, label=f'Class {classes[i]} (AUC = {rocAuc:.2f})')
                rocData['class'].append(classes[i])
                rocData['fpr'].append(fpr)
                rocData['tpr'].append(tpr)
                rocData['auc'].append(rocAuc)
        except:
            continue
    try:
        yBin = yTrueBin[:, :nClasses]
        probsTrim = probs[:, :nClasses]
        validClasses = [i for i in range(nClasses) if np.sum(yBin[:, i]) > 0]

        if validClasses:
            validYBin = yBin[:, validClasses]
            validProbs = probsTrim[:, validClasses]

            macroAuc = roc_auc_score(validYBin, validProbs, multi_class='ovr', average='macro')
            microAuc = roc_auc_score(validYBin, validProbs, multi_class='ovr', average='micro')

            evaluator.metrics[name].update({'macroAuc': macroAuc, 'microAuc': microAuc})
    except Exception as e:
        print(f"Error calculating AUC metrics: {e}")

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set(xlim=[0, 1], ylim=[0, 1.05], title=f'ROC Curves for {name}', xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.legend(loc="best", fontsize='small', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()

    if path and rocData['class']:
        evaluator.savePlot(fig, path)
        if saveCsv:
            allRocData = []
            for i, cls in enumerate(rocData['class']):
                df = pd.DataFrame({'Class': cls, 'FPR': rocData['fpr'][i], 'TPR': rocData['tpr'][i], 'AUC': rocData['auc'][i]})
                allRocData.append(df)
            if allRocData:
                finalDf = pd.concat(allRocData, ignore_index=True)
                evaluator.saveDfToCsv(finalDf, path, "_roc")
    plt.show()
    return rocData

def computeModelRoc(evaluator, name):
    probs = evaluator.results[name]['probabilities']
    trueLabels = evaluator.results[name]['trueLabels']
    classes = list(evaluator.classNames[name].values())
    classToIdx = {cls: i for i, cls in enumerate(classes)}
    binaryCorrect = []
    confidenceScores = []
    validSamples = 0

    for prob, true in zip(probs, trueLabels):
        if true in classToIdx:
            predIdx = np.argmax(prob)
            trueIdx = classToIdx[true]

            binaryCorrect.append(1 if predIdx == trueIdx else 0)
            confidenceScores.append(np.max(prob))
            validSamples += 1

    if validSamples == 0 or np.sum(binaryCorrect) == 0:
        print(f"Warning: No valid samples or all predictions incorrect for model {name}")
        return [0, 1], [0, 0], 0.5
    try:
        fpr, tpr, _ = roc_curve(binaryCorrect, confidenceScores)
        rocAuc = auc(fpr, tpr)

        return fpr, tpr, rocAuc
    except Exception as e:
        print(f"Error calculating ROC for {name}: {e}")
        return [0, 1], [0, 1], 0.5

def plotModelCompareRoc(evaluator, models, title, path=None, saveCsv=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    compareData = {'model': [], 'fpr': [], 'tpr': [], 'auc': [], 'accuracy': []}

    for model in models:
        try:
            fpr, tpr, rocAuc = evaluator.computeModelRoc(model)

            ax.plot(fpr, tpr, lw=2, label=f'{model} (AUC = {rocAuc:.2f})')
            compareData['model'].append(model)
            compareData['fpr'].append(fpr)
            compareData['tpr'].append(tpr)
            compareData['auc'].append(rocAuc)
            compareData['accuracy'].append(evaluator.metrics.get(model, {}).get('accuracy', -1))
        except:
            continue

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if path and compareData['model']:
        evaluator.savePlot(fig, path)

        if saveCsv:
            allData = []

            for i, model in enumerate(compareData['model']):
                df = pd.DataFrame({'Model': model, 'FPR': compareData['fpr'][i], 'TPR': compareData['tpr'][i], 'AUC': compareData['auc'][i]})
                allData.append(df)

            if allData:
                fullDf = pd.concat(allData, ignore_index=True)
                evaluator.saveDfToCsv(fullDf, path, "_roc")

            summaryDf = pd.DataFrame({'Model': compareData['model'], 'AUC': compareData['auc'], 'Accuracy(%)': compareData['accuracy']})
            evaluator.saveDfToCsv(summaryDf, path, "_summary")
    plt.show()
    return compareData

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
