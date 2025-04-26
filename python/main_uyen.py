import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import torch

from DataManager import DataManager
from NN_Model_Uyen import NNModel
from PlotAndEval_uyen import (
    compareModels, plotConfusionMatrix, compareModelTypeVariants, compareBestAcrossModelTypes,
    plotConfigMatrix, plotAllVariantsTrainingCurves, evaluateModel
)

############### GLOBAL CONFIG #################
CURRENT_DIR = os.getcwd()
DATA_DIR = os.path.join(CURRENT_DIR, "../data/")
MODEL_DIR = os.path.join(CURRENT_DIR, "../models")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../output")

CLASSES = ["sitting", "running", "drinking", "eating", "listening_to_music"]
NUM_FILES_LIST = [5] 
IMAGE_SIZE = (260, 260)
DATA_SPLIT = 0.7

K_FOLDS = 2
NUM_EPOCHS = 3
BATCH_SIZE = 5
TARGET_ACCURACY = 95.0 
SAVE_CSV = None
USE_KFOLD = True 

ARCHITECTURES = [
    {'class': NNModel, 'model_type': 'EfficientNet', 'variant': 'b0', 'prefix': 'EfficientNet'},
    {'class': NNModel, 'model_type': 'ResNet', 'variant': '18', 'prefix': 'ResNet'},
]
OPTIMIZERS = ['adam', 'sgd', 'adamw']
LEARNING_RATES = [0.0001, 0.001, 0.01]
DROPOUT_RATE = 0.0
WEIGHT_DECAY = 0.0000

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

########### HELPER #################

def setGlobalSeed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

    # this is for the pc setting > no remove pls
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def generateModelVariants():
    return [
        dict(arch, optimizer=opt, lr=lr, prefix=f"{arch['prefix']}_{opt}_lr{lr}")
        for arch in ARCHITECTURES for opt in OPTIMIZERS for lr in LEARNING_RATES
    ]

################# TRAIN HELPER #################
def trainWithKfold(model, dm, model_name, k=5):
    all_stats, fold_acc = [], []
    
    for i in range(k):
        print(f"\n========= Fold {i+1}/{k} =========")
        model.resetModel(
            numClasses=len(dm.TrainingData["label"].unique()),
            optimizerName=model.optimizerName,
            learningRate=model.learningRate,
            weightDecay=model.weightDecay,
            dropoutRate=model.dropoutRate
        )
        
        model.setupMapMapping(dm.TrainingData) # do not remove as it will break code
        dm.SplitKFold(k=k, foldIndex=i)
        
        if i == 0: 
            dm.PlotDataDistrobution(dm.KTrainingData, dm.KTestData)
        
        stats = model.train(
            dm.KTrainingData, 
            epochs=NUM_EPOCHS, 
            batchSize=BATCH_SIZE,
            saveInterval=1, 
            loadModel=False, 
            saveModel=True, 
            targetAccuracy=TARGET_ACCURACY
        )
        
        if stats: 
            all_stats.append({'fold': i+1, 'epochs': stats})
        
        labels, images = dm.GetKTestLabels(), dm.GetKTestImages()
        preds = [model.predict(img) for img in images]
        acc = 100 * sum(a == p for a, p in zip(labels, preds)) / len(labels)
        fold_acc.append(acc)
        print(f"Fold {i+1} val acc: {acc:.2f}%")
    
    avg, std = np.mean(fold_acc), np.std(fold_acc)
    print(f"\n{model_name} {k}-fold: {avg:.2f}% ± {std:.2f}% | Folds: {', '.join(f'{a:.2f}%' for a in fold_acc)}")
    
    return {
        'fold_stats': all_stats, 
        'fold_accuracies': fold_acc, 
        'mean_accuracy': avg, 
        'std_accuracy': std
    }

def trainSingleModel(model, dm, model_name):
    print(f"\n{'='*10} Training {model_name} (single split) {'='*10}")
    
    model.resetModel(
        numClasses=len(dm.TrainingData["label"].unique()),
        optimizerName=model.optimizerName,
        learningRate=model.learningRate,
        weightDecay=model.weightDecay,
        dropoutRate=model.dropoutRate
    )
    
    model.setupMapMapping(dm.TrainingData) # do not remove as it will break code
    stats = model.train(
        dm.TrainingData, 
        epochs=NUM_EPOCHS, 
        batchSize=BATCH_SIZE,
        saveInterval=1, 
        loadModel=False, 
        saveModel=True, 
        targetAccuracy=TARGET_ACCURACY
    )
    
    last_acc = stats[-1][3] if stats else 0.0
    
    return {
        'fold_stats': [{'fold': 1, 'epochs': stats}], 
        'fold_accuracies': [last_acc], 
        'mean_accuracy': last_acc, 
        'std_accuracy': 0.0
    }

############# EVAL HELPER #################
def calculatePerClassAccuracy(labels, preds, class_mapping):
    class_total = {cls: 0 for cls in class_mapping.values()}
    class_correct = {cls: 0 for cls in class_mapping.values()}
    
    for a, p in zip(labels, preds):
        class_total[a] += 1
        if a == p: 
            class_correct[a] += 1
    
    print("\nPer-Class Accuracy:")
    for cls in class_mapping.values():
        if class_total[cls]:
            acc = 100 * class_correct[cls] / class_total[cls]
            print(f"{cls}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

def analyzeMisclassifications(images, labels, preds):
    confused = [(i, a, p) for i, (a, p) in enumerate(zip(labels, preds)) if a != p]
    print(f"\nMisclassified: {len(confused)}/{len(labels)}")
    if not confused: 
        return

    pair_counts = {}
    for _, a, p in confused:
        key = (a, p)
        pair_counts[key] = pair_counts.get(key, 0) + 1

    print("\nTop misclassifications:")
    top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for (a, p), cnt in top_pairs:
        print(f"  {a} → {p}: {cnt}")

    n = len(confused)
    grid = int(math.ceil(math.sqrt(n)))
    fig, axes = plt.subplots(grid, grid, figsize=(15, 15))
    axes = axes.flatten()

    for i, (idx, a, p) in enumerate(confused):
        if i < len(axes):
            img = images[idx].image if hasattr(images[idx], "image") else images[idx]
            # dont rm this as it will break the code
            if img is not None:
                if not isinstance(img, np.ndarray): 
                    img = np.array(img)
                if img.dtype == np.dtype('O'): 
                    img = np.array(img.tolist(), dtype=np.float32)
                if img.ndim == 3 and img.shape[-1] == 1: 
                    img = img.squeeze(-1)
                axes[i].imshow(img)
            axes[i].set_title(f"A: {a}\nP: {p}", color='red')
            axes[i].axis('off')

    for j in range(len(confused), len(axes)): 
        axes[j].axis('off')

    plt.suptitle(f"Misclassified Images ({n})", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def evaluateAndVisualizeResults(dm, models, results, confusionMatrices, classNames, metrics, kfold_results, variants):
    print("\n" + "="*70 + "\nEVALUATING MODELS\n" + "="*70)
    test_results = {}
    
    for model_name, model in models.items():
        acc = evaluateModel(models, results, confusionMatrices, classNames, metrics, model_name, dm.TestData, BATCH_SIZE)
        test_results[model_name] = acc
        res = results.get(model_name, {})
        preds, labels = res.get('predictions'), res.get('trueLabels')
        class_names = classNames.get(model_name)
        images = dm.TestData["image"].tolist() if hasattr(dm.TestData, "image") else None
        
        if preds is not None and labels is not None and class_names and images:
            calculatePerClassAccuracy(labels, preds, class_names)
            analyzeMisclassifications(images, labels, preds)
    
    compareModels(results, metrics)
    
    for model_name in confusionMatrices:
        plotConfusionMatrix(confusionMatrices, classNames, model_name)
    
    for mt in {n.split('_')[0] for n in results if '_' in n}:
        compareModelTypeVariants(results, classNames, metrics, mt)
    
    compareBestAcrossModelTypes(results, classNames, metrics)
    
    for model_name in models:
        if model_name in kfold_results:
            folds = kfold_results[model_name].get('fold_stats', [])
            all_epochs = [ep for fold in folds for ep in fold.get('epochs', [])]
            if all_epochs:
                results.setdefault(model_name, {})['epochStats'] = all_epochs
    
    plotAllVariantsTrainingCurves(results, models, OUTPUT_DIR)
    
    for mt in {v['prefix'].split('_')[0] for v in variants if 'prefix' in v}:
        configs = [{
            'Model': v['prefix'],
            'Variant': v.get('variant', ''),
            'optimizerName': v.get('optimizer', ''),
            'learningRate': v.get('lr', ''),
            'Accuracy': results.get(v['prefix'], {}).get('accuracy', 0.0)
        } for v in variants if v['prefix'].startswith(mt)]
        
        if configs:
            config_df = pd.DataFrame(configs)
            plotConfigMatrix(
                config_df,
                mt,
                path=os.path.join(OUTPUT_DIR, f"{mt}_config_matrix"),
                saveCsv=False
            )
    
    visualizeKfoldResults(kfold_results, test_results)
    return test_results

def visualizeKfoldResults(kfold_results, test_results):
    model_names = list(kfold_results.keys())
    mean_acc = [r['mean_accuracy'] for r in kfold_results.values()]
    std_acc = [r['std_accuracy'] for r in kfold_results.values()]
    test_acc = [test_results.get(n, 0) for n in model_names]
    
    idx = np.argsort(mean_acc)[::-1]
    model_names = [model_names[i] for i in idx]
    mean_acc = [mean_acc[i] for i in idx]
    std_acc = [std_acc[i] for i in idx]
    test_acc = [test_acc[i] for i in idx]

    plt.figure(figsize=(15, 8))
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, mean_acc, width, yerr=std_acc, 
            label=f'{K_FOLDS}-Fold CV', capsize=5, color='skyblue')
    plt.bar(x + width/2, test_acc, width, 
            label='Test', color='lightcoral')
    
    plt.ylabel('Accuracy (%)')
    plt.title(f'Model Performance: {K_FOLDS}-Fold CV vs Test')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    top_n = min(5, len(model_names))
    print(f"\nTop {top_n} Models:")
    print("="*100)
    print(f"{'Model':<30} {'CV Accuracy':<20} {'Test Accuracy':<20} {'Optimizer':<10} {'Learning Rate':<15}")
    print("-"*100)
    
    for model in model_names[:top_n]:
        parts = model.split('_')
        optimizer = parts[2] if len(parts) > 2 else "N/A"
        lr = parts[3].replace('lr', '') if len(parts) > 3 else "N/A"
        cv_acc = f"{kfold_results[model]['mean_accuracy']:.2f}% ± {kfold_results[model]['std_accuracy']:.2f}%"
        test_acc_str = f"{test_results.get(model, 0):.2f}%"
        print(f"{model:<30} {cv_acc:<20} {test_acc_str:<20} {optimizer:<10} {lr:<15}")

############## MAIN #################
if __name__ == "__main__":
    setGlobalSeed(42)
    
    print("=" * 70 + "\nRUN ML PROJECT HAR\n" + "=" * 70)
    
    dm = DataManager()
    
    for NUM_FILES in NUM_FILES_LIST:
        print(f"\nRunning experiment with NUM_FILES={NUM_FILES}")
        dm.ResetData()
        classFilter = CLASSES
        
        dm.LoadTrainAndTestData(
            folderName=DATA_DIR,
            csvFileName="Training_set.csv",
            numFiles=NUM_FILES,
            classFilter=classFilter,
            split=DATA_SPLIT
        )
        dm.RemoveMissingData()
        dm.ResizeImages(TargetSize=IMAGE_SIZE)
        
        num_classes = len(dm.TrainingData["label"].unique())
        print(f"Loaded {len(dm.TrainingData)} train, {len(dm.TestData)} test, {num_classes} classes")
        
        print("\n" + "=" * 70 + "\nMODEL GENERATION AND TRAINING\n" + "=" * 70)
        models, results, confusionMatrices, classNames, metrics = {}, {}, {}, {}, {}
        variants = generateModelVariants()
        
        print("\nInitializing model variants...")
        for i, v in enumerate(variants):
            print(f"Init model {i+1}/{len(variants)}: {v['prefix']}")
            
            model = v['class'](
                numClasses=num_classes, 
                modelType=v['model_type'], 
                variant=v['variant'],
                modelDir=MODEL_DIR, 
                optimizerName=v['optimizer'], 
                learningRate=v['lr'],
                weightDecay=WEIGHT_DECAY, 
                dropoutRate=DROPOUT_RATE
            )
            
            model.preprocess(dm.TrainingData)
            models[v['prefix']] = model
        
        print("\n" + "=" * 70)
        print(f"TRAINING ALL VARIANTS WITH {'K-FOLD' if USE_KFOLD else 'SINGLE SPLIT'}")
        print("=" * 70)
        
        kfold_results = {}
        
        for model_name, model in models.items():
            if USE_KFOLD:
                fold_results = trainWithKfold(model, dm, model_name, k=K_FOLDS) 
            else:
                fold_results = trainSingleModel(model, dm, model_name)
                
            kfold_results[model_name] = fold_results
            
            if 'fold_stats' in fold_results and fold_results['fold_stats']:
                all_epochs = [ep for fold in fold_results['fold_stats'] for ep in fold.get('epochs', [])]
                results.setdefault(model_name, {})['epochStats'] = all_epochs
        
        test_results = evaluateAndVisualizeResults(dm, models, results, confusionMatrices, classNames, metrics, kfold_results, variants)
        
        print("\n" + "="*70 + "\nANALYSIS COMPLETED\n" + "="*70)