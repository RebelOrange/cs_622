import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import torch

from DataManager import DataManager
from NN_Model import NNModel
from ModelEvaluator import ModelEvaluator, plotConfigMatrix
from TrainingPlot import plotAllVariantsTrainingCurves
from Timer import Timer

# ===================== GLOBAL CONFIGURATION =====================
CURRENT_DIR = os.getcwd()
DATA_DIR = os.path.join(CURRENT_DIR, "../data/")
MODEL_DIR = os.path.join(CURRENT_DIR, "../models")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../output")

CLASSES = ["sitting", "running", "drinking", "eating"]
NUM_FILES_LIST = [50]
IMAGE_SIZE = (260, 260)
DATA_SPLIT = 0.8

K_FOLDS = 2
NUM_EPOCHS = 3
BATCH_SIZE = 8
TARGET_ACCURACY = 95.0
SAVE_CSV = None
USE_KFOLD = True

ARCHITECTURES = [
    {'class': NNModel, 'model_type': 'EfficientNet', 'variant': 'b0', 'prefix': 'EfficientNet'},
    # {'class': NNModel, 'model_type': 'ResNet', 'variant': '18', 'prefix': 'ResNet'},
]
OPTIMIZERS = ['adam', 'sgd']
LEARNING_RATES = [0.0001, 0.001]
DROPOUT_RATE = 0.0
WEIGHT_DECAY = 0.0000

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_global_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def analyze_class_distribution(training_data, test_data):
    train_counts = training_data["label"].value_counts()
    test_counts = test_data["label"].value_counts()
    print("\n--- Class Distribution ---")
    for cls in train_counts.index:
        train_pct = train_counts[cls]/len(training_data)*100
        test_pct = test_counts.get(cls, 0)/len(test_data)*100
        diff = abs(train_pct - test_pct)
        status = "IMBALANCED" if diff > 5 else "OK"
        print(f"{cls}: Train {train_pct:.2f}% vs Test {test_pct:.2f}% ({status})")
    return train_counts, test_counts

def visualize_data_distribution(dm, train_counts, test_counts):
    dm.PlotDataDistrobution(dm.TrainingData, dm.TestData)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(train_counts.index))
    width = 0.35
    plt.bar(x - width/2, train_counts / len(dm.TrainingData) * 100, width, label='Train')
    plt.bar(x + width/2, test_counts / len(dm.TestData) * 100, width, label='Test')
    plt.xlabel('Classes'); plt.ylabel('Percentage (%)')
    plt.title('Class Distribution: Train vs Test')
    plt.xticks(x, train_counts.index); plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(); plt.show()
    dm.ShowRandomImages(numImages=6, showGrayscale=False, showSegmented=False)

def train_with_kfold(model, dm, model_name, k=5):
    all_stats, fold_acc = [], []
    for i in range(k):
        print(f"\n--- Fold {i+1}/{k} ---")
        model.resetModel(numClasses=len(dm.TrainingData["label"].unique()),
                         optimizerName=model.optimizerName,
                         learningRate=model.learningRate,
                         weightDecay=model.weightDecay,
                         dropoutRate=model.dropoutRate)
        model.setupMapMapping(dm.TrainingData)
        dm.SplitKFold(k=k, foldIndex=i)
        if i == 0: dm.PlotDataDistrobution(dm.KTrainingData, dm.KTestData)
        stats = model.train(dm.KTrainingData, epochs=NUM_EPOCHS, batchSize=BATCH_SIZE,
                            saveInterval=1, loadModel=False, saveModel=False, targetAccuracy=TARGET_ACCURACY)
        if stats: all_stats.append({'fold': i+1, 'epochs': stats})
        labels, images = dm.GetKTestLabels(), dm.GetKTestImages()
        preds = [model.predict(img) for img in images]
        acc = 100 * sum(a == p for a, p in zip(labels, preds)) / len(labels)
        fold_acc.append(acc)
        print(f"Fold {i+1} val acc: {acc:.2f}%")
    avg, std = np.mean(fold_acc), np.std(fold_acc)
    print(f"\n{model_name} {k}-fold: {avg:.2f}% ± {std:.2f}% | Folds: {', '.join(f'{a:.2f}%' for a in fold_acc)}")
    return {'fold_stats': all_stats, 'fold_accuracies': fold_acc, 'mean_accuracy': avg, 'std_accuracy': std}

def train_single_model(model, dm, model_name):
    print(f"\n{'='*10} Training {model_name} (single split) {'='*10}")
    model.resetModel(numClasses=len(dm.TrainingData["label"].unique()),
                     optimizerName=model.optimizerName,
                     learningRate=model.learningRate,
                     weightDecay=model.weightDecay,
                     dropoutRate=model.dropoutRate)
    model.setupMapMapping(dm.TrainingData)
    stats = model.train(dm.TrainingData, epochs=NUM_EPOCHS, batchSize=BATCH_SIZE,
                        saveInterval=1, loadModel=False, saveModel=False, targetAccuracy=TARGET_ACCURACY)
    last_acc = stats[-1][3] if stats else 0.0
    return {'fold_stats': [{'fold': 1, 'epochs': stats}], 'fold_accuracies': [last_acc], 'mean_accuracy': last_acc, 'std_accuracy': 0.0}

def generate_model_variants():
    return [
        dict(arch, optimizer=opt, lr=lr, prefix=f"{arch['prefix']}_{opt}_lr{lr}")
        for arch in ARCHITECTURES for opt in OPTIMIZERS for lr in LEARNING_RATES
    ]

def calculate_per_class_accuracy(labels, preds, class_mapping):
    class_total = {cls: 0 for cls in class_mapping.values()}
    class_correct = {cls: 0 for cls in class_mapping.values()}
    for a, p in zip(labels, preds):
        class_total[a] += 1
        if a == p: class_correct[a] += 1
    print("\nPer-Class Accuracy:")
    for cls in class_mapping.values():
        if class_total[cls]:
            acc = 100 * class_correct[cls] / class_total[cls]
            print(f"{cls}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

def analyze_misclassifications(images, labels, preds):
    confused = [(i, a, p) for i, (a, p) in enumerate(zip(labels, preds)) if a != p]
    print(f"\nMisclassified: {len(confused)}/{len(labels)}")
    if not confused: return
    from collections import Counter
    pairs = Counter((a, p) for _, a, p in confused)
    print("\nTop misclassifications:")
    for (a, p), cnt in pairs.most_common(5):
        print(f"  {a} → {p}: {cnt}")
    n = len(confused); grid = int(math.ceil(math.sqrt(n)))
    fig, axes = plt.subplots(grid, grid, figsize=(15, 15)); axes = axes.flatten()
    for i, (idx, a, p) in enumerate(confused):
        if i < len(axes):
            img = images[idx].image if hasattr(images[idx], "image") else images[idx]
            if img is not None:
                if not isinstance(img, np.ndarray): img = np.array(img)
                if img.dtype == np.dtype('O'): img = np.array(img.tolist(), dtype=np.float32)
                if img.ndim == 3 and img.shape[-1] == 1: img = img.squeeze(-1)
                axes[i].imshow(img)
            axes[i].set_title(f"A: {a}\nP: {p}", color='red'); axes[i].axis('off')
    for j in range(len(confused), len(axes)): axes[j].axis('off')
    plt.suptitle(f"Misclassified Images ({n})", fontsize=16)
    plt.tight_layout(); plt.subplots_adjust(top=0.95); plt.show()

def evaluate_and_visualize_results(dm, evaluator, kfold_results, variants):
    print("\n" + "="*70 + "\nEVALUATING MODELS\n" + "="*70)
    test_results = {}
    for model_name, model in evaluator.models.items():
        acc = evaluator.evaluateModel(model_name, dm.TestData, BATCH_SIZE)
        test_results[model_name] = acc
        res = evaluator.results.get(model_name, {})
        preds, labels = res.get('predictions'), res.get('trueLabels')
        class_names = evaluator.classNames.get(model_name)
        images = dm.TestData["image"].tolist() if hasattr(dm.TestData, "image") else None
        if preds is not None and labels is not None and class_names and images:
            calculate_per_class_accuracy(labels, preds, class_names)
            analyze_misclassifications(images, labels, preds)
    evaluator.compareModels()
    for model_name in evaluator.confusionMatrices: evaluator.plotConfusionMatrix(model_name)
    for mt in {n.split('_')[0] for n in evaluator.results if '_' in n}: evaluator.compareModelTypeVariants(mt)
    evaluator.compareBestAcrossModelTypes()
    for model_name in evaluator.models:
        if model_name in kfold_results:
            folds = kfold_results[model_name].get('fold_stats', [])
            all_epochs = [ep for fold in folds for ep in fold.get('epochs', [])]
            if all_epochs:
                evaluator.results.setdefault(model_name, {})['epochStats'] = all_epochs
    plotAllVariantsTrainingCurves(evaluator, OUTPUT_DIR)
    for mt in {v['prefix'].split('_')[0] for v in variants if 'prefix' in v}:
        configs = [{
            'Model': v['prefix'],
            'Variant': v.get('variant', ''),
            'optimizerName': v.get('optimizer', ''),
            'learningRate': v.get('lr', ''),
            'Accuracy': evaluator.results.get(v['prefix'], {}).get('accuracy', 0.0)
        } for v in variants if v['prefix'].startswith(mt)]
        if configs:
            config_df = pd.DataFrame(configs)
            plotConfigMatrix(evaluator, config_df, mt, path=os.path.join(OUTPUT_DIR, f"{mt}_config_matrix"), saveCsv=True)
    visualize_kfold_results(kfold_results, test_results)
    return test_results

def visualize_kfold_results(kfold_results, test_results):
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
    x = np.arange(len(model_names)); width = 0.35
    plt.bar(x - width/2, mean_acc, width, yerr=std_acc, label=f'{K_FOLDS}-Fold CV', capsize=5, color='skyblue')
    plt.bar(x + width/2, test_acc, width, label='Test', color='lightcoral')
    plt.ylabel('Accuracy (%)'); plt.title(f'Model Performance: {K_FOLDS}-Fold CV vs Test')
    plt.xticks(x, model_names, rotation=45, ha='right'); plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(); plt.show()
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

def main():
    set_global_seed(42)
    project_timer = Timer(); project_timer.start()
    print("=" * 70 + "\nINTEGRATED MODEL TRAINING AND EVALUATION PIPELINE\n" + "=" * 70)
    data_loading_timer = Timer(); data_loading_timer.start()
    dm = DataManager()
    for NUM_FILES in NUM_FILES_LIST:
        print(f"\nRunning experiment with NUM_FILES={NUM_FILES}")
        dm.ResetData()
        classFilter = CLASSES  # or set to None if you want all classes
        dm.LoadTrainAndTestData(
            folderName=DATA_DIR,
            csvFileName="Training_set.csv",
            numFiles=NUM_FILES,
            classFilter=classFilter,
            split=DATA_SPLIT)
        dm.RemoveMissingData()
        dm.ResizeImages(TargetSize=IMAGE_SIZE)
        num_classes = len(dm.TrainingData["label"].unique())
        print(f"Loaded {len(dm.TrainingData)} train, {len(dm.TestData)} test, {num_classes} classes")
        train_counts, test_counts = analyze_class_distribution(dm.TrainingData, dm.TestData)
        visualize_data_distribution(dm, train_counts, test_counts)
        data_loading_timer.stop()
        print("\n" + "=" * 70 + "\nMODEL GENERATION AND TRAINING\n" + "=" * 70)
        evaluator = ModelEvaluator()
        variants = generate_model_variants()
        print("\nInitializing model variants...")
        for i, v in enumerate(variants):
            print(f"Init model {i+1}/{len(variants)}: {v['prefix']}")
            model = v['class'](numClasses=num_classes, modelType=v['model_type'], variant=v['variant'],
                               modelDir=MODEL_DIR, optimizerName=v['optimizer'], learningRate=v['lr'],
                               weightDecay=WEIGHT_DECAY, dropoutRate=DROPOUT_RATE)
            model.preprocess(dm.TrainingData)
            evaluator.addModel(v['prefix'], model)
        print("\n" + "=" * 70)
        print(f"STEP 2: TRAINING ALL VARIANTS WITH {'K-FOLD' if USE_KFOLD else 'SINGLE SPLIT'}")
        print("=" * 70)
        training_timer = Timer(); training_timer.start()
        kfold_results = {}
        for model_name, model in evaluator.models.items():
            fold_results = train_with_kfold(model, dm, model_name, k=K_FOLDS) if USE_KFOLD else train_single_model(model, dm, model_name)
            kfold_results[model_name] = fold_results
            if 'fold_stats' in fold_results and fold_results['fold_stats']:
                evaluator.results.setdefault(model_name, {})['epochStats'] = [ep for fold in fold_results['fold_stats'] for ep in fold.get('epochs', [])]
        training_timer.stop()
        evaluation_timer = Timer(); evaluation_timer.start()
        test_results = evaluate_and_visualize_results(dm, evaluator, kfold_results, variants)
        evaluation_timer.stop()
        print("\n" + "=" * 70 + "\nFINAL RESULTS\n" + "=" * 70)
        best_model = max(kfold_results, key=lambda k: kfold_results[k]['mean_accuracy'])
        best_cv = kfold_results[best_model]['mean_accuracy']
        best_std = kfold_results[best_model]['std_accuracy']
        best_test = test_results.get(best_model, 0)
        print(f"\nBEST MODEL: {best_model}\nCV acc: {best_cv:.2f}% ± {best_std:.2f}%\nTest acc: {best_test:.2f}%")
        if SAVE_CSV:
            print("\nExporting results...")
            evaluator.exportAllConfusionMatrices(OUTPUT_DIR, saveCsv=True)
            evaluator.exportTestResults(os.path.join(OUTPUT_DIR, "test_results.csv"), saveCsv=True)
        project_timer.stop()
        print("\n" + "="*70 + "\nANALYSIS COMPLETED\n" + "="*70)

if __name__ == "__main__":
    main()