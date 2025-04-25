import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def compareModels(results, metrics):
    if not metrics:
        return None
    data = [{'Model': n, 'Accuracy (%)': m['accuracy']} for n, m in metrics.items()]
    df = pd.DataFrame(data)
    plt.figure(figsize=(15, 10))
    values = [metrics[model]['accuracy'] for model in metrics]
    bars = plt.bar(list(metrics.keys()), values, color='skyblue')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%', ha='center', va='bottom')
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return df

def plotConfusionMatrix(confusionMatrices, classNames, name, path=None, saveCsv=False):
    if name not in confusionMatrices or name not in classNames:
        return
    cm = confusionMatrices[name]
    labels = list(classNames[name].values())
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'Normalized Confusion Matrix for {name}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    if path:
        fig.savefig(f'{path}.png', bbox_inches='tight')
        if saveCsv:
            pd.DataFrame(cm_norm, index=labels, columns=labels).to_csv(f'{path}_normalized.csv')
    plt.show()
    return cm, cm_norm

def plotConfigMatrix(matrix, modelType, path=None, saveCsv=False):
    if matrix.empty:
        return
    x_col = next((c for c in ['optimizerName', 'optimizer'] if c in matrix.columns), None)
    y_col = next((c for c in ['learningRate', 'lr'] if c in matrix.columns), None)
    if x_col and y_col:
        try:
            pivot = pd.pivot_table(matrix, values='Accuracy', index=y_col, columns=x_col, aggfunc='mean')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='crest', cbar_kws={'label': 'Accuracy (%)'}, ax=ax)
            ax.set_title(f'{modelType} Performance by Optimizer and Learning Rate')
            plt.tight_layout()
            if path:
                fig.savefig(f'{path}.png', bbox_inches='tight')
                if saveCsv:
                    matrix.to_csv(f'{path}_full_results.csv')
            plt.show()
            return
        except Exception:
            pass
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(matrix['Model'], matrix['Accuracy'], color='skyblue')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%', ha='center', va='bottom')
    ax.set_title(f'{modelType} Model Results')
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if path:
        fig.savefig(f'{path}.png', bbox_inches='tight')
        if saveCsv:
            matrix.to_csv(f'{path}_results.csv')
    plt.show()

def plotAllVariantsTrainingCurves(results, models, outputDir):
    model_variants = defaultdict(list)
    for name in models:
        model_type = name.split('_')[0]
        model_variants[model_type].append(name)
    for model_type, variants in model_variants.items():
        plt.figure(figsize=(14, 6))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        for variant in variants:
            stats = None
            if variant in results and 'epochStats' in results[variant]:
                stats = results[variant]['epochStats']
            if not stats:
                csv_file = os.path.join(outputDir, f"{variant}_epoch_stats.csv")
                if os.path.isfile(csv_file):
                    with open(csv_file, 'r') as f:
                        rows = list(csv.reader(f))
                    if rows and rows[0][0].lower() == "epoch":
                        rows = rows[1:]
                    stats = [[int(r[0]), float(r[1]), float(r[2]), float(r[3])] for r in rows]
            if not stats:
                continue
            epochs = [r[0] for r in stats]
            losses = [r[2] for r in stats]
            accs = [r[3] for r in stats]
            label = variant.replace(model_type + "_", "")
            ax1.plot(epochs, accs, marker='o', label=label)
            ax2.plot(epochs, losses, marker='o', label=label)
        ax1.set_title(f"{model_type}: Accuracy per Variant")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax2.set_title(f"{model_type}: Loss per Variant")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        plt.suptitle(f"{model_type} Training Curves (All Variants)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(outputDir, f"{model_type}_all_variants_training_curves.png")
        plt.savefig(save_path)
        plt.show()
        print(f"Saved training curves for {model_type} to {save_path}")

def computeModelRoc(results, classNames, name):
    probs = results[name]['probabilities']
    true_labels = results[name]['trueLabels']
    classes = list(classNames[name].values())
    class_idx = {cls: i for i, cls in enumerate(classes)}
    y_true = np.array([class_idx.get(lbl, -1) for lbl in true_labels])
    mask = y_true != -1
    y_true = y_true[mask]
    probs = np.array(probs)[mask]
    n_classes = len(classes)
    if n_classes == 2:
        y_score = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = auc(fpr, tpr)
        return fpr, tpr, auc_val
    elif n_classes > 2:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)
        return all_fpr, mean_tpr, macro_auc
    else:
        print(f"ROC/AUC not supported for model {name} (no classes)")
        return [0, 1], [0, 0], 0.5

def plotModelCompareRoc(results, classNames, metrics, models, title, path=None, saveCsv=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    compare = {'model': [], 'fpr': [], 'tpr': [], 'auc': [], 'accuracy': []}
    for model in models:
        try:
            fpr, tpr, auc_val = computeModelRoc(results, classNames, model)
            ax.plot(fpr, tpr, lw=2, label=f'{model} (AUC = {auc_val:.2f})')
            compare['model'].append(model)
            compare['fpr'].append(fpr)
            compare['tpr'].append(tpr)
            compare['auc'].append(auc_val)
            compare['accuracy'].append(metrics.get(model, {}).get('accuracy', -1))
        except:
            continue
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if path and compare['model']:
        fig.savefig(f'{path}.png', bbox_inches='tight')
        if saveCsv:
            all_data = []
            for i, model in enumerate(compare['model']):
                df = pd.DataFrame({'Model': model, 'FPR': compare['fpr'][i], 'TPR': compare['tpr'][i], 'AUC': compare['auc'][i]})
                all_data.append(df)
            if all_data:
                pd.concat(all_data, ignore_index=True).to_csv(f'{path}_roc.csv', index=False)
            pd.DataFrame({'Model': compare['model'], 'AUC': compare['auc'], 'Accuracy(%)': compare['accuracy']}).to_csv(f'{path}_summary.csv', index=False)
    plt.show()
    return compare

def compareModelTypeVariants(results, classNames, metrics, modelType, path=None, saveCsv=False):
    variants = [m for m in results if m.startswith(f"{modelType}_") and results[m].get('probabilities') is not None]
    if not variants:
        return []
    title = f'ROC Comparison of {modelType} Variants'
    return plotModelCompareRoc(results, classNames, metrics, variants, title, path, saveCsv)

def compareBestAcrossModelTypes(results, classNames, metrics, path=None, saveCsv=False):
    model_types = {model.split('_')[0] for model in results if '_' in model and results[model].get('probabilities') is not None}
    if not model_types:
        return []
    best_models = []
    for model_type in model_types:
        variants = [m for m in results if m.startswith(f"{model_type}_") and results[m].get('probabilities') is not None]
        if variants:
            best = max(variants, key=lambda v: metrics.get(v, {}).get('accuracy', -1))
            best_models.append(best)
    if not best_models:
        return []
    title = 'Comparison of Best Model Variants'
    return plotModelCompareRoc(results, classNames, metrics, best_models, title, path, saveCsv)

def evaluateModel(models, results, confusionMatrices, classNames, metrics, name, testData, batchSize=32):
    if name not in models:
        raise ValueError(f"\nModel '{name}' not found")
    model = models[name]
    print(f"\nEvaluating model: {name}")
    true_labels = testData["label"].tolist()
    try:
        if hasattr(model, 'predict'):
            preds = model.predict(testData["image"], returnProba=False)
            probs = model.predict(testData["image"], returnProba=True)
        elif hasattr(model, 'predictBatch'):
            preds = model.predictBatch(testData["image"])
            probs = model.predictBatch(testData["image"], returnProba=True)
        elif hasattr(model, 'evaluate'):
            acc = model.evaluate(testData, batchSize)
            print(f"\nModel returned accuracy: {acc:.2f}%")
            return acc
        else:
            raise AttributeError("\nModel has no prediction method")
    except Exception as e:
        print(f"\nError during prediction: {e}")
        return 0
    preds_np = np.array(preds)
    true_np = np.array(true_labels)
    acc = np.mean(preds_np == true_np) * 100.0
    results[name] = {
        'predictions': preds,
        'trueLabels': true_labels,
        'accuracy': acc,
        'probabilities': probs
    }
    all_classes = sorted(set(true_labels + preds))
    classNames[name] = {i: cls for i, cls in enumerate(all_classes)}
    class_idx = {cls: i for i, cls in enumerate(all_classes)}
    n_classes = len(all_classes)
    cm = confusion_matrix(true_labels, preds, labels=all_classes)
    confusionMatrices[name] = cm
    metrics[name] = {'accuracy': acc}
    if probs is not None and n_classes == 2:
        try:
            y_true = np.array([class_idx[label] for label in true_labels])
            y_score = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = auc(fpr, tpr)
            metrics[name].update({'auc': auc_val})
        except Exception:
            metrics[name].update({'auc': None})
    elif probs is not None and n_classes > 2:
        try:
            y_true = np.array([class_idx[label] for label in true_labels])
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            macro_auc = auc(all_fpr, mean_tpr)
            metrics[name].update({'auc': macro_auc})
        except Exception:
            metrics[name].update({'auc': None})
    print(f"\nAccuracy: {acc:.2f}%")
    return acc
