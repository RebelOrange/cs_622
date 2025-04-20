import pandas as pd
import matplotlib.pyplot as plt

def compareModels(evaluator):
    if not evaluator.metrics: return None

    metricsData = [{'Model': n, 'Accuracy (%)': m['accuracy']} for n, m in evaluator.metrics.items()]
    df = pd.DataFrame(metricsData)

    plt.figure(figsize=(15, 10))
    values = [evaluator.metrics[model]['accuracy'] for model in evaluator.metrics]
    bars = plt.bar(list(evaluator.metrics.keys()), values, color='skyblue')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.2f}%', ha='center', va='bottom')

    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return df

def compareModelTypeVariants(evaluator, modelType, path=None, saveCsv=False):
    variants = [m for m in evaluator.results.keys() if m.startswith(f"{modelType}_") and evaluator.results[m].get('probabilities') is not None]

    if not variants: return []
    title = f'ROC Comparison of {modelType} Variants'

    return evaluator.plotModelCompareRoc(variants, title, path, saveCsv)

def compareBestAcrossModelTypes(evaluator, modelTypes=None, path=None, saveCsv=False):
    if not modelTypes:
        modelTypes = set()
        for model in evaluator.results.keys():
            if '_' in model and evaluator.results[model].get('probabilities') is not None:
                modelTypes.add(model.split('_')[0])

    if not modelTypes: return []
    bestModels = []

    for modelType in modelTypes:
        variants = [m for m in evaluator.results.keys() if m.startswith(f"{modelType}_") and evaluator.results[m].get('probabilities') is not None]
        if variants:
            best = max(variants, key=lambda v: evaluator.metrics.get(v, {}).get('accuracy', -1))
            bestModels.append(best)

    if not bestModels: return []

    title = 'Comparison of Best Model Variants'
    return evaluator.plotModelCompareRoc(bestModels, title, path, saveCsv)

def compareBestModelVariants(evaluator, path=None, saveCsv=False):
    return compareBestAcrossModelTypes(evaluator, path=path, saveCsv=saveCsv)
