import os
import pandas as pd

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

def saveDfToCsv(df, path, suffix="", index=False):
    if not path or df.empty: return

    base, ext = os.path.splitext(path)
    csvPath = f"{base}{suffix}.csv"
    df.to_csv(csvPath, index=index)

    return csvPath

def exportAllConfusionMatrices(evaluator, outputDir, saveCsv=False):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    if not evaluator.confusionMatrices: return

    for name in evaluator.confusionMatrices.keys():
        if name in evaluator.classNames:
            path = os.path.join(outputDir, f"{name}_confusion_matrix")
            evaluator.plotConfusionMatrix(name, path=path, saveCsv=saveCsv)

def exportTestResults(evaluator, outputPath, saveCsv=True):
    if not evaluator.metrics or not saveCsv: return

    metricsData = []

    for name, metricDict in evaluator.metrics.items():
        row = {'Model': name}
        standardMetrics = ['accuracy', 'macroAuc', 'microAuc']

        for metric in standardMetrics:
            if metric in metricDict and metricDict[metric] is not None:
                if metric == 'accuracy':
                    row[metric] = f"{metricDict[metric]:.2f}"
                else:
                    row[metric] = f"{metricDict[metric]:.4f}"
        for k, v in metricDict.items():
            if k not in standardMetrics and v is not None:
                row[k] = v
        metricsData.append(row)

    resultDf = pd.DataFrame(metricsData)
    resultDf = resultDf.dropna(axis=1, how='all')
    outputDir = os.path.dirname(outputPath)

    if outputDir and not os.path.exists(outputDir):
        os.makedirs(outputDir)

    resultDf.to_csv(outputPath, index=False)
    return resultDf
