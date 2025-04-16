import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from DataManager import DataManager
from NN_Model import NNModel
from Trainer import WriteCsv, PlotTrainingProgress
from collections import defaultdict

################################ INIT & SETUP ################################
class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.confusionMatrices = {}
        self.classNames = {}
        self.metrics = {}
        self.paramMapping = {
            'optimizer': 'optimizerName',
            'optimizerName': 'optimizerName',
            'lr': 'learningRate',
            'learningRate': 'learningRate',
            'wd': 'weightDecay',
            'l2_wd': 'l2WeightDecay',
            'weightDecay': 'weightDecay',
            'loss': 'lossFunction',
            'lossFunction': 'lossFunction',
            'dropout': 'dropoutRate',
            'dropoutRate': 'dropoutRate'
        }
        self.modelParamMappings = {
            NNModel: {
                'optimizer': 'optimizerName',
                'lr': 'learningRate',
                'wd': 'weightDecay',
                'dropout': 'dropoutRate',
                'loss': 'lossFunction'
            },
        }

    def addModel(self, name, model):
        self.models[name] = model
        self.results[name] = {}
        print(f"\nAdded model: {name}")

################################ EVALUATION ################################
    def evaluateModel(self, name, testData, batchSize=32):
        if name not in self.models:
            raise ValueError(f"\nModel '{name}' not found")
        
        model = self.models[name]
        print(f"\nEvaluating model: {name}")
        
        trueLabels = testData["label"].tolist()
        
        try:
            if hasattr(model, 'predict'):
                predictions = model.predict(testData["image"], returnProba=False)
                probabilities = model.predict(testData["image"], returnProba=True)
            elif hasattr(model, 'predictBatch'):
                predictions = model.predictBatch(testData["image"])
                probabilities = model.predictBatch(testData["image"], returnProba=True)
            elif hasattr(model, 'evaluate'):
                accuracy = model.evaluate(testData, batchSize)
                print(f"\nModel returned accuracy: {accuracy:.2f}%")
                return accuracy
            else:
                raise AttributeError("\nModel has no prediction method")
        except Exception as e:
            print(f"\nError during prediction: {e}")
            return 0
        
        accuracy = accuracy_score(trueLabels, predictions) * 100.0
        
        self.results[name] = {
            'predictions': predictions,
            'trueLabels': trueLabels,
            'accuracy': accuracy,
            'probabilities': probabilities
        }
        
        allClasses = sorted(set(trueLabels + predictions))
        self.classNames[name] = {i: cls for i, cls in enumerate(allClasses)}
        
        classToIdx = {cls: i for i, cls in enumerate(allClasses)}
        trueIdx = [classToIdx[label] for label in trueLabels]
        predIdx = [classToIdx.get(label, -1) for label in predictions]
        
        self.confusionMatrices[name] = confusion_matrix(trueIdx, predIdx, labels=range(len(allClasses)))
        self.metrics[name] = {'accuracy': accuracy}
        
        if probabilities is not None and len(allClasses) > 1:
            try:
                yTrueIdx = np.array([classToIdx[label] for label in trueLabels])
                yTrueBin = label_binarize(yTrueIdx, classes=range(len(allClasses)))
                nClasses = min(len(allClasses), probabilities.shape[1])
                yBin = yTrueBin[:, :nClasses]
                probs = probabilities[:, :nClasses]
                
                macroAuc = roc_auc_score(yBin, probs, multi_class='ovr', average='macro')
                microAuc = roc_auc_score(yBin, probs, multi_class='ovr', average='micro')
                
                self.metrics[name].update({'macroAuc': macroAuc, 'microAuc': microAuc})
            except Exception:
                self.metrics[name].update({'macroAuc': None, 'microAuc': None})
        
        print(f"\nAccuracy: {accuracy:.2f}%")
        return accuracy

    def evaluateModelVariants(self, modelClass, prefix, trainingData, numClasses=None, batchSize=32, numEpochs=5,
                                trainRatio=0.7, valRatio=0.15, testRatio=0.15, maxConfigs=9, modelDir="models",
                                saveBest=False, loadBest=False, savePath=None, saveCsv=False, modelType=None, variant=None):
        
        trainDf, valDf, testDf = self.splitDataset(trainingData, trainRatio, valRatio, testRatio)
        
        if numClasses is None and "label" in trainingData.columns:
            numClasses = len(trainingData["label"].unique())
        if modelClass == NNModel:
            baseModel = modelClass(numClasses=numClasses, modelDir=modelDir, modelType=modelType or "EfficientNet", variant=variant or 'b0')
        else:
            baseModel = modelClass(numClasses=numClasses, modelDir=modelDir, variant=variant or 'b0')
        
        try:
            if not hasattr(baseModel, 'findBestConfig'):
                raise AttributeError(f"{prefix} missing findBestConfig method")
            bestConfig, configs = baseModel.findBestConfig(
                df=trainDf, validationSplit=0.15, epochs=3, batchSize=batchSize, maxConfigs=maxConfigs
            )
        except Exception as e:
            print(f"\nConfig error: {e}")
            return pd.DataFrame()
        
        if not configs:
            return pd.DataFrame()
        
        results = []
        modelName = (modelType or modelClass.__name__).lower()
        bestModelPath = os.path.join(modelDir, f"{modelName}_best.pth")
        bestModelExistsAtStart = os.path.isfile(bestModelPath)
        resumeForAll = loadBest and bestModelExistsAtStart
        
        for i, config in enumerate(configs):
            configName = f"{prefix}_{modelType or ''}_{variant or ''}_{i+1}".replace('__', '_')
            try:
                mappedConfig = self.mapConfigForModel(config, modelClass)
                
                if modelClass == NNModel:
                    model = modelClass(numClasses=numClasses, modelDir=modelDir, modelType=modelType or "EfficientNet", variant=variant or 'b0', **mappedConfig)
                else:
                    model = modelClass(numClasses=numClasses, modelDir=modelDir, **mappedConfig)
                
                self.addModel(configName, model)
                epochStats = None
                
                if hasattr(model, 'train'):
                    epochStats = model.train(
                        df=trainDf, epochs=numEpochs, batchSize=batchSize,
                        saveInterval=numEpochs, loadModel=resumeForAll, saveModel=saveBest
                    )
                
                self.results[configName]['epochStats'] = epochStats if epochStats is not None else []
                accuracy = self.evaluateModel(configName, testDf, batchSize)
                
                if 'epochStats' in locals():
                    self.results[configName]['epochStats'] = epochStats if epochStats is not None else []
                
                resultRow = {'Model': configName, 'Accuracy': accuracy}
                
                if modelType: resultRow['ModelType'] = modelType
                if variant: resultRow['Variant'] = variant
                
                for origKey, value in config.items():
                    mappedKey = self.getParamMappingForDisplay(origKey)
                    resultRow[mappedKey] = value
                
                results.append(resultRow)
            except Exception as e:
                print(f"\nError with config {i+1}: {str(e)}")
                
                if configName not in self.results:
                    self.results[configName] = {}
                self.results[configName]['epochStats'] = []
        
        resultMatrix = pd.DataFrame(results)
        
        if not resultMatrix.empty:
            self.plotConfigMatrix(resultMatrix, prefix, savePath, saveCsv)
        return resultMatrix

    def evaluateArchitectures(self, trainingData, architectures, modelDir="models", numEpochs=5, batchSize=32, maxConfigs=2,
                                trainRatio=0.7, valRatio=0.15, testRatio=0.15, saveBest=False, loadBest=False, outputDir=None):
        if outputDir is not None and not os.path.exists(outputDir):
            os.makedirs(outputDir)
        
        numClasses = len(trainingData["label"].unique())
        resultMatrices = {}
        
        for arch in architectures:
            modelClass = arch['class']
            modelType = arch.get('model_type')
            variant = arch.get('variant')
            prefix = arch.get('prefix', modelClass.__name__)
        
            print(f"\n{'='*70}")
            print(f"Evaluating {prefix} {'with ' + modelType if modelType else ''} {variant or ''}")
            print(f"{'='*70}")
        
            savePath = None
        
            if outputDir:
                savePath = os.path.join(outputDir, f"{prefix}_{modelType or ''}_{variant or ''}_results".replace('__', '_'))
        
            resultMatrix = self.evaluateModelVariants(
                modelClass=modelClass,
                prefix=prefix,
                trainingData=trainingData,
                numClasses=numClasses,
                batchSize=batchSize,
                numEpochs=numEpochs,
                trainRatio=trainRatio,
                valRatio=valRatio,
                testRatio=testRatio,
                maxConfigs=maxConfigs,
                modelDir=modelDir,
                saveBest=saveBest,
                loadBest=loadBest,
                savePath=savePath,
                saveCsv=True,
                modelType=modelType,
                variant=variant
            )
            resultMatrices[f"{prefix}_{modelType or ''}_{variant or ''}".replace('__', '_')] = resultMatrix
        
        if outputDir and len(self.models) > 1:
            self.exportTestResults(os.path.join(outputDir, "all_models_summary.csv"), saveCsv=True)
            self.compareBestAcrossModelTypes(
                path=os.path.join(outputDir, "architecture_comparison"),
                saveCsv=True
            )
        return resultMatrices

    def mapConfigForModel(self, config, modelClass):
        mappedConfig = {}
        paramMap = self.modelParamMappings.get(modelClass, {})
        
        for key, value in config.items():
            mappedKey = paramMap.get(key, key)
            mappedConfig[mappedKey] = value
        return mappedConfig

    def getParamMappingForDisplay(self, key):
        return self.paramMapping.get(key, key)

################################ UTILITIES ################################
    def splitDataset(self, dataset, trainRatio=0.7, valRatio=0.15, testRatio=0.15, seed=42):
        if abs(trainRatio + valRatio + testRatio - 1.0) > 0.001:
            total = trainRatio + valRatio + testRatio
            trainRatio /= total
            valRatio /= total
            testRatio /= total
        
        if testRatio > 0:
            restData, testData = train_test_split(
                dataset, test_size=testRatio, random_state=seed,
                stratify=dataset["label"] if "label" in dataset.columns else None
            )
            testData = testData.reset_index(drop=True)
        else:
            restData = dataset
            testData = pd.DataFrame()
        
        if valRatio > 0:
            valRatioAdj = valRatio / (trainRatio + valRatio)
            trainData, valData = train_test_split(
                restData, test_size=valRatioAdj, random_state=seed,
                stratify=restData["label"] if "label" in restData.columns else None
            )
            trainData = trainData.reset_index(drop=True)
            valData = valData.reset_index(drop=True)
        else:
            trainData = restData.reset_index(drop=True)
            valData = pd.DataFrame()
        
        print(f"\nDataset split: Train: {len(trainData)}, Val: {len(valData)}, Test: {len(testData)}")
        return trainData, valData, testData

################################ PLOTTING & EXPORT ################################
    def savePlot(self, fig, path, suffix=""):
        if not path: return
        imgPath = path
        
        if suffix:
            base, ext = os.path.splitext(path)
            imgPath = f"{base}{suffix}{ext}" if ext else f"{base}{suffix}.png"
        elif not path.lower().endswith(('.png', '.jpg', '.jpeg')):
            imgPath = f"{path}.png"
        
        fig.savefig(imgPath, bbox_inches='tight')
        return imgPath

    def saveDfToCsv(self, df, path, suffix="", index=False):
        if not path or df.empty: return
        
        base, ext = os.path.splitext(path)
        csvPath = f"{base}{suffix}.csv"
        df.to_csv(csvPath, index=index)
        
        return csvPath

    def plotConfusionMatrix(self, name, path=None, saveCsv=False):
        if name not in self.confusionMatrices or name not in self.classNames: return
        
        cm = self.confusionMatrices[name]
        
        classNames = list(self.classNames[name].values())
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
            self.savePlot(fig, path)
            if saveCsv:
                normDf = pd.DataFrame(cmNorm, index=classNames, columns=classNames)
                self.saveDfToCsv(normDf, path, "_normalized", index=True)
        plt.show()
        return cm, cmNorm

    def plotConfigMatrix(self, matrix, modelType, path=None, saveCsv=False):
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
                    self.savePlot(fig, path)
                    if saveCsv:
                        self.saveDfToCsv(matrix, path, "_full_results")
                plt.show()
                return
            except:
                pass
        self.basePlot(matrix, modelType, path, saveCsv)

    def basePlot(self, matrix, modelType="Model", path=None, saveCsv=False):
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
            self.savePlot(fig, path)
            if saveCsv:
                self.saveDfToCsv(matrix, path, "_results")
        plt.show()

    def plotRocCurve(self, name, path=None, saveCsv=False):
        if name not in self.results or self.results[name].get('probabilities') is None: return
        
        probs = self.results[name]['probabilities']
        yTrue = self.results[name]['trueLabels']
        classes = list(self.classNames[name].values())
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
        
                self.metrics[name].update({'macroAuc': macroAuc, 'microAuc': microAuc})
        except Exception as e:
            print(f"Error calculating AUC metrics: {e}")
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set(xlim=[0, 1], ylim=[0, 1.05], title=f'ROC Curves for {name}', xlabel='False Positive Rate', ylabel='True Positive Rate')
        ax.legend(loc="best", fontsize='small', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        
        if path and rocData['class']:
            self.savePlot(fig, path)
            if saveCsv:
                allRocData = []
                for i, cls in enumerate(rocData['class']):
                    df = pd.DataFrame({'Class': cls, 'FPR': rocData['fpr'][i], 'TPR': rocData['tpr'][i], 'AUC': rocData['auc'][i]})
                    allRocData.append(df)
                if allRocData:
                    finalDf = pd.concat(allRocData, ignore_index=True)
                    self.saveDfToCsv(finalDf, path, "_roc")
        plt.show()
        return rocData

    def computeModelRoc(self, name):
        probs = self.results[name]['probabilities']
        trueLabels = self.results[name]['trueLabels']
        classes = list(self.classNames[name].values())
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

    def plotModelCompareRoc(self, models, title, path=None, saveCsv=False):
        fig, ax = plt.subplots(figsize=(10, 8))
        compareData = {'model': [], 'fpr': [], 'tpr': [], 'auc': [], 'accuracy': []}
        
        for model in models:
            try:
                fpr, tpr, rocAuc = self.computeModelRoc(model)
        
                ax.plot(fpr, tpr, lw=2, label=f'{model} (AUC = {rocAuc:.2f})')
                compareData['model'].append(model)
                compareData['fpr'].append(fpr)
                compareData['tpr'].append(tpr)
                compareData['auc'].append(rocAuc)
                compareData['accuracy'].append(self.metrics.get(model, {}).get('accuracy', -1))
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
            self.savePlot(fig, path)
        
            if saveCsv:
                allData = []
        
                for i, model in enumerate(compareData['model']):
                    df = pd.DataFrame({'Model': model, 'FPR': compareData['fpr'][i], 'TPR': compareData['tpr'][i], 'AUC': compareData['auc'][i]})
                    allData.append(df)
        
                if allData:
                    fullDf = pd.concat(allData, ignore_index=True)
                    self.saveDfToCsv(fullDf, path, "_roc")
        
                summaryDf = pd.DataFrame({'Model': compareData['model'], 'AUC': compareData['auc'], 'Accuracy(%)': compareData['accuracy']})
                self.saveDfToCsv(summaryDf, path, "_summary")
        plt.show()
        return compareData

################################ COMPARISON ################################
    def compareModels(self):
        if not self.metrics: return None
        
        metricsData = [{'Model': n, 'Accuracy (%)': m['accuracy']} for n, m in self.metrics.items()]
        df = pd.DataFrame(metricsData)
        
        plt.figure(figsize=(15, 10))
        values = [self.metrics[model]['accuracy'] for model in self.metrics]
        bars = plt.bar(list(self.metrics.keys()), values, color='skyblue')
        
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

    def compareModelTypeVariants(self, modelType, path=None, saveCsv=False):
        variants = [m for m in self.results.keys() if m.startswith(f"{modelType}_") and self.results[m].get('probabilities') is not None]
        
        if not variants: return []
        title = f'ROC Comparison of {modelType} Variants'
        
        return self.plotModelCompareRoc(variants, title, path, saveCsv)

    def compareBestAcrossModelTypes(self, modelTypes=None, path=None, saveCsv=False):
        if not modelTypes:
            modelTypes = set()
        
            for model in self.results.keys():
        
                if '_' in model and self.results[model].get('probabilities') is not None:
                    modelTypes.add(model.split('_')[0])
        
        if not modelTypes: return []
        bestModels = []
        
        for modelType in modelTypes:
            variants = [m for m in self.results.keys() if m.startswith(f"{modelType}_") and self.results[m].get('probabilities') is not None]
        
            if variants:
                best = max(variants, key=lambda v: self.metrics.get(v, {}).get('accuracy', -1))
                bestModels.append(best)
        
        if not bestModels: return []
        
        title = 'Comparison of Best Model Variants'
        return self.plotModelCompareRoc(bestModels, title, path, saveCsv)

    def compareBestModelVariants(self, path=None, saveCsv=False):
        return self.compareBestAcrossModelTypes(path=path, saveCsv=saveCsv)

###################### EXPORT ######################
    def exportAllConfusionMatrices(self, outputDir, saveCsv=False):
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        
        if not self.confusionMatrices: return
        
        for name in self.confusionMatrices.keys():
        
            if name in self.classNames:
                path = os.path.join(outputDir, f"{name}_confusion_matrix")
                self.plotConfusionMatrix(name, path=path, saveCsv=saveCsv)

    def exportTestResults(self, outputPath, saveCsv=True):
        if not self.metrics or not saveCsv: return
        
        metricsData = []
        
        for name, metricDict in self.metrics.items():
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

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Test 1: ModelEvaluator")
    print("="*70)
    
    dm = DataManager()
    currentFolder = os.getcwd()
    dataFolder = os.path.join(currentFolder, "../data/")
    modelDir = os.path.join(currentFolder, "../models")
    outputDir = os.path.join(currentFolder, "../output")
    os.makedirs(modelDir, exist_ok=True)
    os.makedirs(outputDir, exist_ok=True)
    
    print("\nLoading training data...")
    classes = ["sitting", "running", "drinking", "eating"]
    dm.LoadTrainingData(folderName=dataFolder, csvFileName="Training_set.csv", numFiles=100, classFilter=classes)
    dm.RemoveMissingData()
    
    numClasses = len(dm.TrainingData["label"].unique())
    print(f"Loaded {len(dm.TrainingData)} samples with {numClasses} classes")
    evaluator = ModelEvaluator()
    
    print("\n" + "="*70)
    print("Test 2: Comparing Different Model Architectures")
    print("="*70)
    
    architectures = [
        {'class': NNModel, 'model_type': 'EfficientNet', 'variant': 'b0', 'prefix': 'NNEfficientNet'},
        {'class': NNModel, 'model_type': 'ResNet', 'variant': '18', 'prefix': 'NNResNet'},
    ]
    
    results = evaluator.evaluateArchitectures(
        trainingData=dm.TrainingData,
        architectures=architectures,
        modelDir=modelDir,
        numEpochs=3,
        batchSize=8,
        maxConfigs=2,
        outputDir=os.path.join(outputDir, "architecture_comparison"),
        loadBest=True,
        saveBest=True,
    )
    
    print("\n" + "="*70)
    print("Test 3: Comparing Best Models Across Architectures")
    print("="*70)
    
    evaluator.compareModels()
    evaluator.compareBestAcrossModelTypes(
        path=os.path.join(outputDir, "best_architecture_comparison"),
        saveCsv=True
    )
    
    print("\n" + "="*70)
    print("Test 4: Exporting confusion matrices and test results")
    print("="*70)
    
    evaluator.exportAllConfusionMatrices(outputDir, saveCsv=True)
    evaluator.exportTestResults(os.path.join(outputDir, "test_results.csv"), saveCsv=True)
    
    print("\nPlotting ROC curves for all models...")
    for modelName in evaluator.results:
        evaluator.plotRocCurve(
            modelName,
            path=os.path.join(outputDir, f"{modelName}_roc"),
            saveCsv=True
        )
    
    print("\nPlotting confusion matrices for all models...")
    for modelName in evaluator.confusionMatrices:
        evaluator.plotConfusionMatrix(
            modelName,
            path=os.path.join(outputDir, f"{modelName}_cm"),
            saveCsv=True
        )
    
    print("\nAll analysis and plots completed.")
    print("\n" + "="*70)
    print("ModelEvaluator tests completed")
    print("="*70)
    print("\n" + "="*70)
    print("Test 5: Saving and plotting epoch stats for all models/configs")
    print("="*70)
    
    for modelName, model in evaluator.models.items():
        epochStats = None
        if modelName in evaluator.results and 'epochStats' in evaluator.results[modelName]:
            epochStats = evaluator.results[modelName]['epochStats']
        elif hasattr(model, 'epochStats'):
            epochStats = getattr(model, 'epochStats')
        elif hasattr(model, 'trainingHistory'):
            epochStats = getattr(model, 'trainingHistory')
        
        csvFile = os.path.join(outputDir, f"{modelName}_epoch_stats.csv")
        csvExists = os.path.isfile(csvFile)
        
        if epochStats and len(epochStats) > 0 and not csvExists:
            print(f"Case 1: Training from scratch for {modelName}. Saving all epochs.")
            WriteCsv(epochStats, csvFile)
        elif (not epochStats or len(epochStats) == 0) and csvExists:
            print(f"Case 2: Loaded model for {modelName}.")
        elif epochStats and len(epochStats) > 0 and csvExists:
            print(f"Case 3: Resumed training for {modelName}. Appending new epochs.")
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
        else:
            print(f"No epoch stats found for {modelName}")
        
            if modelName in evaluator.results:
                print(f"  - evaluator.results[{modelName}]: {evaluator.results[modelName].keys()}")
                if 'epochStats' in evaluator.results[modelName]:
                    print(f"  - epochStats length: {len(evaluator.results[modelName]['epochStats'])}")
            else:
                print(f"  - Model {modelName} not found in evaluator.results")
            print("  - Possible reasons: training failed, exception occurred, or model.train() did not return stats.")
    
    print("\nAll epoch stats saved.")
    print("\n" + "="*70)
    print("ModelEvaluator tests completed")
    print("="*70)
    
    plotAllVariantsTrainingCurves(evaluator, outputDir)