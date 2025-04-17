import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from NN_Model import NNModel
from PlotUltilities import plotConfusionMatrix, plotConfigMatrix, basePlot, plotRocCurve, computeModelRoc, plotModelCompareRoc
from DataUltilities import splitDataset
from ExportUtilities import savePlot, saveDfToCsv, exportAllConfusionMatrices, exportTestResults
from ModelComp import  compareModels, compareModelTypeVariants, compareBestAcrossModelTypes, compareBestModelVariants

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
                                saveBest=False, loadBest=False, savePath=None, saveCsv=False, modelType=None, variant=None, targetAccuracy=None):
        
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
                        saveInterval=numEpochs, loadModel=resumeForAll, saveModel=saveBest,
                        targetAccuracy=targetAccuracy
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
                                trainRatio=0.7, valRatio=0.15, testRatio=0.15, saveBest=False, loadBest=False, outputDir=None, targetAccuracy=None):
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
                variant=variant,
                targetAccuracy=targetAccuracy
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
        return splitDataset(dataset, trainRatio, valRatio, testRatio, seed)

################################ PLOTTING & EXPORT ################################
    def savePlot(self, fig, path, suffix=""):
        return savePlot(fig, path, suffix)

    def saveDfToCsv(self, df, path, suffix="", index=False):
        return saveDfToCsv(df, path, suffix, index)

    def plotConfusionMatrix(self, name, path=None, saveCsv=False):
        return plotConfusionMatrix(self, name, path, saveCsv)

    def plotConfigMatrix(self, matrix, modelType, path=None, saveCsv=False):
        return plotConfigMatrix(self, matrix, modelType, path, saveCsv)

    def basePlot(self, matrix, modelType="Model", path=None, saveCsv=False):
        return basePlot(self, matrix, modelType, path, saveCsv)

    def plotRocCurve(self, name, path=None, saveCsv=False):
        return plotRocCurve(self, name, path, saveCsv)

    def computeModelRoc(self, name):
        return computeModelRoc(self, name)

    def plotModelCompareRoc(self, models, title, path=None, saveCsv=False):
        return plotModelCompareRoc(self, models, title, path, saveCsv)

################################ COMPARISON ################################
    def compareModels(self):
        return compareModels(self)

    def compareModelTypeVariants(self, modelType, path=None, saveCsv=False):
        return compareModelTypeVariants(self, modelType, path, saveCsv)

    def compareBestAcrossModelTypes(self, modelTypes=None, path=None, saveCsv=False):
        return compareBestAcrossModelTypes(self, modelTypes, path, saveCsv)

    def compareBestModelVariants(self, path=None, saveCsv=False):
        return compareBestModelVariants(self, path, saveCsv)

###################### EXPORT ######################
    def exportAllConfusionMatrices(self, outputDir, saveCsv=False):
        return exportAllConfusionMatrices(self, outputDir, saveCsv)

    def exportTestResults(self, outputPath, saveCsv=True):
        return exportTestResults(self, outputPath, saveCsv)