import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import math
from Model import Model
from DataManager import DataFrameImage, DataManager
from ModelManager import ModelManager
from Timer import Timer

################################ INIT & SETUP ################################
class NNModel(Model):
    def __init__(self, numClasses, modelType="EfficientNet", variant='b0', modelDir="models",
                    optimizerName='Adam', learningRate=0.001, weightDecay=0.0, lossFunction='cross_entropy',
                    dropoutRate=0.2, useRegularization=False):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.modelType = modelType.lower()
        self.variant = variant
        self.modelDir = modelDir
        self.dropoutRate = dropoutRate
        self.optimizerName = optimizerName.lower()
        self.learningRate = learningRate
        self.weightDecay = weightDecay if useRegularization else 0.0
        self.lossFunctionName = lossFunction
        self.classToIdx = None
        self.idxToClass = None

        print(f"Using device: {self.device}")

        self.initModel(numClasses)
        self.setupTransform()
        self.modelManager = ModelManager(model=self.model, optimizer=self.optimizer, device=self.device, model_dir=self.modelDir)
        
        print("\n" + "="*50)
        self.printModelInfo()

    def initModel(self, numClasses):
        print(f"\nLoading {self.modelType}-{self.variant} model...")
        modelInitializers = {"efficientnet": self.initEfficientNet, "resnet": self.initResNet}
        
        if self.modelType not in modelInitializers:
            raise ValueError(f"Unsupported model type: {self.modelType}. Supported: {list(modelInitializers.keys())}")
        
        modelInitializers[self.modelType](numClasses)
        self.model = self.model.to(self.device)
        
        lossFunctions = {'cross_entropy': nn.CrossEntropyLoss(), 'label_smoothing': nn.CrossEntropyLoss(label_smoothing=0.1)}
        
        if self.lossFunctionName not in lossFunctions:
            raise ValueError(f"Unsupported loss: {self.lossFunctionName}. Supported: {list(lossFunctions.keys())}")
        self.criterion = lossFunctions[self.lossFunctionName].to(self.device)
        
        optimizerCreators = {
            'adam': lambda: optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay),
            'sgd': lambda: optim.SGD(self.model.parameters(), lr=self.learningRate, momentum=0.9, nesterov=True, weight_decay=self.weightDecay),
            'adamw': lambda: optim.AdamW(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        }
        
        if self.optimizerName not in optimizerCreators:
            print(f"Warning: Unsupported optimizer '{self.optimizerName}', defaulting to Adam")
            self.optimizerName = 'adam'
        
        self.optimizer = optimizerCreators[self.optimizerName]()

    def initEfficientNet(self, numClasses):
        modelBuilders = {'b0': models.efficientnet_b0, 'b1': models.efficientnet_b1, 'b4': models.efficientnet_b4,
                            'b5': models.efficientnet_b5, 'b6': models.efficientnet_b6, 'b7': models.efficientnet_b7}
        
        if self.variant not in modelBuilders:
            raise ValueError(f"Unsupported EfficientNet variant: {self.variant}. Supported: {list(modelBuilders.keys())}")
        
        self.model = modelBuilders[self.variant](weights="DEFAULT")
        numFeatures = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(nn.Dropout(self.dropoutRate), nn.Linear(numFeatures, numClasses)) if self.dropoutRate > 0 else nn.Linear(numFeatures, numClasses)

    def initResNet(self, numClasses):
        modelBuilders = {'18': models.resnet18, '34': models.resnet34, '50': models.resnet50, '101': models.resnet101, '152': models.resnet152}
        
        if self.variant not in modelBuilders:
            raise ValueError(f"Unsupported ResNet variant: {self.variant}. Supported: {list(modelBuilders.keys())}")
        
        self.model = modelBuilders[self.variant](weights="DEFAULT")
        numFeatures = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(self.dropoutRate), nn.Linear(numFeatures, numClasses)) if self.dropoutRate > 0 else nn.Linear(numFeatures, numClasses)

    def setupTransform(self):
        commonTransforms = [transforms.ToPILImage(), transforms.Resize((260, 260)), transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        
        trainAug = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), 
                    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),]
        
        self.transform = transforms.Compose([transforms.ToPILImage()] + trainAug + commonTransforms[1:])
        self.evalTransform = transforms.Compose(commonTransforms)

    def printModelInfo(self):
        regMethods = []
        
        if self.dropoutRate > 0: regMethods.append(f"Dropout ({self.dropoutRate})")
        if self.weightDecay > 0: regMethods.append(f"L2 regularization (weightDecay={self.weightDecay})")
        
        regSummary = ", ".join(regMethods) if regMethods else "None"
        print(f"\nModel: {self.modelType}-{self.variant}")
        print(f"Training configuration:")
        print(f"  - Optimizer: {self.optimizerName}")
        print(f"  - Learning rate: {self.learningRate}")
        print(f"  - Loss function: {self.lossFunctionName}")
        print(f"  - Regularization: {regSummary}")

################################ DATA PREP ################################
    def preprocImg(self, image, isTraining=True):
        imgArr = image.image if isinstance(image, DataFrameImage) else image
        
        if imgArr.ndim == 2: imgArr = np.stack([imgArr] * 3, axis=2)
        
        transform = self.transform if isTraining else self.evalTransform
        return transform(imgArr)

    def setupMapMapping(self, df):
        if "label" not in df.columns:
            if self.classToIdx is not None:
                print("\nUsing existing class mapping as input df has no labels")
                return
            raise ValueError("No class mapping exists and input dataframe has no labels")
        
        uniqueClasses = sorted(df["label"].unique())
        
        self.classToIdx = {c: i for i, c in enumerate(uniqueClasses)}
        self.idxToClass = {i: c for i, c in enumerate(uniqueClasses)}
        print(f"\nSet up mapping for {len(uniqueClasses)} classes: {uniqueClasses}")

    def prepBatch(self, dataframe, batchIndices, isTraining=True):
        if self.classToIdx is None: raise ValueError("Class mapping not initialized. Call setupMapMapping() first")
        
        batchData = dataframe.iloc[batchIndices]
        imgTensors = [self.preprocImg(img, isTraining) for img in batchData["image"]]
        imgBatch = torch.stack(imgTensors).to(self.device)
        
        labelIndices = [self.classToIdx[label] for label in batchData["label"]]
        labelTensor = torch.tensor(labelIndices).to(self.device)
        
        return imgBatch, labelTensor

    def preprocess(self, df):
        if df is None or len(df) == 0: raise ValueError("Error: DataFrame is None or empty")
        
        try:
            sampleImg = df["image"].iloc[0]
            imgArr = sampleImg.image if isinstance(sampleImg, DataFrameImage) else sampleImg
            if imgArr.ndim < 2: raise ValueError("Error: Images must be 2D or 3D arrays")
        except Exception as e:
            raise ValueError(f"Error examining images: {str(e)}")
        
        return df

# === TRAINING ===
    def train(self, df, epochs=10, batchSize=32, saveInterval=5, loadModel=True, saveModel=True, targetAccuracy=90):
        startTime = time.time()
        
        self.setupMapMapping(df)
        self.model.train()
        
        nSamples = len(df)
        indices = np.arange(nSamples)
        numBatches = (nSamples + batchSize - 1) // batchSize
        startEpoch, bestLoss = 0, float('inf')
        
        if loadModel:
            startEpoch, bestLoss = self.loadModel(self.modelType)
            print(f"\n{'Resuming training from epoch ' + str(startEpoch + 1) + ' with best loss: ' + f'{bestLoss:.4f}' if startEpoch > 0 else 'Starting training from beginning'}")
        
        print(f"\nTraining plan: {epochs} epochs, {nSamples} samples, {numBatches} batches per epoch")
        epochStats = []
        
        for epoch in range(startEpoch, startEpoch + epochs):
            print(f"\nEpoch {epoch + 1}/{startEpoch + epochs}")
            print("Progress: [", end="")
            
            epochStartTime = time.time()
            epochLoss, epochAcc = self.trainEpoch(df, indices, batchSize, numBatches)
            
            print("]")
            
            epochTime = time.time() - epochStartTime
            self.displayEpochResults(epoch, startEpoch + epochs, epochTime, epochLoss, epochAcc)
            epochStats.append([epoch + 1, epochTime, epochLoss, epochAcc])
            
            if saveModel:
                if epochLoss < bestLoss:
                    bestLoss = epochLoss
                    self.saveModel(epoch + 1, epochLoss, self.modelType, best=True)
                    print(f"\nNew best model saved with loss: {epochLoss:.4f}")
                elif (epoch + 1) % saveInterval == 0:
                    self.saveModel(epoch + 1, epochLoss, self.modelType, best=False)
                    print(f"\nCheckpoint saved at epoch {epoch + 1}")
            if targetAccuracy is not None and epochAcc >= targetAccuracy:
                print(f"\nTarget accuracy {targetAccuracy}% reached!")
                if saveModel:
                    self.saveModel(epoch + 1, epochLoss, self.modelType, best=True)
                    print(f"Final model saved with loss: {epochLoss:.4f}")
                break
        
        totalTime = time.time() - startTime
        print(f"\nTotal training time: {totalTime:.1f}s")
        return epochStats

    def trainEpoch(self, df, indices, batchSize, numBatches):
        nSamples = len(indices)
        runningLoss = 0.0
        correct = 0
        total = 0
        np.random.shuffle(indices)
        
        for i in range(0, nSamples, batchSize):
            batchIndices = indices[i:i + batchSize]
            inputs, labels = self.prepBatch(df, batchIndices, isTraining=True)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            self.optimizer.step()
            runningLoss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            batchSz = labels.size(0)
            
            correct += (predicted == labels).sum().item()
            total += batchSz
            currentBatch = (i // batchSize) + 1
            
            self.displayProcess(currentBatch, numBatches)
        epochLoss = runningLoss / numBatches
        epochAcc = 100 * correct / total if total > 0 else 0
        return epochLoss, epochAcc

    def displayProcess(self, currentBatch, numBatches):
        displayInterval = max(1, numBatches // 20)
        
        if currentBatch % displayInterval == 0 or currentBatch == numBatches:
            progress = int(30 * currentBatch / numBatches)
            numSymbols = progress - len(str(currentBatch)) - len(str(numBatches)) - 3
            print("=" * max(0, numSymbols), end="")
            print(f" {currentBatch}/{numBatches} ", end="", flush=True)

    def displayEpochResults(self, epoch, totalEpochs, epochTime, epochLoss, epochAcc):
        print(f"\nEpoch {epoch+1}/{totalEpochs} completed in {epochTime:.1f}s - Loss: {epochLoss:.4f}, Accuracy: {epochAcc:.2f}%")
        
        remainingEpochs = totalEpochs - (epoch + 1)
        
        if remainingEpochs > 0:
            estTime = epochTime * remainingEpochs
            h, rem = divmod(estTime, 3600)
            m, s = divmod(rem, 60)
            print(f"Estimated remaining time: {int(h)}h {int(m)}m {int(s)}s")

################################ PREDICTION & EVAL ################################
    def predict(self, image=None, returnProba=False):
        if image is None: return None
        
        self.model.eval()
        
        if isinstance(image, pd.Series):
            return self.predictBatch(image, returnProba)
        else:
            return self.predictSingle(image, returnProba)

    def predictSingle(self, image, returnProba=False):
        with torch.no_grad():
            imgTensor = self.preprocImg(image, isTraining=False)
            imgTensor = imgTensor.unsqueeze(0).to(self.device)
            
            outputs = self.model(imgTensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            if returnProba: return probabilities.cpu().numpy()
            
            _, predictedIdx = torch.max(outputs, 1)
            predictedClass = predictedIdx.item()
            
            if self.idxToClass is not None:
                predictedLabel = self.idxToClass[predictedClass]
                confidence = probabilities[0][predictedClass].item()
                print(f"\nPrediction: {predictedLabel} (confidence: {confidence:.2f})")
                return predictedLabel
            else:
                return predictedClass

    def predictBatch(self, images, returnProba=False):
        outputsAll = []
        
        with torch.no_grad():
            for img in images:
                imgTensor = self.preprocImg(img, isTraining=False)
                imgTensor = imgTensor.unsqueeze(0).to(self.device)
                
                outputs = self.model(imgTensor)
                
                if returnProba:
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    outputsAll.append(probs.cpu().numpy())
                else:
                    _, predictedIdx = torch.max(outputs, 1)
                    predicted = self.idxToClass[predictedIdx.item()] if self.idxToClass else predictedIdx.item()
                    outputsAll.append(predicted)
        if returnProba and outputsAll:
            return np.vstack(outputsAll)
        
        return outputsAll

    def evaluate(self, df, batchSize=32):
        self.model.eval()
        nSamples = len(df)
        indices = np.arange(nSamples)
        predictions, labels, confidences = [], [], []
        correct = 0
        total = 0
        
        if self.classToIdx is None and 'label' in df.columns:
            self.setupMapMapping(df)
        
        with torch.no_grad():
            for i in range(0, nSamples, batchSize):
                batchIndices = indices[i:i+batchSize]
                inputs, labelBatch = self.prepBatch(df, batchIndices, isTraining=False)
                
                outputs = self.model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
                predictedNp = predicted.cpu().numpy()
                labelsNp = labelBatch.cpu().numpy()
                
                conf = [probabilities[i, predicted[i]].item() for i in range(len(predicted))]
                total += labelBatch.size(0)
                correct += (predicted == labelBatch).sum().item()
                predictions.extend(predictedNp)
                labels.extend(labelsNp)
                confidences.extend(conf)

        preds = [self.idxToClass[idx] for idx in predictions] if self.idxToClass else predictions
        trues = [self.idxToClass[idx] for idx in labels] if self.idxToClass else labels
        
        resultsDf = pd.DataFrame({'true_label': trues, 'predicted_label': preds, 'confidence': confidences, 'correct': [p == l for p, l in zip(preds, trues)]})
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"\nEvaluation completed: {correct}/{total} correct ({accuracy:.2f}%)")
        
        return accuracy, resultsDf

################################ HYPERPARAM SEARCH ################################
    def findBestConfig(self, df, validationSplit=0.2, epochs=5, batchSize=32, maxConfigs=5, learningRates=None, weightDecays=None, dropoutRates=None, optimizers=None):
        print("\nStarting hyperparameter tuning...")
        
        trainDf, valDf = self.splitData(df, validationSplit)
        
        print(f"\nSplit data: {len(trainDf)} training samples, {len(valDf)} validation samples")
        
        configs = self.generateConfig(
            maxConfigs=maxConfigs,
            learningRates=learningRates or [0.01, 0.001, 0.0001],
            weightDecays=weightDecays or [0.0, 0.0001],
            dropoutRates=dropoutRates or [0.0, 0.2],
            optimizers=optimizers or ['adam', 'sgd', 'adamw']
        )
        
        if not configs:
            print("\nNo configurations generated. Using current configuration.")
            currentConfig = {'optimizer': self.optimizerName, 'lr': self.learningRate, 'wd': self.weightDecay, 'dropout': self.dropoutRate}
            
            return currentConfig, [currentConfig]
        
        if self.classToIdx is None:
            self.setupMapMapping(df)
        
        results = []
        bestValAcc = 0
        bestConfig = None
        bestModelState = None
        
        print("\nTesting configurations:")
        print("-" * 80)
        print(f"{'#':^5} {'Optimizer':^10} {'LR':^10} {'Weight Decay':^12} {'Dropout':^8} {'Val Accuracy':^12}")
        print("-" * 80)
        
        for i, config in enumerate(configs):
            self.resetModel(
                numClasses=len(self.classToIdx),
                optimizerName=config['optimizer'],
                learningRate=config['lr'],
                weightDecay=config['wd'],
                dropoutRate=config['dropout'],
            )
            
            self.model.train()
            
            for _ in range(epochs):
                self.trainEpoch(trainDf, np.arange(len(trainDf)), batchSize, (len(trainDf) + batchSize - 1) // batchSize)
            
            valAcc, _ = self.evaluate(valDf, batchSize)
            results.append({'config': config, 'val_acc': valAcc})
            
            print(f"{i:^5} {config['optimizer']:^10} {config['lr']:<10.6f} {config['wd']:<12.6f} {config['dropout']:<8.1f} {valAcc:<12.2f}%")
            
            if valAcc > bestValAcc:
                bestValAcc = valAcc
                bestConfig = config
                bestModelState = self.model.state_dict().copy()
        
        if bestConfig is None:
            print("\nWarning: No configuration improved performance. Using first configuration.")
            bestConfig = configs[0]
        print("-" * 80)
        print(f"\nBest configuration: {bestConfig} with validation accuracy: {bestValAcc:.2f}%")
        print("\nTop configurations:")
        
        results.sort(key=lambda x: x['val_acc'], reverse=True)
        
        for i, result in enumerate(results[:min(5, len(results))]):
            print(f"{i+1}. {result['config']} - Accuracy: {result['val_acc']:.2f}%")
        self.resetModel(
            numClasses=len(self.classToIdx),
            optimizerName=bestConfig['optimizer'],
            learningRate=bestConfig['lr'],
            weightDecay=bestConfig['wd'],
            dropoutRate=bestConfig['dropout'],
        )
        
        if bestModelState is not None:
            self.model.load_state_dict(bestModelState)
        return bestConfig, configs

    def resetModel(self, numClasses, optimizerName, learningRate, weightDecay, dropoutRate):
        self.optimizerName = optimizerName
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.dropoutRate = dropoutRate
        
        if self.modelType == "efficientnet":
            numFeatures = self.model.classifier[1].in_features if hasattr(self.model.classifier[1], 'in_features') else self.model.classifier[1][1].in_features
            self.model.classifier[1] = nn.Sequential(nn.Dropout(self.dropoutRate), nn.Linear(numFeatures, numClasses)) if self.dropoutRate > 0 else nn.Linear(numFeatures, numClasses)
        else:
            numFeatures = self.model.fc.in_features if hasattr(self.model.fc, 'in_features') else self.model.fc[1].in_features
            self.model.fc = nn.Sequential(nn.Dropout(self.dropoutRate), nn.Linear(numFeatures, numClasses)) if self.dropoutRate > 0 else nn.Linear(numFeatures, numClasses)
        
        self.model = self.model.to(self.device)
        
        optimizerCreators = {
            'adam': lambda: optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay),
            'sgd': lambda: optim.SGD(self.model.parameters(), lr=self.learningRate, momentum=0.9, nesterov=True, weight_decay=self.weightDecay),
            'adamw': lambda: optim.AdamW(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        }
        self.optimizer = optimizerCreators[self.optimizerName]()

    def splitData(self, df, validationSplit):
        try:
            trainDf, valDf = train_test_split(df, test_size=validationSplit, random_state=42, stratify=df["label"] if "label" in df.columns else None)
        except ValueError as e:
            print(f"\nWarning: Stratified split failed ({str(e)}). Using regular random split.")
            trainDf, valDf = train_test_split(df, test_size=validationSplit, random_state=42)
        
        trainDf = trainDf.reset_index(drop=True)
        valDf = valDf.reset_index(drop=True)
        
        if "label" in df.columns:
            uniqueClasses = set(df["label"].unique())
            trainClasses = set(trainDf["label"].unique())
            valClasses = set(valDf["label"].unique())
            
            missingTrain = uniqueClasses - trainClasses
            missingVal = uniqueClasses - valClasses
            
            if missingTrain or missingVal:
                print(f"\nWarning: Split resulted in uneven class distribution:")
                if missingTrain: print(f"- Classes missing in training: {missingTrain}")
                if missingVal: print(f"- Classes missing in validation: {missingVal}")
        return trainDf, valDf

    def generateConfig(self, maxConfigs, learningRates, weightDecays, dropoutRates, optimizers):
        fixedDropout = dropoutRates[0] if dropoutRates else 0.0
        fixedWd = weightDecays[0] if weightDecays else 0.0
        
        configs = []
        
        for optimizer in optimizers:
            for lr in learningRates:
                configs.append({'optimizer': optimizer, 'lr': lr, 'wd': fixedWd, 'dropout': fixedDropout})
                if len(configs) >= maxConfigs: break
            if len(configs) >= maxConfigs: break
        
        print(f"\nGenerated {len(configs)} configurations to try (fixed dropout={fixedDropout}, wd={fixedWd})")
        return configs

# === VISUALIZATION ===
    def plotTrainingHistory(self, epochStats):
        if not epochStats: return
        
        try:
            epochs = [stat[0] for stat in epochStats]
            times = [stat[1] for stat in epochStats]
            losses = [stat[2] for stat in epochStats]
            accs = [stat[3] for stat in epochStats]
            totalTime = sum(times)
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            axes[0].plot(epochs, losses, 'o-', color="red", linewidth=2, markersize=6)
            axes[0].set_title("Training Loss", fontsize=14)
            axes[0].set_xlabel("Epoch", fontsize=12)
            axes[0].set_ylabel("Loss", fontsize=12)
            axes[0].grid(True, linestyle='--', alpha=0.7)
            
            axes[1].plot(epochs, accs, 'o-', color="blue", linewidth=2, markersize=6)
            axes[1].set_title("Training Accuracy", fontsize=14)
            axes[1].set_xlabel("Epoch", fontsize=12)
            axes[1].set_ylabel("Accuracy (%)", fontsize=12)
            axes[1].grid(True, linestyle='--', alpha=0.7)
            axes[1].set_ylim(bottom=0)
            
            fig.suptitle(f"{self.modelType}-{self.variant} Training History\nTotal time: {totalTime:.1f}s, Final accuracy: {accs[-1]:.2f}%", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            plt.show()
        except Exception as e:
            print(f"Error plotting training history: {str(e)}")

    def visualizePredictions(self, df, numSamples=16, figsize=(12, 12)):
        if len(df) == 0: return
        
        try:
            sampleDf = df.sample(numSamples) if len(df) > numSamples else df
            numSamples = len(sampleDf)
            predictedLabels = self.predict(sampleDf["image"])
            
            cols = min(4, numSamples)
            rows = math.ceil(numSamples / cols)
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = np.array([axes])
            elif cols == 1:
                axes = np.array([[ax] for ax in axes])
            for i in range(numSamples):
                row, col = i // cols, i % cols
                ax = axes[row, col]
            
                img = sampleDf["image"].iloc[i]
                imgArr = img.image if isinstance(img, DataFrameImage) else img
            
                if imgArr.ndim == 2:
                    ax.imshow(imgArr, cmap='gray')
                else:
                    ax.imshow(imgArr)
            
                trueLabel = sampleDf["label"].iloc[i]
                predLabel = predictedLabels[i]
            
                color = 'green' if trueLabel == predLabel else 'red'
            
                ax.set_title(f"True: {trueLabel}\nPred: {predLabel}", color=color)
                ax.axis('off')
            
            for i in range(numSamples, rows * cols):
                row, col = i // cols, i % cols
                axes[row, col].axis('off')
            
            plt.suptitle(f"{self.modelType}-{self.variant} Model Predictions", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            
            correct = sum(1 for i in range(len(sampleDf)) if sampleDf["label"].iloc[i] == predictedLabels[i])
            accuracy = 100 * correct / len(sampleDf)
            print(f"Sample accuracy: {accuracy:.2f}% ({correct}/{len(sampleDf)})")
        except Exception as e:
            print(f"Error visualizing predictions: {str(e)}")

################################MODEL SAVE/LOAD ################################
    def saveModel(self, epoch, loss, modelName=None, best=True):
        modelName = modelName or self.modelType
        
        if best:
            self.modelManager.save_best(epoch, loss, modelName)
        else:
            self.modelManager.save(epoch, loss, modelName)

    def loadModel(self, modelName=None, filename=None):
        modelName = modelName or self.modelType
        return self.modelManager.load(best_only=(filename is None), model_name=modelName, filename=filename)

    def setOptimizer(self, optimizerName="Adam", learningRate=0.001):
        self.optimizerName = optimizerName.lower()
        self.learningRate = learningRate
        
        optimizerCreators = {
            'adam': lambda: optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay),
            'sgd': lambda: optim.SGD(self.model.parameters(), lr=self.learningRate, momentum=0.9, nesterov=True, weight_decay=self.weightDecay),
            'adamw': lambda: optim.AdamW(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        }
        
        if self.optimizerName not in optimizerCreators:
            print(f"Warning: Unsupported optimizer '{self.optimizerName}', defaulting to Adam")
            self.optimizerName = 'adam'
        
        self.optimizer = optimizerCreators[self.optimizerName]()
        print(f"Updated optimizer to {self.optimizerName} with learning rate {self.learningRate}")

################################ CONFIG TESTS & COMPLETE TEST ################################
    def runConfigTests(self, trainDf, testDf, optimizers=None, learningRates=None, epochs=3, batchSize=8, maxConfigs=9, saveInterval=1, loadModel=False, saveModel=True, numSamplesVisualize=8):
        print("\nRunning configuration tests...")
        
        optimizers = optimizers or ['adam', 'sgd', 'adamw']
        learningRates = learningRates or [0.001, 0.01, 0.1]
        configs = [{'optimizer': opt, 'lr': lr, 'wd': self.weightDecay, 'dropout': self.dropoutRate} for opt in optimizers for lr in learningRates]
        
        if maxConfigs < len(configs):
            print(f"Limiting to {maxConfigs} configurations out of {len(configs)} possible combinations")
            configs = configs[:maxConfigs]
        
        print(f"\nTesting {len(configs)} configurations with fixed dropout rate ({self.dropoutRate}) and weight decay ({self.weightDecay}):")
        
        for i, cfg in enumerate(configs):
            print(f"  Config {i+1}: {cfg['optimizer']}, lr={cfg['lr']}")
        
        allResults = []
        bestModelPath = os.path.join(self.modelDir, f"{self.modelType}_best.pth")
        bestModelExists = os.path.isfile(bestModelPath)
        resumeForAll = bestModelExists
        
        for i, config in enumerate(configs):
            print(f"\n{'='*80}")
            print(f"Testing configuration {i+1}/{len(configs)}: {config}")
            print(f"{'='*80}")
        
            self.resetModel(
                numClasses=len(self.classToIdx) if self.classToIdx else len(set(trainDf["label"].unique())),
                optimizerName=config['optimizer'],
                learningRate=config['lr'],
                weightDecay=config['wd'],
                dropoutRate=config['dropout'],
            )
        
            loadThisVariant = resumeForAll
        
            print(f"\nTraining with configuration: {config}")
        
            epochStats = self.train(
                df=trainDf,
                epochs=epochs,
                batchSize=batchSize,
                saveInterval=saveInterval,
                loadModel=loadThisVariant,
                saveModel=saveModel
            )
        
            print(f"\nEvaluating configuration on test set")
        
            testAccuracy, resultsDf = self.evaluate(testDf, batchSize=batchSize)
            result = {'config': config, 'accuracy': testAccuracy, 'epochs': len(epochStats), 'epoch_stats': epochStats, 'results_df': resultsDf}
            allResults.append(result)
        
            print(f"\nVisualizing predictions for configuration {i+1}")
            self.visualizePredictions(testDf, numSamples=numSamplesVisualize)
        
            print(f"\nTraining history for configuration {i+1}")
            self.plotTrainingHistory(epochStats)
        
        self.printResultsSummary(allResults)
        bestIdx = max(range(len(allResults)), key=lambda i: allResults[i]['accuracy']) if allResults else 0
        bestConfig = configs[bestIdx] if allResults else configs[0]
        
        return allResults, bestConfig

    def runCompleteTest(self, dataPath, modelDir, classFilter=None, numFiles=None, maxConfigs=9):
        trainValDf, testDf = self.loadDataForTesting(dataPath=dataPath, classFilter=classFilter, numFiles=numFiles)
        optimizers = ['adam', 'sgd', 'adamw']
        learningRates = [0.001, 0.01, 0.1]
        totalCombos = len(optimizers) * len(learningRates)
        actualMaxConfigs = min(maxConfigs, totalCombos)
        
        print(f"Testing {min(actualMaxConfigs, len(optimizers) * len(learningRates))} configurations")
        print(f"Using optimizers: {optimizers}")
        print(f"Using learning rates: {learningRates}")
        print(f"Fixed dropout rate: {self.dropoutRate}, fixed weight decay: {self.weightDecay}")
        
        allResults, bestConfig = self.runConfigTests(
            trainDf=trainValDf,
            testDf=testDf,
            optimizers=optimizers,
            learningRates=learningRates,
            epochs=3,
            batchSize=8,
            maxConfigs=actualMaxConfigs
        )
        
        print("\nComplete test finished successfully")
        return bestConfig

    def loadDataForTesting(self, dataPath, classFilter=None, numFiles=None, validationSplit=0.2):
        dm = DataManager()
        csvFile = os.path.join(dataPath, "Training_set.csv")
        dm.LoadTrainingData(folderName=dataPath, csvFileName="Training_set.csv", classFilter=classFilter, numFiles=numFiles)
        dm.RemoveMissingData()
        
        trainValDf, testDf = self.splitData(dm.TrainingData, validationSplit)
        
        print(f"Data loaded: {len(dm.TrainingData)} total samples")
        print(f"Split into: {len(trainValDf)} train+val samples, {len(testDf)} test samples")
        print(f"Classes: {dm.TrainingData['label'].unique()}")
        
        return trainValDf, testDf

    def printResultsSummary(self, results):
        if not results: return
        print("\n" + "="*100)
        print("SUMMARY OF ALL CONFIGURATIONS")
        print("="*100)
        print(f"{'#':^5} {'Optimizer':^10} {'LR':^10} {'Weight Decay':^12} {'Dropout':^8} {'Test Acc':^12}")
        print("-" * 100)
        
        for i, result in enumerate(results):
            cfg = result['config']
            print(f"{i:^5} {cfg['optimizer']:^10} {cfg['lr']:<10.6f} {cfg['wd']:<12.6f} {cfg['dropout']:<8.1f} {result['accuracy']:<12.2f}%")
        
        if results:
            bestIdx = max(range(len(results)), key=lambda i: results[i]['accuracy'])
            bestResult = results[bestIdx]
        
            print("\n" + "="*100)
            print(f"BEST CONFIGURATION: #{bestIdx} - {bestResult['config']}")
            print(f"Test Accuracy: {bestResult['accuracy']:.2f}%")
            print("="*100)

if __name__ == "__main__":
    print("Testing NNModel implementation...")
    
    currentFolder = os.getcwd()
    dataPath = os.path.join(currentFolder, "../data/")
    modelDir = os.path.join(currentFolder, "../models")
    os.makedirs(modelDir, exist_ok=True)

    
    classes = ["sitting", "running", "drinking", "eating"]
    numClasses = 4
    
    nnModel = NNModel(
        numClasses=numClasses,
        modelType="EfficientNet",
        variant='b0',
        modelDir=modelDir,
    )
    
    maxConfigs = 2
    bestConfig = nnModel.runCompleteTest(
        dataPath=dataPath,
        modelDir=modelDir,
        classFilter=classes,
        numFiles=100,
        maxConfigs=maxConfigs
    )
    
    print(f"\nBest configuration found: {bestConfig}")
    print("\nTesting completed.")