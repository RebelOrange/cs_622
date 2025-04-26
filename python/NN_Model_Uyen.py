import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os
from Model import Model
from DataManager import DataFrameImage, DataManager
from ModelManager import ModelManager

################################ INIT & SETUP ################################
class NNModel(Model):
    def __init__(self, numClasses, modelType="EfficientNet", variant='b0', modelDir="models",
                    optimizerName='Adam', learningRate=0.001, weightDecay=0.0, lossFunction='cross_entropy',
                    dropoutRate=0.0, useRegularization=False):
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

################################ DATA PREP ################################
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

    def preprocImg(self, image, isTraining=True):
        imgArr = image.image if isinstance(image, DataFrameImage) else image
        
        if imgArr.ndim == 2: imgArr = np.stack([imgArr] * 3, axis=2)
        transform = self.transform if isTraining else self.evalTransform
        
        return transform(imgArr)

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

################################ TRAINING ################################
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
        accuracy = 100 * correct / total if total > 0 else 0
        
        return accuracy, {'true_label': trues, 'predicted_label': preds, 'confidence': confidences, 'correct': [p == l for p, l in zip(preds, trues)]}

################################ MODEL SAVE/LOAD & RESET ################################
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

    def saveModel(self, epoch, loss, modelName=None, best=True):
        modelName = modelName or self.modelType
        if best:
            self.modelManager.save_best(epoch, loss, modelName)
        else:
            self.modelManager.save(epoch, loss, modelName)

    def loadModel(self, modelName=None, filename=None):
        modelName = modelName or self.modelType
        return self.modelManager.load(best_only=(filename is None), model_name=modelName, filename=filename)
