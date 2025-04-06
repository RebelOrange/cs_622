import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import random
from Model import Model
from DataManager import DataFrameImage, DataManager
from ModelManager import ModelManager
from torch.optim import RAdam

"""
Reference: 
- https://pytorch.org/vision/stable/models/efficientnet.html
- https://pytorch.org/hub/nvidia_deeplearningexamples_efficientnet/
- https://github.com/lukemelas/EfficientNet-PyTorch
- https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
- https://www.geeksforgeeks.org/ml-introduction-to-transfer-learning/
- https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- https://www.freecodecamp.org/news/deep-learning-with-pytorch/
- https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- https://pytorch.org/docs/stable/index.html
- https://pytorch.org/docs/stable/optim.html
"""

"""TODO DONE 
- Add more optimizers (custom optimizers) -> done
- Add more loss functions (custom loss functions) -> done
- Regularization techniques (L1, L2, Dropout)? -> done
- Softmax for multi-class classification -> done
- Add more data augmentation techniques -> done

"""

"""TODO NOT DONE
- Custom Learning Rate Scheduler -> not done ? should do? or too muh already
- Remove the pre-trained weight so that it is more from scratch -> too complex, should i try? 
- Add better comments and documentation ****
- Clean up and organize + optimize the code ****
- Chnage into betteer splitting and training functions ****
"""

######################## EFF NET IMPLEMENTATION ########################
class EfficientNet(Model):
    def __init__(self, num_classes, variant='b0', model_dir="models",
                optimizer_name='adam', learning_rate=0.001, l2_weight_decay=0.0001,
                loss_function='cross_entropy', dropout_rate=0.2, use_l1_reg=False, 
                l1_strength=0.0001, use_l2_reg=True):
        super().__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.class_to_idx = None
        self.idx_to_class = None
        
        self.model_dir = model_dir
        self.dropout_rate = dropout_rate
        self.use_l1_reg = use_l1_reg
        self.l1_strength = l1_strength
        self.use_l2_reg = use_l2_reg
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.l2_weight_decay = l2_weight_decay if use_l2_reg else 0.0
        self.loss_function_name = loss_function
        
        self.loadInitModel(variant, num_classes)
        self.setupTransform()
        self.createLossFunction(loss_function)
        self.createOptimizer(optimizer_name, learning_rate)

        self.model_manager = ModelManager(
            model=self.model, optimizer=self.optimizer,
            device=self.device, model_dir=self.model_dir
        )
        
        print("\n" + "="*50)
        self.printModelInfo()

    ######################## MODEL INITIALIZATION ########################
    def loadInitModel(self, variant, num_classes):
        print(f"\nLoading EfficientNet-{variant} model...")
    
        supported_variants = ['b0', 'b1', 'b4', 'b5', 'b6', 'b7']
        
        if variant not in supported_variants:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}. Supported variants: {supported_variants}")
        
        if variant == 'b0':
            self.model = models.efficientnet_b0(weights='DEFAULT')
        elif variant == 'b1':
            self.model = models.efficientnet_b1(weights='DEFAULT')
        elif variant == 'b4':
            self.model = models.efficientnet_b4(weights='DEFAULT')
        elif variant == 'b5':
            self.model = models.efficientnet_b5(weights='DEFAULT')
        elif variant == 'b6':
            self.model = models.efficientnet_b6(weights='DEFAULT')
        elif variant == 'b7':
            self.model = models.efficientnet_b7(weights='DEFAULT')
        
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        self.model = self.model.to(self.device)

    def setupTransform(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),     
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),                                         
            transforms.ToTensor(),                           
            transforms.Normalize(                            
                mean=[0.485, 0.456, 0.406],                  
                std=[0.229, 0.224, 0.225]                    
            )
        ])

    ######################## LOSS FUNCTIONS ########################
    def focal_loss(self, outputs, targets, gamma=2.0):
        ce = nn.functional.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** gamma * ce
        return loss.mean()

    def f1_loss(self, outputs, targets, epsilon=1e-7):
        outputs = nn.functional.softmax(outputs, dim=1)
        targets_one_hot = torch.zeros_like(outputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        tp = torch.sum(outputs * targets_one_hot, dim=0)
        fp = torch.sum(outputs * (1 - targets_one_hot), dim=0)
        fn = torch.sum((1 - outputs) * targets_one_hot, dim=0)
        
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        return 1 - f1.mean()

    def dice_loss(self, outputs, targets, epsilon=1e-6):
        outputs = nn.functional.softmax(outputs, dim=1)
        
        targets_one_hot = torch.zeros_like(outputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        intersection = torch.sum(outputs * targets_one_hot, dim=0)
        cardinality_pred = torch.sum(outputs, dim=0)
        cardinality_true = torch.sum(targets_one_hot, dim=0)
        
        dice = (2 * intersection + epsilon) / (cardinality_pred + cardinality_true + epsilon)
        return 1 - dice.mean()

    def createLossFunction(self, loss_function):
        loss_functions = {
            'cross_entropy': nn.CrossEntropyLoss(),
            'label_smoothing': nn.CrossEntropyLoss(label_smoothing=0.1),
            'huber': nn.HuberLoss(),
        }
        
        if loss_function == 'focal':
            self.criterion = self.focal_loss
        elif loss_function == 'f1_loss':
            self.criterion = self.f1_loss
        elif loss_function == 'dice':
            self.criterion = self.dice_loss
        else:
            self.criterion = loss_functions.get(loss_function, nn.CrossEntropyLoss())
            if loss_function not in loss_functions:
                print(f"Warning: Unknown loss function '{loss_function}', defaulting to CrossEntropyLoss")

    ######################## OPTIMIZER ########################
    def createOptimizer(self, optimizer_name, learning_rate):
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=learning_rate,
                weight_decay=self.l2_weight_decay
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=learning_rate,
                momentum=0.9, 
                nesterov=True,
                weight_decay=self.l2_weight_decay
            )
        elif optimizer_name == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(), 
                lr=learning_rate,
                momentum=0.9,
                weight_decay=self.l2_weight_decay
            )
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate,
                weight_decay=self.l2_weight_decay
            )
        elif optimizer_name == 'radam':
            try:
                self.optimizer = RAdam(
                    self.model.parameters(),
                    lr=learning_rate,
                    weight_decay=self.l2_weight_decay
                )
            except ImportError:
                print("Warning: RAdam not available, use Adam instead")
                self.optimizer = optim.Adam(
                    self.model.parameters(), 
                    lr=learning_rate,
                    weight_decay=self.l2_weight_decay
                )
        else:
            print(f"Warning: Unknown optimizer {optimizer_name}, defaulting to Adam")
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=learning_rate,
                weight_decay=self.l2_weight_decay
            )

    ######################## DATA PROCESSING ########################
    def preprocessImages(self, image):
        if isinstance(image, DataFrameImage):
            image_array = image.image
        else:
            image_array = image
            
        # Convert grayscale to RGB if needed as EfficientNet expects RGB
        if image_array.ndim == 2:
            image_array = np.stack([image_array, image_array, image_array], axis=2)
            
        return self.transform(image_array)
    
    def setupClassMapping(self, df):
        unique_classes = sorted(df["label"].unique())
        
        self.class_to_idx = {}  
        self.idx_to_class = {} 
        
        for idx, class_name in enumerate(unique_classes):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
        
        print(f"\nSet up mapping for {len(unique_classes)} classes: {unique_classes}")

    def prepareBatch(self, dataframe, batch_indices):
        batch_data = dataframe.iloc[batch_indices]
        
        image_tensors = [self.preprocessImages(img) for img in batch_data["image"]]
        image_batch = torch.stack(image_tensors).to(self.device)
        
        if self.class_to_idx is None:
            raise ValueError("Class mapping not initialized. Do setupClassMapping() before prepare batches.")
        
        label_indices = [self.class_to_idx[label] for label in batch_data["label"]]
        label_tensor = torch.tensor(label_indices).to(self.device)
        
        return image_batch, label_tensor

    def preprocess(self, df):
        if df is None:
            raise ValueError("Error: Dataframe is None")
            
        if len(df) == 0:
            raise ValueError("Error: Empty dataframe provided")

        try:
            sample_img = df["image"].iloc[0]
            if isinstance(sample_img, DataFrameImage):
                img_array = sample_img.image
            else:
                img_array = sample_img
                
            if img_array.ndim < 2:
                raise ValueError("Error: Images must be 2D or 3D arrays")
                
        except Exception as e:
            raise ValueError(f"Error examining images: {str(e)}")
        
        return df
    
    ######################## TRAINING FUNCTIONS ########################
    def train(self, df, epochs=10, batch_size=32, save_interval=1, save_best=True, load_best=True):
        self.setupTraining(df)
        n_samples = len(df)
        indices = np.arange(n_samples)
        num_batches = (n_samples + batch_size - 1) // batch_size
        
        start_epoch = 0
        best_loss = float('inf')
        if load_best:
            start_epoch, best_loss = self.loadModel("EfficientNet")
            if start_epoch > 0:
                print(f"\nResuming training from epoch {start_epoch+1} with best loss: {best_loss:.4f}")
        else:
            print("\nStarting training from beginning")

        print(f"\nTraining plan: {epochs} epochs, {n_samples} samples, {num_batches} batches per epoch")
        
        for epoch in range(start_epoch, start_epoch + epochs):
            print(f"\nEpoch {epoch+1}/{start_epoch + epochs}")
            print("Progress: [", end="")
            
            epoch_start_time = time.time()
            epoch_loss, epoch_acc = self.trainEpoch(df, indices, batch_size, num_batches, track_metrics=True)
            print("]")

            epoch_time = time.time() - epoch_start_time
            self.displayEpoch(epoch, start_epoch + epochs, epoch_time, epoch_loss, epoch_acc)
            
            self.saveCheckpoint(epoch, epoch_loss, best_loss, save_interval, save_best)
            if epoch_loss < best_loss:
                best_loss = epoch_loss

    def setupTraining(self, df):
        if self.class_to_idx is None:
            self.setupClassMapping(df)
        self.model.train()

    def trainEpoch(self, df, indices, batch_size, num_batches, track_metrics=True):
        n_samples = len(indices)
        running_loss = 0.0
        correct = 0
        total = 0
        
        self.model.train()
        
        np.random.shuffle(indices)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            inputs, label_indices = self.prepareBatch(df, batch_indices)
            
            if track_metrics:
                batch_loss, batch_correct, batch_total = self.processBatch(inputs, label_indices, return_metrics=True)
                
                running_loss += batch_loss
                correct += batch_correct
                total += batch_total

                current_batch = (i // batch_size) + 1
                self.displayProcess(current_batch, num_batches)
            else:
                self.processBatch(inputs, label_indices, return_metrics=False)
        
        if track_metrics:
            epoch_loss = running_loss / num_batches
            epoch_acc = 100 * correct / total if total > 0 else 0
            return epoch_loss, epoch_acc
        else:
            return None, None

    def processBatch(self, inputs, label_indices, return_metrics=True):
        self.optimizer.zero_grad()
        
        outputs = self.model(inputs)
        
        loss = self.criterion(outputs, label_indices)
        
        if self.use_l1_reg:
            l1_norm = sum(p.abs().sum() for p in self.model.parameters())
            loss += self.l1_strength * l1_norm
        
        loss.backward()
        self.optimizer.step()
        
        if return_metrics:
            batch_loss = loss.item()
            _, predicted = torch.max(outputs.data, 1)
            batch_total = label_indices.size(0)
            batch_correct = (predicted == label_indices).sum().item()
            return batch_loss, batch_correct, batch_total
        
        return None

    ######################## UTILITY FUNCTIONS ########################
    def printModelInfo(self):
        regularization_methods = []
        if self.dropout_rate > 0:
            regularization_methods.append(f"Dropout ({self.dropout_rate})")
        if self.use_l1_reg:
            regularization_methods.append(f"L1 reg (strength={self.l1_strength})")
        if self.use_l2_reg:
            regularization_methods.append(f"L2 reg (strength={self.l2_weight_decay})")
        
        reg_summary = ", ".join(regularization_methods) if regularization_methods else "None"
        
        print(f"\nModel initialized with: optimizer={self.optimizer_name}, lr={self.learning_rate}, "
            f"loss={self.loss_function_name}, regularization=[{reg_summary}]")

    def displayProcess(self, current_batch, num_batches):
        display_interval = max(1, num_batches // 20)
        if current_batch % display_interval == 0 or current_batch == num_batches:
            progress = int(30 * current_batch / num_batches)
            num_symbols = progress - len(str(current_batch)) - len(str(num_batches)) - 3
            print("=" * num_symbols, end="")
            print(f" {current_batch}/{num_batches} ", end="", flush=True)

    def displayEpoch(self, epoch, total_epochs, epoch_time, epoch_loss, epoch_acc):
        print(f"\nEpoch {epoch+1}/{total_epochs} completed in {epoch_time:.1f}s - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
        remaining_epochs = total_epochs - (epoch + 1)
        if remaining_epochs > 0:
            estimated_time = epoch_time * remaining_epochs
            hours, remainder = divmod(estimated_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"\nEstimated remaining time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    def saveCheckpoint(self, epoch, epoch_loss, best_loss, save_interval, save_best):
        if epoch_loss < best_loss:
            if save_best:
                self.saveModel(epoch+1, epoch_loss, "EfficientNet", best=True)
                print(f"\nNew best model saved with loss: {epoch_loss:.4f}")
            else:
                print(f"\nNew best loss: {epoch_loss:.4f} (model not saved)")
        elif (epoch + 1) % save_interval == 0:
            self.saveModel(epoch+1, epoch_loss, "EfficientNet", best=False)
            print(f"\nCheckpoint saved at epoch {epoch+1}")

    ######################## PREDICTION FUNCTIONS ########################
    def predict(self, image=None):
        if image is None:
            return None
            
        self.model.eval()
        
        if isinstance(image, pd.Series):
            return self.predictBatch(image)
        else:
            return self.predictSingle(image)

    def predictSingle(self, image):
        with torch.no_grad():
            img_tensor = self.preprocessImages(image)  
            img_tensor = img_tensor.unsqueeze(0) 
            img_tensor = img_tensor.to(self.device)  
            
            outputs = self.model(img_tensor)
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            _, predicted_idx = torch.max(outputs, 1)
            confidence = probabilities[0][predicted_idx.item()].item()
            
            predicted_label = self.idx_to_class[predicted_idx.item()]

            print(f"\nModel Predicted: {predicted_label} (confidence: {confidence:.2f})")
            return predicted_label

    def predictBatch(self, images):
        predictions = []
        
        with torch.no_grad():
            for img in images:
                img_tensor = self.preprocessImages(img)  
                img_tensor = img_tensor.unsqueeze(0) 
                img_tensor = img_tensor.to(self.device) 
                
                outputs = self.model(img_tensor)
                
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                _, predicted_idx = torch.max(outputs, 1)
                
                predicted_label = self.idx_to_class[predicted_idx.item()]
                predictions.append(predicted_label)
                
        return predictions
    
    ######################## MODEL SAVING & LOADING ########################
    def saveModel(self, epoch, loss, model_name="EfficientNet", best=True):
        if best:
            self.model_manager.save_best(epoch, loss, model_name)
        else:
            self.model_manager.save(epoch, loss, model_name)

    def loadModel(self, model_name="EfficientNet"):
        return self.model_manager.load(best_only=True, model_name=model_name)
    
    ######################## EVALUATION & TUNING ########################
    def evaluate(self, df, batch_size=32):
        self.model.eval()
        n_samples = len(df)
        indices = np.arange(n_samples)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                inputs, label_indices = self.prepareBatch(df, batch_indices)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += label_indices.size(0)
                correct += (predicted == label_indices).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"\nEvaluation completed: {correct}/{total} correct ({accuracy:.2f}%)")
        
        return accuracy

    def findBestConfig(self, df, validation_split=0.2, epochs=5, batch_size=32, max_configs=10):
        print("\nStarting tuning best configuration...")
        
        train_df, val_df = self.splitData(df, validation_split)
        print(f"\nSplit data: {len(train_df)} training samples, {len(val_df)} validation samples")
        
        configs = self.generateConfigs(max_configs)
        
        if self.class_to_idx is None:
            self.setupClassMapping(df)
        
        original_state = self.model.state_dict().copy()
        
        results = []
        best_val_acc = 0
        best_config = None
        best_model_state = None
        
        self.printConfigHeader()
        
        for i, config in enumerate(configs):
            self.resetModel(
                num_classes=len(self.class_to_idx),
                optimizer_name=config['optimizer'],
                learning_rate=config['lr'],
                l2_weight_decay=config['wd'],
                loss_function=config['loss'],
                dropout_rate=config['dropout'],
                use_l1_reg=config['l1']
            )

            self.model.train()
            for epoch in range(epochs):
                self.trainEpoch(df, np.arange(len(df)), batch_size, (len(df) + batch_size - 1) // batch_size, track_metrics=False)

            val_acc = self.evaluate(val_df, batch_size)
            
            results.append({'config': config, 'val_acc': val_acc})
            
            self.printConfigResult(i, config, val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_config = config
                best_model_state = self.model.state_dict().copy()
        
        self.printConfigSummary(results, best_config, best_val_acc)

        self.resetModel(
            num_classes=len(self.class_to_idx),
            optimizer_name=best_config['optimizer'],
            learning_rate=best_config['lr'],
            l2_weight_decay=best_config['wd'],
            loss_function=best_config['loss'],
            dropout_rate=best_config['dropout'],
            use_l1_reg=best_config['l1']
        )
        self.model.load_state_dict(best_model_state)
        
        return best_config
    
    def splitData(self, df, validation_split):
        n_samples = len(df)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        val_size = int(validation_split * n_samples)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        
        return train_df, val_df
    
    def generateConfigs(self, max_configs):
        optimizer_options = ['adam', 'sgd', 'adamw', 'rmsprop', 'radam']
        lr_options = [0.01, 0.001, 0.0001]
        l2_weight_decay_options = [0.0, 0.0001, 0.001, 0.01]
        loss_function_options = ['cross_entropy', 'focal', 'label_smoothing', 'f1_loss', 'dice']
        dropout_options = [0.0, 0.2, 0.5]
        l1_options = [False, True]
        
        configs = []
        for opt in optimizer_options:
            for lr in lr_options:
                for wd in l2_weight_decay_options:
                    for loss in loss_function_options:
                        for dropout in dropout_options:
                            for l1 in l1_options:
                                configs.append({
                                    'optimizer': opt,
                                    'lr': lr,
                                    'wd': wd,
                                    'loss': loss,
                                    'dropout': dropout,
                                    'l1': l1
                                })
        
        total_configs = len(configs)
        print(f"\nGenerated {total_configs} possible configurations")
        
        if max_configs is not None and max_configs < total_configs:
            configs = self.sampleConfigs(configs, max_configs)
            
        return configs
    
    def sampleConfigs(self, configs, max_configs):
        baseline_configs = [
            {'optimizer': 'adam', 'lr': 0.001, 'wd': 0.0001, 'loss': 'cross_entropy', 'dropout': 0.2, 'l1': False},
            {'optimizer': 'sgd', 'lr': 0.01, 'wd': 0.0001, 'loss': 'cross_entropy', 'dropout': 0.2, 'l1': False},
            {'optimizer': 'rmsprop', 'lr': 0.001, 'wd': 0.0, 'loss': 'cross_entropy', 'dropout': 0.2, 'l1': False},
            {'optimizer': 'adamw', 'lr': 0.001, 'wd': 0.001, 'loss': 'cross_entropy', 'dropout': 0.2, 'l1': False},
            {'optimizer': 'radam', 'lr': 0.001, 'wd': 0.0001, 'loss': 'cross_entropy', 'dropout': 0.2, 'l1': False}
        ]
        
        remaining_configs = [c for c in configs if c not in baseline_configs]
        
        random_configs = random.sample(remaining_configs, min(max_configs - len(baseline_configs), len(remaining_configs)))
        limited_configs = baseline_configs + random_configs
        
        print(f"\nLimited to {len(limited_configs)} configurations "
            f"({len(baseline_configs)} baselines + {len(random_configs)} random)")
        
        return limited_configs
    
    ######################## PRINTING FUNCTIONS FOR TUNING ########################
    def printConfigHeader(self):
        print("\nTesting configurations:")
        print("-" * 80)
        print(f"{'Config':^8} {'Optimizer':^10} {'LR':^8} {'Loss':^15} {'Dropout':^8} {'L1':^6} {'L2 (wd)':^8} {'Val Acc':^10}")
        print("-" * 80)
    
    def printConfigResult(self, i, config, val_acc):
        print(f"{i:^8} {config['optimizer']:^10} {config['lr']:^8.4f} {config['loss']:^15} "
            f"{config['dropout']:^8.1f} {str(config['l1']):^6} {config['wd']:^8.5f} {val_acc:^10.2f}%")
    
    def printConfigSummary(self, results, best_config, best_val_acc):
        results.sort(key=lambda x: x['val_acc'], reverse=True)
        
        print("-" * 80)
        print(f"\nBest configuration: {best_config} with validation accuracy: {best_val_acc:.2f}%")
        
        print("\nTop 5 configurations:")
        for i, result in enumerate(results[:5]):
            print(f"{i+1}. {result['config']} - Acc: {result['val_acc']:.2f}%")
    
    def resetModel(self, num_classes, optimizer_name, learning_rate, l2_weight_decay, 
                loss_function, dropout_rate, use_l1_reg):
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.l2_weight_decay = l2_weight_decay
        self.loss_function_name = loss_function
        self.dropout_rate = dropout_rate
        self.use_l1_reg = use_l1_reg
        self.l1_strength = 0.0001
        
        num_features = self.model.classifier[1][1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        self.model = self.model.to(self.device)
        
        self.createLossFunction(loss_function)
        self.createOptimizer(optimizer_name, learning_rate)

##################################### Testing Code ############################################
if __name__ == "__main__":
    print("Test 1: Loading training data")
    dm = DataManager()
    current_folder = os.getcwd()
    dm.LoadTrainingData(folderName=current_folder+"/../data/", csvFileName="Training_set.csv", numFiles=100)
    dm.RemoveMissingData()
    
    print("Test 2: Creating model")
    num_classes = len(dm.TrainingData["label"].unique())
    model_dir = os.path.join(current_folder, "../models")
    os.makedirs(model_dir, exist_ok=True)
    
    print("Test 3: Finding optimal configuration")
    model = EfficientNet(num_classes=num_classes, variant='b0', model_dir=model_dir)
    best_config = model.findBestConfig(dm.TrainingData, validation_split=0.2, epochs=2, batch_size=8, max_configs=5)
    
    print("Test 4: Training model with best configuration")
    model = EfficientNet(
        num_classes=num_classes,
        variant='b0',
        model_dir=model_dir,
        optimizer_name=best_config['optimizer'],
        learning_rate=best_config['lr'],
        l2_weight_decay=best_config['wd'],
        loss_function=best_config['loss'],
        dropout_rate=best_config['dropout'],
        use_l1_reg=best_config['l1']
    )
    model.train(dm.TrainingData, epochs=2, batch_size=8, save_interval=2, save_best=False, load_best=False)
    
    print("\nTest 5: Testing model predictions...")
    test_batch = dm.TrainingData.sample(10)
    predicted_labels = model.predict(test_batch["image"])
    # todo: make it in grid isnead of one by one
    for i in range(len(test_batch)):
        plt.figure(figsize=(6, 6))
        img_obj = test_batch["image"].iloc[i]
        img_array = img_obj.image
        
        plt.imshow(img_array)
        true_label = test_batch["label"].iloc[i]
        pred_label = predicted_labels[i]
        plt.title(f"True: {true_label} | Predicted: {pred_label}")
        plt.axis('off')
        plt.show()

    correct = sum(1 for a, p in zip(test_batch["label"], predicted_labels) if a == p)
    accuracy = 100 * correct / len(test_batch["label"])
    print(f"Batch test accuracy: {accuracy:.2f}%")