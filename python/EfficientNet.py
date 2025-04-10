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
import random
import math
from Model import Model
from DataManager import DataFrameImage, DataManager
from ModelManager import ModelManager

######################## EFF NET IMPLEMENTATION ########################
class EfficientNet(Model):
    def __init__(self, num_classes, variant='b0', model_dir="models",
                optimizer_name='adam', learning_rate=0.001, l2_weight_decay=None, 
                loss_function='cross_entropy', dropout_rate=None, use_l2_reg=None):
        super().__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.class_to_idx, self.idx_to_class = None, None
        self.model_dir = model_dir
        self.dropout_rate = dropout_rate or 0.0
        self.use_l2_reg = use_l2_reg or False
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.l2_weight_decay = l2_weight_decay if (self.use_l2_reg and l2_weight_decay) else 0.0
        self.loss_function_name = loss_function
        
        self.loadInitModel(variant, num_classes)
        self.setupTransform()
        self.createLossFunction(loss_function)
        self.createOptimizer(optimizer_name, learning_rate)
        self.model_manager = ModelManager(model=self.model, optimizer=self.optimizer, device=self.device, model_dir=self.model_dir)
        
        print("\n" + "="*50)
        self.printModelInfo()

    ######################## MODEL INITIALIZATION ########################
    def loadInitModel(self, variant, num_classes):
        print(f"\nLoading EfficientNet-{variant} model...")
        supported_variants = ['b0', 'b1', 'b4', 'b5', 'b6', 'b7']
        
        if variant not in supported_variants:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}. Supported variants: {supported_variants}")
        
        model_builders = {
            'b0': models.efficientnet_b0,
            'b1': models.efficientnet_b1,
            'b4': models.efficientnet_b4,
            'b5': models.efficientnet_b5,
            'b6': models.efficientnet_b6,
            'b7': models.efficientnet_b7
        }
        
        self.model = model_builders[variant](weights='DEFAULT')
        num_features = self.model.classifier[1].in_features
        
        self.model.classifier[1] = (
            nn.Sequential(nn.Dropout(self.dropout_rate), nn.Linear(num_features, num_classes)) 
            if self.dropout_rate > 0 else 
            nn.Linear(num_features, num_classes)
        )
        self.model = self.model.to(self.device)

    def setupTransform(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # eff net need size to be over 224x224
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),     
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),                                         
            transforms.ToTensor(),                           
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                    
        ])

    ######################## LOSS FUNCTIONS ########################
    def createLossFunction(self, loss_function):
        loss_functions = {
            'cross_entropy': nn.CrossEntropyLoss(),
            'label_smoothing': nn.CrossEntropyLoss(label_smoothing=0.1),
        }

        if loss_function in loss_functions:
            self.criterion = loss_functions[loss_function].to(self.device)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}. Supported functions: {list(loss_functions.keys())}")

    ######################## OPTIMIZER ########################
    def createOptimizer(self, optimizer_name, learning_rate):
        optimizers = {
            'adam': lambda: optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=self.l2_weight_decay),
            'sgd': lambda: optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=self.l2_weight_decay),
            'adamw': lambda: optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=self.l2_weight_decay)
        }
        
        if optimizer_name == 'radam':
            try:
                self.optimizer = optim.RAdam(self.model.parameters(), lr=learning_rate, weight_decay=self.l2_weight_decay)
                return
            except ImportError:
                print("Warning: RAdam not available, use Adam instead")
                optimizer_name = 'adam'
        
        self.optimizer = optimizers.get(optimizer_name, optimizers['adam'])()

    ######################## DATA PROCESSING ########################
    def preprocessImages(self, image):
        image_array = image.image if isinstance(image, DataFrameImage) else image
        
        # Convert grayscale to RGB if needed
        # dont remove this as eff net use RGB images
        if image_array.ndim == 2:
            image_array = np.stack([image_array] * 3, axis=2)
            
        return self.transform(image_array)
    
    # dont remove this as it will break the load model and error the model
    def setupClassMapping(self, df):
        if "label" not in df.columns:
            if self.class_to_idx is not None:
                print("\nUsing existing class mapping as input df has no labels")
                return
            raise ValueError("No class mapping exists and input dataframe has no labels")
        
        unique_classes = sorted(df["label"].unique())
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}
        self.idx_to_class = {idx: class_name for idx, class_name in enumerate(unique_classes)}
        
        print(f"\nSet up mapping for {len(unique_classes)} classes: {unique_classes}")

    def prepareBatch(self, dataframe, batch_indices):
        # keep this error message as it help to debug load model !!! do not remove
        if self.class_to_idx is None:
            raise ValueError("Class mapping not initialized. Do setupClassMapping() before prepare batches.")
            
        batch_data = dataframe.iloc[batch_indices]
        image_tensors = [self.preprocessImages(img) for img in batch_data["image"]]
        image_batch = torch.stack(image_tensors).to(self.device)
        
        label_indices = [self.class_to_idx[label] for label in batch_data["label"]]
        label_tensor = torch.tensor(label_indices).to(self.device)
        
        return image_batch, label_tensor

    def preprocess(self, df):
        if df is None or len(df) == 0:
            raise ValueError("Error: Dataframe is None or empty")

        try:
            sample_img = df["image"].iloc[0]
            img_array = sample_img.image if isinstance(sample_img, DataFrameImage) else sample_img
            
            if img_array.ndim < 2:
                raise ValueError("Error: Images must be 2D or 3D arrays")
                
        except Exception as e:
            raise ValueError(f"Error examining images: {str(e)}")
        
        return df
    
    ######################## TRAINING FUNCTIONS ########################
    def train(self, df, epochs=10, batch_size=32, save_interval=5, save_best=True, load_best=True):
        self.setupTraining(df)
        n_samples = len(df)
        indices = np.arange(n_samples)
        num_batches = (n_samples + batch_size - 1) // batch_size
        
        start_epoch, best_loss = 0, float('inf')
        if load_best:
            start_epoch, best_loss = self.loadModel("EfficientNet")
            print(f"\n{'Resuming training from epoch ' + str(start_epoch+1) + ' with best loss: ' + f'{best_loss:.4f}' if start_epoch > 0 else 'Starting training from beginning'}")
        
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
        running_loss, correct, total = 0.0, 0, 0
        
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
                self.displayProcess((i // batch_size) + 1, num_batches)
            else:
                self.processBatch(inputs, label_indices, return_metrics=False)
        
        if track_metrics:
            epoch_loss = running_loss / num_batches
            epoch_acc = 100 * correct / total if total > 0 else 0
            return epoch_loss, epoch_acc
        return None, None

    def processBatch(self, inputs, label_indices, return_metrics=True):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, label_indices)
        
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

    # do not remove this fucntion as it is safeguard in case trainning fail
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
        return self.predictBatch(image) if isinstance(image, pd.Series) else self.predictSingle(image)

    def predictSingle(self, image):
        with torch.no_grad():
            img_tensor = self.preprocessImages(image).unsqueeze(0).to(self.device)
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
                img_tensor = self.preprocessImages(img).unsqueeze(0).to(self.device)
                outputs = self.model(img_tensor)
                _, predicted_idx = torch.max(outputs, 1)
                predictions.append(self.idx_to_class[predicted_idx.item()])
                
        return predictions
    
    ######################## MODEL SAVING & LOADING ########################
    # do not remove these 2
    def saveModel(self, epoch, loss, model_name="EfficientNet", best=True):
        if best:
            self.model_manager.save_best(epoch, loss, model_name)
        else:
            self.model_manager.save(epoch, loss, model_name)

    def loadModel(self, model_name="EfficientNet", filename=None):
        return self.model_manager.load(best_only=(filename is None), model_name=model_name, filename=filename)
    
    ######################## EVALUATION & TUNING ########################
    def evaluate(self, df, batch_size=32):
        self.model.eval()
        n_samples = len(df)
        indices = np.arange(n_samples)
        correct, total = 0, 0
        
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

    def findBestConfig(self, df, validation_split=0.2, epochs=5, batch_size=32, max_configs=9, 
                    l2_values=None, dropout_values=None, loss_functions=None, validation_data=None):
        print("\nStarting tuning best configuration...")
        
        if validation_data is not None:
            train_df = df
            val_df = validation_data
            print(f"\nUsing provided validation data: {len(train_df)} training samples, {len(val_df)} validation samples")
        else:
            train_df, val_df = self.splitData(df, validation_split)
            print(f"\nSplit data: {len(train_df)} training samples, {len(val_df)} validation samples")
        
        configs = self.generateConfigs(
            max_configs=max_configs,
            l2_values=l2_values,
            dropout_values=dropout_values,
            loss_functions=loss_functions
        )
        
        if self.class_to_idx is None:
            self.setupClassMapping(df)
        
        results = []
        best_val_acc = 0
        best_config, best_model_state = None, None
        
        self.printConfigHeader()
        
        for i, config in enumerate(configs):
            self.resetModel(
                num_classes=len(self.class_to_idx),
                optimizer_name=config['optimizer'],
                learning_rate=config['lr'],
                l2_weight_decay=config['wd'],
                loss_function=config['loss'],
                dropout_rate=config['dropout'],
            )

            self.model.train()
            for epoch in range(epochs):
                self.trainEpoch(train_df, np.arange(len(train_df)), batch_size, (len(train_df) + batch_size - 1) // batch_size, track_metrics=False)

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
        )
        self.model.load_state_dict(best_model_state)
        
        return best_config, configs
    
    def splitData(self, df, validation_split):
        train_df, val_df = train_test_split(df, test_size=validation_split, random_state=42)
        
        train_df = self.preprocess(df=train_df)
        val_df = self.preprocess(df=val_df)
        
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        
        return train_df, val_df
    
    def generateConfigs(self, max_configs=9, l2_values=None, dropout_values=None, loss_functions=None):
        # Base options - can change if need but maybe dont
        optimizer_options = ['adam', 'sgd', 'adamw']
        lr_options = [0.01, 0.001, 0.0001]
        
        # optional config, dont remove this or it will err
        l2_values = [0.0] if l2_values is None else l2_values
        dropout_values = [0.0] if dropout_values is None else dropout_values
        loss_functions = ['cross_entropy'] if loss_functions is None else loss_functions
        
        total_combinations = len(optimizer_options) * len(lr_options) * len(l2_values) * len(dropout_values) * len(loss_functions)
        
        configs = []
        for i, combo in enumerate(self.configGenerator(optimizer_options, lr_options, l2_values, dropout_values, loss_functions)):
            configs.append(combo)
            if 0 < max_configs <= i+1:
                break
                
        dimensions = [f"{len(opts)} {name}s" for name, opts in [
            ('optimizer', optimizer_options), ('learning rate', lr_options),
            ('L2 weight', l2_values), ('dropout rate', dropout_values),
            ('loss function', loss_functions)
        ] if len(opts) > 1]
        
        summary = " - ".join(dimensions) if dimensions else "default grid"
        
        message = f"\nGenerated {len(configs)} configurations "
        message += f"(limited from {summary})" if 0 < max_configs < total_combinations else f"({summary})"
        print(message)
            
        return configs
    
    # helper func, to remove nested loops in gen configs -> don't remove 
    # look eww with nested loop but it work so dont touch
    def configGenerator(self, opts, lrs, wds, dropouts, losses):
        for opt in opts:
            for lr in lrs:
                for wd in wds:
                    for dropout in dropouts:
                        for loss in losses:
                            yield {
                                'optimizer': opt, 'lr': lr, 'wd': wd,
                                'loss': loss, 'dropout': dropout
                            }
    
    ######################## PRINTING FUNCTIONS FOR TUNING ########################
    def printConfigHeader(self):
        print("\nTesting configurations:")
        print("-" * 80)
        print(f"{'Config':^8} {'Optimizer':^10} {'LR':^8} {'Loss':^15} {'Dropout':^8} {'L2 (wd)':^8} {'Val Acc':^10}")
        print("-" * 80)
    
    def printConfigResult(self, i, config, val_acc):
        print(f"{i:^8} {config['optimizer']:^10} {config['lr']:^8.4f} {config['loss']:^15} {config['dropout']:^8.1f} {config['wd']:^8.5f} {val_acc:^10.2f}%")
    
    def printConfigSummary(self, results, best_config, best_val_acc):
        results.sort(key=lambda x: x['val_acc'], reverse=True)
        
        print("-" * 80)
        print(f"\nBest configuration: {best_config} with validation accuracy: {best_val_acc:.2f}%")
        
        print("\nTop 5 configurations:")
        for i, result in enumerate(results[:5]):
            print(f"{i+1}. {result['config']} - Acc: {result['val_acc']:.2f}%")
    
    def resetModel(self, num_classes, optimizer_name, learning_rate, l2_weight_decay, loss_function, dropout_rate):
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.l2_weight_decay = l2_weight_decay if l2_weight_decay is not None else 0.0
        self.loss_function_name = loss_function
        self.dropout_rate = dropout_rate if dropout_rate is not None else 0.0
        
        if hasattr(self.model.classifier[1], 'in_features'):
            num_features = self.model.classifier[1].in_features
        else:
            num_features = self.model.classifier[1][1].in_features
        
        if self.dropout_rate > 0:
            self.model.classifier[1] = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(num_features, num_classes)
            )
        else:
            self.model.classifier[1] = nn.Linear(num_features, num_classes)
            
        self.model = self.model.to(self.device)
        
        self.createLossFunction(loss_function)
        self.createOptimizer(optimizer_name, learning_rate)

##################################### Testing Code ############################################
if __name__ == "__main__":
    print("Test 1: Loading training data")
    dm = DataManager()
    current_folder = os.getcwd()
    
    dm.LoadTrainingData(folderName=current_folder+"/../data/", csvFileName="Training_set.csv", numFiles=50)
    dm.RemoveMissingData()
    
    print("Test 2: Creating model")
    num_classes = len(dm.TrainingData["label"].unique())
    model_dir = os.path.join(current_folder, "../models")
    os.makedirs(model_dir, exist_ok=True)
    model = EfficientNet(num_classes=num_classes, variant='b0', model_dir=model_dir)
    
    print("Test 3: Finding optimal configuration")
    # test train/eval/test
    train_val_df, test_df = model.splitData(dm.TrainingData, validation_split=0.2)
    
    print(f"Data split ratio - Train+Val: {len(train_val_df)}/{len(dm.TrainingData)}, Test: {len(test_df)}/{len(dm.TrainingData)}")

    best_config, configs = model.findBestConfig(
        df=train_val_df,
        validation_split=0.15,
        epochs=3,
        batch_size=8,
        max_configs=2,
    )

    #print("\n Configs list test: ")
    #for config in configs:
    #    print(config)
    
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
    )
    model.train(train_val_df, epochs=3, batch_size=8, save_interval=2, save_best=False, load_best=False)
    
    # uncomment to test predict but maybe dont as it take long time. This is just test, we will use the model eval file to do eval so code bellow 
    # here can be removed but maybe keep it uncomment for now for sake of testing
    """
    print("\nTest 5: Testing model predictions...")
    # load testing data
    dm.LoadTestData(folderName=current_folder+"/../data/", csvFileName="Testing_set.csv")

    model.setupClassMapping(dm.TrainingData)

    #start_epoch, best_loss = model.loadModel("EfficientNet")
    #print(f"\nLoaded model from epoch {start_epoch} with best loss: {best_loss:.4f}")
    model.model.eval()

    test_predictions = model.predict(dm.TestData["image"])

    result_df = pd.DataFrame({
        'image_id': range(len(dm.TestData)),
        'predicted_label': test_predictions
    })
    result_df.to_csv("test_predictions.csv", index=False)
    print("Predictions saved to test_predictions.csv")
    

    print("\nTest 6: Testing model predictions on a batch of training data...")
    num_images = 50
    test_batch = dm.TrainingData.sample(num_images)
    predicted_labels = model.predict(test_batch["image"])
    
    max_cols = 5
    cols = min(num_images, max_cols)
    rows = math.ceil(num_images / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    
    if rows == 1:
        axes = np.array([axes])
    
    if num_images == 1:
        axes = np.array([[axes]])
    
    axes_flat = axes.flatten()
    
    for i in range(num_images):
        ax = axes_flat[i]
        
        img_obj = test_batch["image"].iloc[i]
        img_array = img_obj.image
        
        ax.imshow(img_array)
        true_label = test_batch["label"].iloc[i]
        pred_label = predicted_labels[i]
        
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        ax.axis('off')
    
    for i in range(num_images, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Model Predictions on {num_images} Test Images", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()

    correct = sum(1 for a, p in zip(test_batch["label"], predicted_labels) if a == p)
    accuracy = 100 * correct / len(test_batch["label"])
    print(f"Batch test accuracy: {accuracy:.2f}%")
    
    print("Test 7: Predict/Actual of Whole Training Data + Accuracy")
    predicted_labels = model.predict(dm.TrainingData["image"])
    correct = sum(1 for a, p in zip(dm.TrainingData["label"], predicted_labels) if a == p)
    accuracy = 100 * correct / len(dm.TrainingData["label"])
    print(f"Batch test accuracy: {accuracy:.2f}%")
    
    print("Done")
    print("="*50)
    """