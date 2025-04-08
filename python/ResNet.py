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


class ResNet(Model):
    def __init__(self, num_classes, variant='50', model_dir="models"):
        super().__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.class_to_idx = None
        self.idx_to_class = None
        
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"Loading ResNet-{variant} model...")
        
        if variant == '18':  
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif variant == '34': 
            self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif variant == '50': 
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif variant == '101': 
            self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        elif variant == '152':  
            self.model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),                                              
            transforms.ToTensor(),                           
            transforms.Normalize(                            
                mean=[0.485, 0.456, 0.406],                  
                std=[0.229, 0.224, 0.225]                    
            )
        ])
            

        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
        self.model = self.model.to(self.device)
        
        # Optimizer for training
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model_dir = model_dir
        self.model_manager = ModelManager(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            model_dir=self.model_dir
        )

    ################################ Data Processing Methods ###################################
    def PreprocessImages(self, image):
        if isinstance(image, DataFrameImage):
            image_array = image.image
        else:
            image_array = image
            
        if image_array.ndim == 2:
            image_array = np.stack([image_array, image_array, image_array], axis=2)
            
        return self.transform(image_array)
    
    def SetupClassMapping(self, df):
        unique_classes = sorted(df["label"].unique())
        
        self.class_to_idx = {}  
        self.idx_to_class = {} 
        
        for idx, class_name in enumerate(unique_classes):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
        
        print(f"Set up mapping for {len(unique_classes)} classes: {unique_classes}")

    def PrepareBatch(self, dataframe, batch_indices):
        batch_data = dataframe.iloc[batch_indices]
        
        image_tensors = []
        for img in batch_data["image"]:
            tensor = self.PreprocessImages(img)
            image_tensors.append(tensor)
        
        image_batch = torch.stack(image_tensors).to(self.device)
        
        if self.class_to_idx is None:
            raise ValueError("Class mapping not initialized. Do SetupClassMapping() before prepare batches.")
        
        label_indices = []

        for label in batch_data["label"]:
            index = self.class_to_idx[label]
            label_indices.append(index)

        label_tensor = torch.tensor(label_indices)
        label_tensor = label_tensor.to(self.device)
        
        return image_batch, label_tensor

    def Preprocess(self, df):
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
    
    ################################ Training Methods #####################################
    def Train(self, df, epochs=10, batch_size=32, save_interval=1, save_best=True, load_best=True):
        self.SetupTraining(df)
        n_samples = len(df)
        indices = np.arange(n_samples)
        num_batches = (n_samples + batch_size - 1) // batch_size

        start_epoch = 0
        best_loss = float('inf')

        if load_best:
            start_epoch, best_loss = self.LoadModel("ResNet")
            if start_epoch > 0:
                print(f"Resuming training from epoch {start_epoch+1} with best loss: {best_loss:.4f}")
        else:
            print("Start from beginning")

        print(f"Starting training: {epochs} epochs, {n_samples} samples, {num_batches} batches per epoch")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("Progress: [", end="")
            
            epoch_start_time = time.time()
            epoch_loss, epoch_acc = self.TrainEpoch(df, indices, batch_size, num_batches)
            print("]")

            epoch_time = time.time() - epoch_start_time
            self.DisplayEpoch(epoch, epochs, epoch_time, epoch_loss, epoch_acc)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if save_best:
                    self.SaveModel(epoch+1, epoch_loss, "ResNet", best=True)
                    print(f"New best model saved with loss: {epoch_loss:.4f}")
                else:
                    print(f"New best loss: {epoch_loss:.4f} (model not saved)")
            elif (epoch + 1) % save_interval == 0:
                self.SaveModel(epoch+1, epoch_loss, "ResNet", best=False)
                print(f"No new best model. Checkpoint saved at epoch {epoch+1}")

    def SetupTraining(self, df):
        if self.class_to_idx is None:
            self.SetupClassMapping(df)
        self.model.train()

    def TrainEpoch(self, df, indices, batch_size, num_batches):
        n_samples = len(indices)
        running_loss = 0.0
        correct = 0
        total = 0
        
        np.random.shuffle(indices)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            inputs, label_indices = self.PrepareBatch(df, batch_indices)
            
            batch_loss, batch_correct, batch_total = self.ProcessBatch(inputs, label_indices)
            
            running_loss += batch_loss
            correct += batch_correct
            total += batch_total

            current_batch = (i // batch_size) + 1
            self.DisplayProcess(current_batch, num_batches)
        
        epoch_loss = running_loss / num_batches
        epoch_acc = 100 * correct / total if total > 0 else 0
        
        return epoch_loss, epoch_acc

    def ProcessBatch(self, inputs, label_indices):
        self.optimizer.zero_grad()
        
        outputs = self.model(inputs)
        loss = self.criterion(outputs, label_indices)
        loss.backward()
        self.optimizer.step()
        
        batch_loss = loss.item()
        _, predicted = torch.max(outputs.data, 1)
        batch_total = label_indices.size(0)
        batch_correct = (predicted == label_indices).sum().item()
        
        return batch_loss, batch_correct, batch_total

    def DisplayProcess(self, current_batch, num_batches):
        if current_batch % max(1, num_batches // 20) == 0 or current_batch == num_batches:
            progress = int(30 * current_batch / num_batches)
            print("=" * (progress - len(str(current_batch)) - len(str(num_batches)) - 3), end="")
            print(f" {current_batch}/{num_batches} ", end="", flush=True)

    def DisplayEpoch(self, epoch, epochs, epoch_time, epoch_loss, epoch_acc):
        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.1f}s - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
        remaining_epochs = epochs - (epoch + 1)
        if remaining_epochs > 0:
            estimated_time = epoch_time * remaining_epochs
            hours, remainder = divmod(estimated_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Estimated remaining time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    ################################ Prediction Methods ##################################
    def Predict(self, image=None):
        if image is None:
            return None
            
        self.PrepareModel()
        
        if isinstance(image, pd.Series):
            return self.PredictBatch(image)
        else:
            return self.PredictSingle(image)

    def PrepareModel(self):
        self.model.eval()
        
    def PredictSingle(self, image):
        with torch.no_grad():
            img_tensor = self.PreprocessImages(image)  
            img_tensor = img_tensor.unsqueeze(0) 
            img_tensor = img_tensor.to(self.device)  
            
            outputs = self.model(img_tensor)
            
            _, predicted_idx = torch.max(outputs, 1)
            
            predicted_label = self.idx_to_class[predicted_idx.item()]

            print(f"Model Predicted: {predicted_label}")
            return predicted_label

    def PredictBatch(self, images):
        predictions = []
        with torch.no_grad():
            for img in images:
                img_tensor = self.PreprocessImages(img)  
                img_tensor = img_tensor.unsqueeze(0) 
                img_tensor = img_tensor.to(self.device) 
                
                outputs = self.model(img_tensor)
                
                _, predicted_idx = torch.max(outputs, 1)
                
                predicted_label = self.idx_to_class[predicted_idx.item()]
                predictions.append(predicted_label)
                
        return predictions
    
    ################################ Save/Load Models Methods ##################################
    def SaveModel(self, epoch, loss, model_name="ResNet", best=True):
        if best:
            self.model_manager.save_best(epoch, loss, model_name)
        else:
            self.model_manager.save(epoch, loss, model_name)

    def LoadModel(self, model_name="ResNet"):
        return self.model_manager.load(best_only=True, model_name=model_name)

##################################### Testing Code ############################################
if __name__ == "__main__":
    print("Test 1: Loading training data...")
    dm = DataManager()
    current_folder = os.getcwd()
    dm.LoadTrainingData(folderName=current_folder+"/../data/", csvFileName="Training_set.csv", numFiles=100)
    
    print("Test 2: Preprocessing data...")
    dm.RemoveMissingData()
    
    num_classes = len(dm.TrainingData["label"].unique())
    
    # Create test folder to save best model to for save/load model test
    model_dir = os.path.join(current_folder, "../models")
    os.makedirs(model_dir, exist_ok=True)
    
    print("Test 3: Initializing ResNet model...")
    model = ResNet(num_classes=num_classes, variant='50', model_dir=model_dir)
    model.Preprocess(dm.TrainingData)
    
    print("Test 4: Training model...")
    model.Train(dm.TrainingData, epochs=2, batch_size=8, save_interval=2, save_best=False, load_best=False)
    
    print("Test 5: Testing batch prediction...")
    test_batch = dm.TrainingData.sample(10)
    test_images = test_batch["image"]
    test_labels = test_batch["label"]
    predicted_labels = model.Predict(test_images)

    print("\nBatch Prediction Results:")
    print("-------------------------")
    for i, (actual, predicted) in enumerate(zip(test_labels, predicted_labels)):
        status = "v" if actual == predicted else "x"
        print(f"Sample {i+1}: Actual: {actual}, Predicted: {predicted} {status}")

    # Calculate accuracy
    correct = sum(1 for a, p in zip(test_labels, predicted_labels) if a == p)
    accuracy = 100 * correct / len(test_labels)
    print(f"\nBatch accuracy: {accuracy:.2f}%")