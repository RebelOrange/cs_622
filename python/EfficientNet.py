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
class EfficientNet(Model):
    """
    class EfficientNet:
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate):
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
    """
    def __init__(self, num_classes, variant='b0', model_dir="models"):
        super().__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.class_to_idx = None
        self.idx_to_class = None
        
        # Loss function for training
        # might need to change to determine the best loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # pre-traned models.... which one to use?
        # b0/1 are like baseline 
        # b4/5 are like medium
        # b6/7 are like large
        print(f"Loading EfficientNet-{variant} model...")
        # Replace the deprecated pretrained=True with weights parameter

        if variant == 'b0':  
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif variant == 'b1': 
            self.model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        elif variant == 'b4': 
            self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        elif variant == 'b5': 
            self.model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        elif variant == 'b6':  
            self.model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1)
        elif variant == 'b7': 
            self.model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")
        
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
            
        num_features = self.model.classifier[1].in_features
        # might need to adjust this as we can add like ReLU or other activation functions to the model
        # to make it better
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.2),  # prevent overfitting? but then maybe we can add seperate func to deal with overfit? as it might be better to have it as a seperate func
            nn.Linear(num_features, num_classes)  # classification layer
        )
        
        self.model = self.model.to(self.device)
        
        # optimizer for training
        # might need to change to determine the best optimizer but use Adam for now
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
        # Extract image array from DataFrameImage if needed
        if isinstance(image, DataFrameImage):
            image_array = image.image
        else:
            image_array = image
            
        # extra check? bc efficientnet use color img so if we have grayscale we need to convert it?
        # could remove this if we do the code well in main
        if image_array.ndim == 2:  # Check if grayscale (2D array)
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
                
            ## add something more? 
                
        except Exception as e:
            raise ValueError(f"Error examining images: {str(e)}")
        
        return df
    
    ################################ Training Methods #####################################
    def Train(self, df, epochs=10, batch_size=32, save_interval=1):
        self.SetupTraining(df)
        n_samples = len(df)
        indices = np.arange(n_samples)
        num_batches = (n_samples + batch_size - 1) // batch_size

        start_epoch, best_loss = self.LoadModel("EfficientNet")
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch+1} with best loss: {best_loss:.4f}")
        else:
            best_loss = float('inf')

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
                self.SaveModel(epoch+1, epoch_loss, "EfficientNet", best=True)
                print(f"New best model saved with loss: {epoch_loss:.4f}")
            elif (epoch + 1) % save_interval == 0:
                self.SaveModel(epoch+1, epoch_loss, "EfficientNet", best=False)
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

    ########## Need to add more here ##########
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
    def SaveModel(self, epoch, loss, model_name="EfficientNet", best=True):
        if best:
            self.model_manager.save_best(epoch, loss, model_name)
        else:
            self.model_manager.save(epoch, loss, model_name)

    def LoadModel(self, model_name="EfficientNet"):
        return self.model_manager.load(best_only=True, model_name=model_name)

##################################### Testing Code ############################################
if __name__ == "__main__":
    print("Test 1: Loading training data...")
    dm = DataManager()
    current_folder = os.getcwd()
    dm.LoadTrainingData(folderName=current_folder+"/../data/", csvFileName="Training_set.csv", numFiles=100)
    
    print("Test 2: Preprocessing data...")
    dm.RemoveMissingData()
    # maybe using tranformations from pytorch could be better?? if so maybe we can do that in preprocess()
    #dm.ResizeImages(TargetSize=(224, 224))
    #dm.NormalizeImages()
    
    num_classes = len(dm.TrainingData["label"].unique())
    # dm.PrintStats()

    # create test folder to save best model to for save/load model test
    model_dir = os.path.join(current_folder, "../models")
    os.makedirs(model_dir, exist_ok=True)
    
    print("Test 3: Initializing EfficientNet model...")
    model = EfficientNet(num_classes=num_classes, variant='b0', model_dir=model_dir)
    model.Preprocess(dm.TrainingData) # preprocess the data again

    # might need to do split data or cross validation to get better results? 
    
    print("Test 4: Training model...")
    model.Train(dm.TrainingData, epochs=2, batch_size=8, save_interval=2)
    
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